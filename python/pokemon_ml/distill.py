"""Distillation: a network taught to play like another one.

A bigger network cannot be trained from scratch on round 1's search data: 50,000
games are too few, and medium and large overfit them
(docs/expert-iteration-round1-review.md). It can learn from a trained network
instead, on as many positions as there is time for, since no search is needed. The
teacher (`--teacher`, normally update 15342) is asked what it would play and how it
rates each position, and the student (fresh at `--model-size`, with the teacher's
feature settings) is trained towards that:

    policy   KL(teacher || student) over the whole choice, exactly: Team Preview's
             choices, or in battle the first position's action and then the second
             position's given every legal first one
    value    0.5 · (student's value - teacher's value)²

Positions come from games played as training goes, never stored on disk. Each side
of each game is played by the teacher or by the student (`--student-play`, the
student's share, drawn per side and kept for the game): the teacher's games show the
student good play, the student's show it the positions its own mistakes lead to,
where it needs the teacher's answers too. A buffer of the latest `--buffer`
positions is sampled for each step, and every position is used about `--reuse`
times.

Held-out positions, from games the teacher plays against itself on another seed,
measure progress every `--eval-every` steps in `log.jsonl`:

    policy_kl     KL(teacher || student), nats (0: the same choices)
    agreement     how often the student's likeliest choice is the teacher's
    value_mse     the student's value against the teacher's

`best.pt` is the student with the lowest held-out loss and `checkpoint.pt` the last.
Both load wherever a checkpoint does, including `pokemon_ml.train
--init-checkpoint` to go on with self-play. Whether distillation worked is decided
by play: the student against the teacher, about 50%.

    python -m pokemon_ml.distill --teacher runs/x/checkpoint.pt --model-size large \\
        --pool teams/split/train.json --out runs/distill/large --hours 3
"""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .env import Env, N_ACTIONS, OBS_F, OBS_I, OBS_M, PREVIEW, part


def policy(model, f, i, mask):
    """The model's choice, as `Model.act` makes it: Team Preview's log-probabilities
    [B, N_PREVIEW]; the first position's [B, N_ACTIONS] and its legal actions; the
    second position's [P, N_ACTIONS] for each legal first action (row `rows[p]`,
    first action `firsts[p]`) and their legal actions; and the value [B]. Rows that
    are Team Preview have no battle pairs."""
    field, mons, acts, move_tokens = model.encode(f, i)
    b = len(f)
    preview = part(i, "i", "info")[:, 0] == PREVIEW
    lg0, m0 = model._masked(model.position_logits(mons, acts, move_tokens, 0), mask[:, :N_ACTIONS])
    battle = m0 & ~preview[:, None]
    rows, firsts = battle.nonzero(as_tuple=True)
    mask1 = mask[:, N_ACTIONS:].reshape(b, N_ACTIONS, N_ACTIONS)[rows, firsts]
    lg1 = model.position_logits(mons[rows], acts[rows], move_tokens[rows], 1, firsts)
    lg1, m1 = model._masked(lg1, mask1)
    return dict(
        preview=preview,
        logp_preview=F.log_softmax(model.preview_logits(mons), -1),
        logp0=F.log_softmax(lg0, -1), m0=m0,
        rows=rows, firsts=firsts,
        logp1=F.log_softmax(lg1, -1), m1=m1,
        value=model.value(field),
    )


def divergence(teacher, student):
    """KL(teacher || student) per row, over the whole choice. Both from `policy` on the
    same observations, so their pairs line up."""
    kl_preview = (teacher["logp_preview"].exp() * (teacher["logp_preview"] - student["logp_preview"])).sum(-1)
    gap0 = torch.where(teacher["m0"], teacher["logp0"] - student["logp0"], torch.zeros_like(student["logp0"]))
    kl0 = (teacher["logp0"].exp() * gap0).sum(-1)
    gap1 = torch.where(teacher["m1"], teacher["logp1"] - student["logp1"], torch.zeros_like(student["logp1"]))
    kl1 = (teacher["logp1"].exp() * gap1).sum(-1)  # per legal first action
    rows, firsts = teacher["rows"], teacher["firsts"]
    weight = teacher["logp0"][rows, firsts].exp()  # how often the teacher plays that first action
    kl_battle = kl0.index_add(0, rows, weight * kl1)
    return torch.where(teacher["preview"], kl_preview, kl_battle)


def favourite(p):
    """The likeliest choice per row: a Team Preview choice, or a battle pair as a0 · N_ACTIONS + a1."""
    b = len(p["logp0"])
    joint = torch.full((b, N_ACTIONS * N_ACTIONS), -math.inf, device=p["logp0"].device)
    pair = p["logp0"][p["rows"], p["firsts"]][:, None] + p["logp1"].masked_fill(~p["m1"], -math.inf)
    index = p["firsts"][:, None] * N_ACTIONS + torch.arange(N_ACTIONS, device=pair.device)
    joint[p["rows"][:, None].expand_as(index), index] = pair
    return torch.where(p["preview"], p["logp_preview"].argmax(-1), joint.argmax(-1))


def losses(teacher, student, f, i, mask, value_coef=0.5):
    with torch.no_grad():
        t = policy(teacher, f, i, mask)
    s = policy(student, f, i, mask)
    kl = divergence(t, s)
    value = 0.5 * (s["value"] - t["value"]).pow(2)
    loss = kl.mean() + value_coef * value.mean()
    return loss, dict(policy_kl=kl.detach(), value_mse=2 * value.detach(), t=t, s=s)


class Buffer:
    """The latest `size` positions, on the device, sampled uniformly."""

    def __init__(self, size, device):
        self.f = torch.zeros(size, OBS_F, device=device)
        self.i = torch.zeros(size, OBS_I, dtype=torch.int16, device=device)
        self.mask = torch.zeros(size, OBS_M, dtype=torch.uint8, device=device)
        self.size, self.at, self.count = size, 0, 0

    def add(self, f, i, mask):
        n = len(f)
        index = (self.at + torch.arange(n, device=f.device)) % self.size
        self.f[index], self.i[index], self.mask[index] = f, i, mask
        self.at = (self.at + n) % self.size
        self.count = min(self.size, self.count + n)

    def sample(self, n, generator):
        index = torch.randint(self.count, (n,), device=self.f.device, generator=generator)
        return self.f[index], self.i[index], self.mask[index]


class Games:
    """Games in which each side is played by the teacher or the student, kept for the game."""

    def __init__(self, env, teacher, student, student_play, rng, device):
        self.env, self.teacher, self.student, self.device = env, teacher, student, device
        self.share, self.rng = student_play, rng
        self.by_student = rng.random(2 * env.envs) < student_play
        self.games = self.positions = 0

    @torch.no_grad()
    def step(self):
        """Plays one move in every game. Returns the positions with a choice to make."""
        env = self.env
        f, i, mask = (torch.from_numpy(x).to(self.device) for x in (env.f, env.i, env.mask))
        actions = torch.zeros(len(f), 2, dtype=torch.long, device=self.device)
        mine = torch.from_numpy(self.by_student).to(self.device)
        for net, rows in ((self.student, mine), (self.teacher, ~mine)):
            if rows.any():
                was = net.training
                net.eval()
                actions[rows] = net.act(f[rows], i[rows], mask[rows])[0]
                net.train(was)
        choice = part(i, "i", "info")[:, 3] == 1
        env.step(actions.cpu().numpy())
        done = np.repeat(env.done.astype(bool), 2)
        self.by_student[done] = self.rng.random(int(done.sum())) < self.share
        self.games += int(env.done.sum())
        self.positions += int(choice.sum())
        return f[choice], i[choice], mask[choice]


@torch.no_grad()
def held_out_positions(env, teacher, count, device):
    """`count` positions with a choice to make, from games the teacher plays against itself."""
    games = Games(env, teacher, teacher, 0.0, np.random.default_rng(0), device)
    parts = []
    while sum(len(p[0]) for p in parts) < count:
        parts.append(games.step())
    f, i, mask = (torch.cat(x)[:count] for x in zip(*parts))
    return f, i, mask


@torch.no_grad()
def evaluate(teacher, student, held, batch=1024, value_coef=0.5):
    student.eval()
    kl = agree = mse = 0.0
    n = len(held[0])
    for at in range(0, n, batch):
        f, i, mask = (x[at : at + batch] for x in held)
        _, s = losses(teacher, student, f, i, mask, value_coef)
        kl += float(s["policy_kl"].sum())
        mse += float(s["value_mse"].sum())
        agree += float((favourite(s["t"]) == favourite(s["s"])).sum())
    return dict(policy_kl=kl / n, agreement=agree / n, value_mse=mse / n, positions=n)


def save(path, model, extra):
    from .expert_train import save as save_checkpoint

    save_checkpoint(path, model, extra)


def train(teacher, student, env, held, out, device, steps, lr=3e-4, batch=1024, buffer=65536, reuse=2.0,
          student_play=0.5, warmup=500, eval_every=1000, hours=None, value_coef=0.5, weight_decay=0.0,
          max_grad_norm=1.0, seed=0, extra=None):
    """Distils `teacher` into `student` on positions from `env` for `steps` steps (or `hours`),
    measuring on `held` (from `held_out_positions`). Writes the log and checkpoints to `out`;
    returns the summary."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    teacher.eval().requires_grad_(False)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    games = Games(env, teacher, student, student_play, rng, device)
    memory = Buffer(buffer, device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = lambda step: min(1.0, (step + 1) / max(1, warmup)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    log = (out / "log.jsonl").open("w")
    record = lambda **row: (log.write(json.dumps(row) + "\n"), log.flush())
    start = time.time()
    baseline = evaluate(teacher, student, held, value_coef=value_coef)
    record(step=0, held_out=baseline, baseline=True, steps=steps)
    loss_of = lambda m: m["policy_kl"] + value_coef * 0.5 * m["value_mse"]
    best = (loss_of(baseline), 0)
    save(out / "best.pt", student, dict(extra or {}, step=0, held_out=baseline))
    fresh, step, running = 0.0, 0, []
    while memory.count < min(buffer, 4 * batch):  # a first fill, so early batches are not one game's
        memory.add(*games.step())
    while step < steps and not (hours and time.time() - start > hours * 3600):
        while fresh < batch / reuse:
            f, i, mask = games.step()
            memory.add(f, i, mask)
            fresh += len(f)
        fresh -= batch / reuse
        student.train()
        loss, s = losses(teacher, student, *memory.sample(batch, generator), value_coef)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        step += 1
        running.append((float(s["policy_kl"].mean()), float(s["value_mse"].mean()), float(norm)))
        if step % eval_every == 0 or step == steps:
            held_out = evaluate(teacher, student, held, value_coef=value_coef)
            k, v, g = np.mean(running, 0)
            running = []
            record(step=step, lr=scheduler.get_last_lr()[0], train_policy_kl=float(k), train_value_mse=float(v),
                   grad_norm=float(g), held_out=held_out, games=games.games, positions=games.positions,
                   seconds=time.time() - start)
            if loss_of(held_out) < best[0]:
                best = (loss_of(held_out), step)
                save(out / "best.pt", student, dict(extra or {}, step=step, held_out=held_out))
    final = evaluate(teacher, student, held, value_coef=value_coef)
    save(out / "checkpoint.pt", student, dict(extra or {}, step=step, held_out=final))
    summary = dict(steps=step, best_step=best[1], baseline=baseline, final=final, games=games.games,
                   positions=games.positions, seconds=time.time() - start)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.close()
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", required=True, help="the checkpoint to learn from")
    p.add_argument("--model-size", choices=("small", "medium", "large"), help="a fresh student of this size")
    p.add_argument("--init", help="or a student checkpoint to go on from")
    p.add_argument("--pool", required=True, help="the training pool (never the evaluation one)")
    p.add_argument("--out", required=True, help="a new directory for the log and checkpoints")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--hours", type=float, help="stop after this long, whatever --steps says")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--buffer", type=int, default=65536, help="positions kept to sample from")
    p.add_argument("--reuse", type=float, default=2.0, help="how many times each position is trained on, on average")
    p.add_argument("--student-play", type=float, default=0.5, help="the share of sides the student plays")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--held-out", type=int, default=20000, help="held-out positions")
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--open-sheets", type=float, default=0.5)
    p.add_argument("--no-vary", action="store_true", help="play the pool's teams as they are (training varies them)")
    p.add_argument("--max-turns", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    if bool(a.init) == bool(a.model_size):
        p.error("give exactly one of --model-size and --init")
    if not 0 <= a.student_play <= 1 or a.reuse <= 0:
        p.error("--student-play is a share in [0, 1] and --reuse is positive")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        p.error(f"{out} is not empty: give a new directory")
    from .checkpoint import file_hash
    from .evaluation import load
    from .model import PRESETS, Config, Model
    from .train import pick_device

    device = pick_device(a.device)
    torch.manual_seed(a.seed)
    teacher = load(a.teacher, device)
    if a.init:
        student = load(a.init, device)
    else:  # the teacher's features and heads, at the student's size
        student = Model(Config(**{**asdict(teacher.config), **PRESETS[a.model_size]})).to(device)
    if teacher.config.history or student.config.history:
        p.error("networks that read decision history cannot be distilled yet")
    env = Env(a.pool, a.envs, a.seed, a.open_sheets, not a.no_vary, max_turns=a.max_turns, threads=a.threads)
    held_env = Env(a.pool, a.envs, a.seed + 7_000_003, a.open_sheets, not a.no_vary, max_turns=a.max_turns,
                   threads=a.threads)
    held = held_out_positions(held_env, teacher, a.held_out, device)
    del held_env
    extra = dict(
        teacher=a.teacher,
        teacher_sha256=file_hash(a.teacher),
        init=a.init,
        model_size=a.model_size,
        pool=a.pool,
        pool_sha256=file_hash(a.pool),
        settings={k: v for k, v in vars(a).items() if k != "out"},
    )
    out.mkdir(parents=True, exist_ok=True)
    parameters = sum(x.numel() for x in student.parameters())
    (out / "config.json").write_text(json.dumps(dict(extra, parameters=parameters, model=asdict(student.config)),
                                                indent=2) + "\n")
    print(f"student: {parameters:,} parameters; teacher: {sum(x.numel() for x in teacher.parameters()):,}", flush=True)
    summary = train(teacher, student, env, held, out, device, a.steps, lr=a.lr, batch=a.batch, buffer=a.buffer,
                    reuse=a.reuse, student_play=a.student_play, warmup=a.warmup, eval_every=a.eval_every,
                    hours=a.hours, value_coef=a.value_coef, weight_decay=a.weight_decay, seed=a.seed, extra=extra)
    print(json.dumps({k: summary[k] for k in ("steps", "best_step", "baseline", "final", "games", "positions")}))


if __name__ == "__main__":
    main()
