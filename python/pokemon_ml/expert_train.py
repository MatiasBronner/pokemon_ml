"""Expert iteration's training step: a network taught by its own search.

Reads the shards `pokemon_ml.expert_data` writes and trains a network on them:

    policy   towards the search's mixed strategy, on decisions search made:
             cross-entropy, -Σ target(pair) · log p(pair), over the pairs searched
    value    towards how each game ended for the learner, on every decision:
             0.5 · (value - outcome)², as PPO trains it

The network starts from `--init` (normally the checkpoint that generated the
data), or fresh at `--model-size` to test a bigger one. Whole games are held out
for validation (`--holdout`), so no position of a validation game is trained
on. Every `--eval-every` steps and at the end the log (`log.jsonl`) records, on
the held-out games:

    policy_ce        the policy's cross-entropy against the search's strategy
    policy_kl        that less the strategy's own entropy: how far the policy is
                     from the search (0: exactly the search). Cross-entropy alone
                     cannot fall below the entropy, since the search mixes
    agreement        how often the policy's likeliest searched pair is the
                     search's likeliest
    value_mse        the value's squared error against outcomes
    value_explained  the share of the outcomes' variance the value explains

and the same for the starting network, as a baseline. `checkpoint.pt` is the
network at the end and `best.pt` the one with the lowest held-out loss, both
loadable wherever a checkpoint is (`pokemon_ml.evaluation`, `pokemon_ml.search`).

Whether the round worked is not decided here but by play: the new network with
search against the old one with search (`pokemon_ml.search --opponent OLD
--opponent-search`), clearly above 50%.

    python -m pokemon_ml.expert_train --data runs/expert/round1 \\
        --init runs/x/checkpoint.pt --out runs/expert/round1-lr1e-4 --lr 1e-4
"""

import argparse
import json
import math
import time
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .env import N_ACTIONS

FIELDS = ("f", "i", "mask", "searched", "candidates", "target", "outcome", "game")


def load_data(dirs, max_rows=0):
    """Every shard of every directory in `dirs`, as NumPy arrays, with `game` made
    unique across directories. Returns the arrays and the manifests."""
    parts, manifests = {k: [] for k in FIELDS}, []
    for d, path in enumerate(map(Path, dirs)):
        manifest = json.loads((path / "manifest.json").read_text())
        manifests.append(dict(path=str(path), **{k: manifest[k] for k in ("games", "rows", "checkpoint_sha256")}))
        for shard in manifest["shards"]:
            with np.load(path / shard["file"]) as data:
                for k in FIELDS:
                    a = data[k]
                    if k == "game":
                        a = a.astype(np.int64) + (d << 40)
                    parts[k].append(a)
    widths = {a.shape[1] for a in parts["candidates"]}
    if len(widths) > 1:  # directories searched with different numbers of candidates
        width = max(widths)
        for k, pad in (("candidates", -1), ("target", 0)):
            parts[k] = [np.pad(a, [(0, 0), (0, width - a.shape[1])] + [(0, 0)] * (a.ndim - 2), constant_values=pad)
                        for a in parts[k]]
    data = {k: np.concatenate(v) for k, v in parts.items()}
    if max_rows and len(data["f"]) > max_rows:
        data = {k: v[:max_rows] for k, v in data.items()}
    return data, manifests


def split_games(games, holdout, seed=0):
    """A boolean per row, True for rows of held-out games: whole games, chosen by a hash of their id."""
    unique = np.unique(games)
    keys = np.array([zlib.crc32(f"{seed}:{g}".encode()) for g in unique], np.uint64)
    held = set(unique[keys < np.uint64(holdout * 2**32)].tolist())
    return np.array([g in held for g in games.tolist()])


def pair_log_probs(model, f, i, mask, candidates):
    """log p(pair) under `model` for each candidate pair, [B, K], the way `Model.act`
    chooses (the second position given the first). Padding (-1) gets an arbitrary
    value: weigh it by zero."""
    _, mons, acts, move_tokens = model.encode(f, i)
    b, k = candidates.shape[:2]
    mask0 = mask[:, :N_ACTIONS].bool()
    mask1 = mask[:, N_ACTIONS:].reshape(b, N_ACTIONS, N_ACTIONS).bool()
    lg0, _ = model._masked(model.position_logits(mons, acts, move_tokens, 0), mask0)
    logp0 = F.log_softmax(lg0, -1)
    a0, a1 = candidates[..., 0].clamp(min=0).long(), candidates[..., 1].clamp(min=0).long()
    first = a0.reshape(-1)
    rep = lambda x: x.repeat_interleave(k, 0)
    lg1 = model.position_logits(rep(mons), rep(acts), rep(move_tokens), 1, first)
    rows = torch.arange(b, device=f.device).repeat_interleave(k)
    lg1, _ = model._masked(lg1, mask1[rows, first])
    logp1 = F.log_softmax(lg1, -1).gather(1, a1.reshape(-1, 1)).reshape(b, k)
    return logp0.gather(1, a0) + logp1


def losses(model, batch):
    """The policy and value losses on a batch (a dict of tensors), and what to log."""
    f, i, mask = batch["f"], batch["i"], batch["mask"]
    searched, target, outcome = batch["searched"], batch["target"], batch["outcome"]
    field = model.encode(f, i)[0]
    value = model.value(field)
    value_loss = 0.5 * (value - outcome).pow(2).mean()
    stats = dict(value_loss=value_loss.detach(), value=value.detach())
    if searched.any():
        s = searched.nonzero().squeeze(1)
        logp = pair_log_probs(model, f[s], i[s], mask[s], batch["candidates"][s])
        t = target[s]
        policy_loss = -(t * torch.where(t > 0, logp, torch.zeros_like(logp))).sum(-1).mean()
        # Cross-entropy cannot fall below the target's own entropy (the search mixes): what is
        # left above it, the KL divergence, is how far the policy is from the search.
        entropy = -(t * torch.where(t > 0, t.clamp_min(1e-12).log(), torch.zeros_like(t))).sum(-1).mean()
        real = batch["candidates"][s][..., 0] >= 0
        agree = (logp.masked_fill(~real, -1e9).argmax(-1) == t.argmax(-1)).float()
        stats.update(policy_loss=policy_loss.detach(), policy_kl=(policy_loss - entropy).detach(),
                     agreement=agree.mean().detach(), searched=len(s))
    else:
        policy_loss = value_loss * 0
        stats.update(policy_loss=policy_loss.detach(), policy_kl=policy_loss.detach(),
                     agreement=torch.tensor(float("nan")), searched=0)
    return policy_loss, value_loss, stats


def to_batch(data, rows, device):
    out = {}
    for k in FIELDS:
        a = data[k][rows]
        t = torch.as_tensor(a)
        if k in ("target", "outcome", "f"):
            t = t.float()
        out[k] = t.to(device, non_blocking=True)
    out["searched"] = out["searched"].bool()
    return out


@torch.no_grad()
def evaluate(model, data, rows, device, batch=2048):
    """Held-out metrics: policy cross-entropy and agreement on searched rows, value error on all."""
    model.eval()
    ce = kl = agree = searched = 0.0
    values, outcomes = [], []
    for start in range(0, len(rows), batch):
        b = to_batch(data, rows[start : start + batch], device)
        p, _, s = losses(model, b)
        if s["searched"]:
            ce += float(p) * s["searched"]
            kl += float(s["policy_kl"]) * s["searched"]
            agree += float(s["agreement"]) * s["searched"]
            searched += s["searched"]
        values.append(s["value"].cpu().numpy())
        outcomes.append(b["outcome"].cpu().numpy())
    v, o = np.concatenate(values), np.concatenate(outcomes)
    mse = float(np.mean((v - o) ** 2))
    return dict(
        policy_ce=ce / searched if searched else None,
        policy_kl=kl / searched if searched else None,
        agreement=agree / searched if searched else None,
        value_mse=mse,
        value_explained=float(1 - mse / np.var(o)) if np.var(o) > 0 else None,
        rows=len(rows),
        searched_rows=int(searched),
    )


def save(path, model, extra):
    from .checkpoint import contract

    state = dict(model=model.state_dict(), model_config=asdict(model.config), contract=contract(), expert=extra)
    torch.save(state, str(path) + ".tmp")
    Path(str(path) + ".tmp").replace(path)


def train(model, data, out, device, lr=1e-4, epochs=2, batch=1024, holdout=0.05, value_coef=0.5,
          policy_coef=1.0, warmup=200, eval_every=500, weight_decay=0.0, max_grad_norm=0.5, seed=0, extra=None):
    """Trains `model` on `data` (from `load_data`), writing logs and checkpoints to `out`. Returns the final metrics."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    held = split_games(data["game"], holdout, seed)
    train_rows, val_rows = np.flatnonzero(~held), np.flatnonzero(held)
    if not len(train_rows) or not len(val_rows):
        raise ValueError("too few games to hold some out: generate more, or change --holdout")
    generator = np.random.default_rng(seed)
    torch.manual_seed(seed)
    steps = max(1, epochs * math.ceil(len(train_rows) / batch))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = lambda step: min(1.0, (step + 1) / max(1, warmup)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    log = (out / "log.jsonl").open("w")
    start = time.time()
    baseline = evaluate(model, data, val_rows, device)
    record = lambda **row: (log.write(json.dumps(row) + "\n"), log.flush())
    record(step=0, held_out=baseline, baseline=True, train_rows=len(train_rows), val_rows=len(val_rows), steps=steps)
    best = (baseline["value_mse"] * value_coef + (baseline["policy_ce"] or 0) * policy_coef, 0)
    save(out / "best.pt", model, dict(extra or {}, step=0, held_out=baseline))
    step, running = 0, []
    for epoch in range(epochs):
        order = generator.permutation(train_rows)
        for at in range(0, len(order), batch):
            model.train()
            b = to_batch(data, np.sort(order[at : at + batch]), device)
            policy_loss, value_loss, s = losses(model, b)
            loss = policy_coef * policy_loss + value_coef * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            running.append((float(s["policy_loss"]), float(s["policy_kl"]), float(s["value_loss"]), float(norm)))
            if step % eval_every == 0 or step == steps:
                held_out = evaluate(model, data, val_rows, device)
                p, k, v, g = np.mean(running, 0)
                running = []
                record(step=step, epoch=epoch, lr=scheduler.get_last_lr()[0], train_policy_loss=float(p),
                       train_policy_kl=float(k), train_value_loss=float(v), grad_norm=float(g), held_out=held_out,
                       seconds=time.time() - start)
                score = held_out["value_mse"] * value_coef + (held_out["policy_ce"] or 0) * policy_coef
                if score < best[0]:
                    best = (score, step)
                    save(out / "best.pt", model, dict(extra or {}, step=step, held_out=held_out))
    final = evaluate(model, data, val_rows, device)
    save(out / "checkpoint.pt", model, dict(extra or {}, step=step, held_out=final))
    summary = dict(steps=step, best_step=best[1], baseline=baseline, final=final, seconds=time.time() - start)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.close()
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", action="append", required=True, help="a directory expert_data wrote; repeatable")
    p.add_argument("--init", help="checkpoint to start from (normally the one that generated the data)")
    p.add_argument("--model-size", choices=("small", "medium", "large"), help="start fresh at this size instead")
    p.add_argument("--out", required=True, help="a new directory for the log and checkpoints")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--holdout", type=float, default=0.05, help="share of games held out for validation")
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--policy-coef", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-rows", type=int, default=0, help="use at most this many rows (0: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)
    if bool(a.init) == bool(a.model_size):
        p.error("give exactly one of --init and --model-size")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        p.error(f"{out} is not empty: give a new directory")
    from .train import pick_device
    from .checkpoint import file_hash
    from .evaluation import load
    from .model import Model, preset

    device = pick_device(a.device)
    torch.manual_seed(a.seed)
    model = load(a.init, device) if a.init else Model(preset(a.model_size)).to(device)
    if model.config.history:
        p.error("networks that read decision history cannot be trained on this data yet")
    data, manifests = load_data(a.data, a.max_rows)
    extra = dict(
        data=manifests,
        init=a.init,
        init_sha256=file_hash(a.init) if a.init else None,
        model_size=a.model_size,
        settings={k: v for k, v in vars(a).items() if k not in ("data", "out")},
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(extra, rows=len(data["f"])), indent=2) + "\n")
    summary = train(model, data, out, device, lr=a.lr, epochs=a.epochs, batch=a.batch, holdout=a.holdout,
                    value_coef=a.value_coef, policy_coef=a.policy_coef, warmup=a.warmup, eval_every=a.eval_every,
                    weight_decay=a.weight_decay, seed=a.seed, extra=extra)
    print(json.dumps({k: summary[k] for k in ("steps", "best_step", "baseline", "final")}))


if __name__ == "__main__":
    main()
