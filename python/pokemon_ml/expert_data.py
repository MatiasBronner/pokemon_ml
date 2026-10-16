"""Games to learn from: a network with search against a league, recorded.

Expert iteration trains a network on what search makes of its positions: the
policy towards the search's mixed strategy, the value towards how games
actually end. This writes that data.

The learner (`--checkpoint`) plays every game with search: by default the
anchored search with exploration (`--explore 2 --noise 0.25`), which on Oscar
lost nothing in strength and lets pairs the policy rates low be tried. Its
opponent is drawn from a league for each game and kept for the whole game, so
the data is not only the learner against itself:

    --opponent self+search:0.3          the learner's own network, with search
    --opponent self:0.2                 the learner's own network, from its policy
    --opponent runs/a/old.pt+search:0.2 a frozen checkpoint, with search
    --opponent runs/a/old.pt:0.2        a frozen checkpoint, from its policy
    --opponent greedy:0.1               a scripted player (random, greedy, lookahead)

The number after the colon is the weight with which that opponent is drawn.
Opponents never explore. Their decisions are not recorded.

For every decision of the learner with more than one legal pair (Team
Preview included), a row of each shard holds:

    f, i, mask     the learner's observation, as the network reads it
    searched       whether search made the decision (battle decisions with
                   open team sheets; Team Preview and closed sheets are left
                   to the policy, which then has no target but the result)
    candidates     the pairs searched, [K, 2], padded with -1
    target         the search's mixed strategy over them, [K], padded with 0:
                   the policy's target
    value          what the search expected (its equilibrium value)
    outcome        how the game ended for the learner: 1, 0 or -1, the
                   value's target
    game, turn, opponent, explored (which candidates exploration added)

Shards are `shard-00000.npz` and so on, compressed, with `manifest.json`
recording the settings, checkpoint hashes and totals. Use the training pool:
data generated on evaluation teams would leak them into training.

    python -m pokemon_ml.expert_data --checkpoint runs/x/checkpoint.pt \\
        --pool teams/split/train.json --out runs/expert/round1 --games 10000 \\
        --opponent self+search:0.3 --opponent self:0.2 --opponent runs/x/old.pt:0.3 --opponent greedy:0.2
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .env import Env, PREVIEW, part
from .evaluation import load
from .search import Config, search_batch

SCRIPTED = ("random", "greedy", "lookahead")


@dataclass
class Opponent:
    """One member of the league. `target`: "self", a scripted player's name, or a checkpoint path."""

    target: str
    search: bool = False
    weight: float = 1.0

    @property
    def name(self):
        return self.target + ("+search" if self.search else "")


def parse_opponent(spec):
    """`TARGET[+search][:weight]`, e.g. `self+search:0.3`, `greedy:0.1`, `runs/a/old.pt`."""
    target, weight = spec, 1.0
    if ":" in spec:
        head, tail = spec.rsplit(":", 1)
        try:
            weight = float(tail)
            target = head
        except ValueError:
            pass
    search = target.endswith("+search")
    if search:
        target = target[: -len("+search")]
    if not target or weight <= 0:
        raise ValueError(f"bad opponent {spec!r}: give TARGET[+search][:weight] with a positive weight")
    if search and target in SCRIPTED:
        raise ValueError(f"bad opponent {spec!r}: a scripted player cannot search")
    return Opponent(target, search, weight)


class Shards:
    """Rows gathered and written out every `size` rows."""

    FIELDS = ("f", "i", "mask", "searched", "candidates", "target", "value", "outcome", "game", "turn",
              "opponent", "explored")

    def __init__(self, out, size, width):
        self.out, self.size, self.width = Path(out), size, width
        self.rows = {k: [] for k in self.FIELDS}
        self.written, self.count, self.searched = [], 0, 0

    def add(self, **row):
        for k in self.FIELDS:
            self.rows[k].append(row[k])
        self.searched += bool(row["searched"])
        if len(self.rows["f"]) >= self.size:
            self.flush()

    def flush(self):
        if not self.rows["f"]:
            return
        name = f"shard-{len(self.written):05d}.npz"
        arrays = {k: np.stack(v) if k in ("f", "i", "mask", "candidates", "target", "explored") else np.asarray(v)
                  for k, v in self.rows.items()}
        np.savez_compressed(self.out / name, **arrays)
        self.written.append(dict(file=name, rows=len(self.rows["f"])))
        self.count += len(self.rows["f"])
        self.rows = {k: [] for k in self.FIELDS}


@torch.no_grad()
def generate(env, model, league, models, games, config, out, device, seed=0, shard_size=20000):
    """Plays `games` games of `model` with search (`config`) against opponents drawn from
    `league` (a list of `Opponent`; `models` maps each checkpoint target to its network,
    "self" to `model`), and writes the learner's decisions to shards in `out`.
    Returns the totals that `manifest.json` records."""
    if model.config.history or any(m.config.history for m in models.values()):
        raise ValueError("search does not support models that read decision history")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    weights = np.array([o.weight for o in league], float)
    weights /= weights.sum()
    width = config.candidates + config.explore
    shards = Shards(out, shard_size, width)
    plain = replace(config, explore=0, explore_replies=0, noise=0.0)  # how opponents search

    n = env.envs
    side = np.arange(n) % 2
    mine, theirs = 2 * np.arange(n) + side, 2 * np.arange(n) + 1 - side
    opponent = rng.choice(len(league), n, p=weights)  # drawn per game, kept for the game
    game_id = np.arange(n)
    next_game = n
    pending = [[] for _ in range(n)]  # this game's rows, waiting for its outcome
    finished, results = 0, {o.name: [0, 0, 0] for o in league}  # wins, ties, losses per opponent
    decisions, start = 0, time.time()

    while finished < games:
        f, i, mask = (torch.as_tensor(x, device=device) for x in (env.f, env.i, env.mask))
        info = part(env.i, "i", "info")
        sheets = env.raw.open_sheets()
        actions = np.zeros((2 * n, 2), np.int64)

        # Opponents: by kind, all games of a kind together.
        for k, member in enumerate(league):
            slots = np.flatnonzero(opponent == k)
            if not len(slots):
                continue
            rows = theirs[slots]
            if member.target in SCRIPTED:
                actions[rows] = env.baseline(member.target)[rows]
                continue
            net = models[member.target]
            actions[rows] = net.act(f[rows], i[rows], mask[rows])[0].cpu().numpy()
            if member.search:
                requests = []
                for g in slots:
                    row = theirs[g]
                    if info[row, 0] != PREVIEW and info[row, 3] == 1 and sheets[g]:
                        decisions += 1
                        requests.append((int(g), int(1 - side[g]), env.f[row], env.i[row], env.mask[row],
                                         seed * 1_000_003 + decisions))
                for (g, *_), (pair, _) in zip(requests, search_batch(net, env, requests, plain, rng, device)):
                    actions[theirs[g]] = pair

        # The learner: its policy everywhere, search where it can.
        actions[mine] = model.act(f[mine], i[mine], mask[mine])[0].cpu().numpy()
        requests = []
        for g in range(n):
            row = mine[g]
            if info[row, 0] != PREVIEW and info[row, 3] == 1 and sheets[g]:
                decisions += 1
                requests.append((g, int(side[g]), env.f[row], env.i[row], env.mask[row], seed * 1_000_003 + decisions))
        found = dict(zip((r[0] for r in requests), search_batch(model, env, requests, config, rng, device)))
        for g in range(n):
            row = mine[g]
            if info[row, 3] != 1:
                continue  # nothing to choose
            candidates = np.full((width, 2), -1, np.int16)
            target = np.zeros(width, np.float32)
            explored = np.zeros(width, bool)
            value, did_search = np.nan, False
            if g in found:
                pair, result = found[g]
                actions[row] = pair
                if result.get("searched"):
                    did_search = True
                    k = len(result["pairs"])
                    candidates[:k] = result["pairs"]
                    target[:k] = result["strategy"]
                    explored[:k] = result["explored"]
                    value = result["value"]
            pending[g].append(dict(
                f=env.f[row].copy(), i=env.i[row].copy(), mask=env.mask[row].copy(), searched=did_search,
                candidates=candidates, target=target, value=np.float32(value), game=game_id[g],
                turn=int(info[row, 2]), opponent=int(opponent[g]), explored=explored,
            ))

        env.step(actions)
        for g in np.flatnonzero(env.done):
            outcome = int(env.reward[g, side[g]])
            if finished < games:
                for row in pending[g]:
                    shards.add(outcome=np.int8(outcome), **row)
                finished += 1
                results[league[opponent[g]].name][{1: 0, 0: 1, -1: 2}[outcome]] += 1
            pending[g] = []
            # The next game in this slot: a new opponent, a new id.
            opponent[g] = rng.choice(len(league), p=weights)
            game_id[g] = next_game
            next_game += 1
    shards.flush()
    return dict(
        games=finished,
        rows=shards.count,
        searched_rows=shards.searched,
        seconds=time.time() - start,
        shards=shards.written,
        results={k: dict(wins=w, ties=t, losses=l) for k, (w, t, l) in results.items()},
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="the learner: the network that searches and is recorded")
    p.add_argument("--opponent", action="append", default=[], help="TARGET[+search][:weight], repeatable")
    p.add_argument("--pool", required=True, help="the training pool (never the evaluation one)")
    p.add_argument("--out", required=True, help="a new directory for the shards and manifest.json")
    p.add_argument("--games", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--open-sheets", type=float, default=1.0, help="decisions with closed sheets are not searched")
    p.add_argument("--no-vary", action="store_true", help="play the pool's teams as they are (training varies them)")
    p.add_argument("--max-turns", type=int, default=100)
    p.add_argument("--shard-size", type=int, default=20000, help="rows per shard")
    defaults = Config(explore=2, noise=0.25)
    for name in ("candidates", "replies", "worlds", "dice", "iterations", "explore", "explore_replies", "chunk"):
        p.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    for name in ("noise", "noise_alpha", "anchor"):
        p.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    a = p.parse_args(argv)
    from .train import pick_device
    from .checkpoint import file_hash

    league = [parse_opponent(s) for s in a.opponent] or [parse_opponent("self+search")]
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        p.error(f"{out} is not empty: give a new directory")
    config = Config(
        candidates=a.candidates, replies=a.replies, worlds=a.worlds, dice=a.dice, iterations=a.iterations,
        explore=a.explore, explore_replies=a.explore_replies, noise=a.noise, noise_alpha=a.noise_alpha,
        anchor=a.anchor, chunk=a.chunk,
    )
    device = pick_device(a.device)
    torch.manual_seed(a.seed)  # the policies sample their moves with torch's generator
    model = load(a.checkpoint, device)
    models = {"self": model}
    for member in league:
        if member.target not in SCRIPTED and member.target not in models:
            models[member.target] = load(member.target, device)
    env = Env(a.pool, a.envs, a.seed, a.open_sheets, not a.no_vary, max_turns=a.max_turns, threads=a.threads)
    totals = generate(env, model, league, models, a.games, config, out, device, seed=a.seed, shard_size=a.shard_size)
    manifest = dict(
        **totals,
        search=asdict(config),
        league=[dict(asdict(o), name=o.name) for o in league],
        checkpoint=a.checkpoint,
        checkpoint_sha256=file_hash(a.checkpoint),
        opponent_sha256={o.target: file_hash(o.target) for o in league if o.target not in SCRIPTED + ("self",)},
        pool=a.pool,
        pool_sha256=file_hash(a.pool),
        seed=a.seed,
        open_sheets=a.open_sheets,
        vary=not a.no_vary,
        fields=list(Shards.FIELDS),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("games", "rows", "searched_rows", "seconds")}))


if __name__ == "__main__":
    main()
