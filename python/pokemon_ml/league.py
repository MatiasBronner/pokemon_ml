"""Measuring one version of the network against others.

Win rates against fixed players stop telling much once they are beaten. What
keeps telling is whether a new version beats the old ones. During training a
copy of the network is put aside every so often (a snapshot), each new one
plays some of those before it and the three scripted players, and all the
results are fitted with Elo ratings on one scale: a difference of 100 points
is a 64% chance of winning, 200 is 76%, 400 is 91%. The untrained network the
run started from is the zero.

    python -m pokemon_ml.league --run runs/first --pool teams/2027-frankfurt.json

plays more games between a run's snapshots than training had time for, fits
the ratings again and prints the table.
"""

import argparse
import json
import math
import os

import numpy as np
import torch

from . import model as model_lib
from .env import Env

SCRIPTED = ("random", "greedy", "lookahead")
START = "000000"


@torch.no_grad()
def play_models(env, first, second, games, device):
    from .evaluation import play

    report = play(env, first, second, games, device)
    return report["score"], report["games"]


def play_scripted(env, first, second, games):
    """The same between two scripted players."""
    side = np.arange(env.envs) % 2
    mine = 2 * np.arange(env.envs) + side
    points = played = 0.0
    while played < games:
        actions = env.baseline(second)
        actions[mine] = env.baseline(first)[mine]
        env.step(actions)
        over = env.done.astype(bool)
        result = env.reward[over, side[over]]
        points += float((result > 0).sum()) + 0.5 * float((result == 0).sum())
        played += float(over.sum())
    return points / played, int(played)


def scripted_results(make_env, games, seed=0, kinds=SCRIPTED):
    """The scripted players against each other, so that their ratings do not rest on how
    badly early networks lose to each of them."""
    pairs = [(a, b) for k, a in enumerate(kinds) for b in kinds[:k]]
    return [
        (a, b, *play_scripted(make_env(seed + k), a, b, games))
        for k, (a, b) in enumerate(pairs)
    ]


def expected(rating, other):
    """The chance a player rated `rating` beats one rated `other`."""
    return 1.0 / (1.0 + 10.0 ** ((other - rating) / 400.0))


def fit(results, anchor=START, rounds=400):
    """Elo ratings that best explain `results`, a list of (player, opponent, share of the
    games the player won, games). `anchor` is held at zero. Each pair that met is given one
    extra game, drawn: it keeps a player that won or lost everything from running off to
    infinity, and counts for little beside real games."""
    players = sorted({name for a, b, _, _ in results for name in (a, b)})
    rating = {name: 0.0 for name in players}
    met = {}
    for a, b, share, games in results:
        key, won = ((a, b), share * games) if a < b else ((b, a), (1 - share) * games)
        wins, n = met.get(key, (0.5, 1.0))
        met[key] = (wins + won, n + games)
    for _ in range(rounds):
        # One step of Newton's method on each player in turn.
        for name in players:
            if name == anchor:
                continue
            slope = curve = 0.0
            for (a, b), (wins, n) in met.items():
                if name not in (a, b):
                    continue
                other = b if name == a else a
                won = wins if name == a else n - wins
                p = expected(rating[name], rating[other])
                slope += won - n * p
                curve += n * p * (1 - p)
            if curve > 0:
                rating[name] += max(
                    -200.0, min(200.0, slope / curve * 400.0 / math.log(10))
                )
    shift = rating.get(anchor, 0.0)
    return {name: r - shift for name, r in rating.items()}


def snapshot_path(run, name):
    return os.path.join(run, "snapshots", name + ".pt")


def snapshots(run):
    """The names of a run's snapshots, oldest first."""
    folder = os.path.join(run, "snapshots")
    return (
        sorted((f[:-3] for f in os.listdir(folder) if f.endswith(".pt")), key=int)
        if os.path.isdir(folder)
        else []
    )


def save_snapshot(run, update, model):
    name = f"{update:06d}"
    os.makedirs(os.path.join(run, "snapshots"), exist_ok=True)
    path = snapshot_path(run, name)
    from .checkpoint import contract

    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.save_config(),
            "contract": contract(),
        },
        path + ".tmp",
    )
    os.replace(path + ".tmp", path)
    return name


def load_snapshot(run, name, device):
    saved = torch.load(snapshot_path(run, name), map_location="cpu", weights_only=False)
    from .checkpoint import verify_contract

    verify_contract(saved)
    net = model_lib.Model(model_lib.Config(**saved["model_config"])).to(device)
    net.load_state_dict(saved["model"])
    return net.eval()


def earlier(names, count):
    """Which of the snapshots before the newest to play it against: the one just before, then
    twice as far back each time, so that there are close rivals and old ones."""
    picked, back = [], 1
    while len(picked) < count and back <= len(names):
        picked.append(names[-back])
        back *= 2
    if names and names[0] not in picked:
        picked.append(names[0])
    return picked


class Ratings:
    """A run's results between versions, kept in `<run>/ratings.json`, and the ratings fitted to them."""

    def __init__(self, run):
        self.path = os.path.join(run, "ratings.json")
        self.results = []
        if os.path.exists(self.path):
            with open(self.path) as file:
                self.results = [tuple(r) for r in json.load(file)["results"]]
        self.ratings = fit(self.results) if self.results else {}

    def add(self, results):
        self.results += [tuple(r) for r in results]
        self.ratings = fit(self.results)
        matrix = {}
        for a, b, share, games in self.results:
            cell = matrix.setdefault(a, {}).setdefault(b, {"points": 0, "games": 0})
            cell["points"] += share * games
            cell["games"] += games
        for row in matrix.values():
            for cell in row.values():
                cell["score"] = cell["points"] / cell["games"]
        with open(self.path + ".tmp", "w") as file:
            json.dump(
                {
                    "results": self.results,
                    "ratings": self.ratings,
                    "matchup_matrix": matrix,
                },
                file,
                indent=1,
            )
        os.replace(self.path + ".tmp", self.path)


def measure(
    run,
    name,
    model,
    make_env,
    device,
    games,
    rivals=4,
    seed=0,
    archetypes=None,
    include_lookahead=True,
):
    """Plays the snapshot `name` (the network `model`) against the three scripted players and
    some earlier snapshots. Returns the results, as `fit` takes them. With a run's first
    snapshot come the scripted players' games against each other."""
    kinds = SCRIPTED if include_lookahead else SCRIPTED[:2]
    results = (
        []
        if snapshots(run)[:1] != [name]
        else scripted_results(make_env, 2 * games, seed + 50, kinds)
    )
    for k, kind in enumerate(kinds):
        from .evaluation import play

        report = play(
            make_env(seed + k), model, kind, games, device, archetypes=archetypes
        )
        results.append((name, kind, report["score"], report["games"]))
        save_report(run, name, kind, report, seed + k)
    before = [s for s in snapshots(run) if int(s) < int(name)]
    for k, rival in enumerate(earlier(before, rivals)):
        report = play(
            make_env(seed + 10 + k),
            model,
            load_snapshot(run, rival, device),
            games,
            device,
            archetypes=archetypes,
        )
        results.append((name, rival, report["score"], report["games"]))
        save_report(run, name, rival, report, seed + 10 + k)
    return results


def save_report(run, name, rival, report, seed):
    folder = os.path.join(run, "evaluation")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{name}-vs-{rival}.json"), "w") as f:
        json.dump(dict(evaluation_offset=seed, **report), f, indent=2)


def table(ratings, results):
    """The ratings, best first, with each one's games."""
    games = {}
    for a, b, _, n in results:
        games[a], games[b] = games.get(a, 0) + n, games.get(b, 0) + n
    lines = [f"{'version':>12}  {'rating':>7}  {'games':>7}"]
    for name, rating in sorted(ratings.items(), key=lambda item: -item[1]):
        lines.append(f"{name:>12}  {rating:7.0f}  {games.get(name, 0):7.0f}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True, help="the run whose snapshots to rate")
    p.add_argument("--pool", required=True, help="team pool to play with")
    p.add_argument(
        "--games",
        type=int,
        default=400,
        help="games for each pair of versions that meet",
    )
    p.add_argument(
        "--rivals", type=int, default=6, help="earlier versions each one plays"
    )
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args(argv)
    from .train import pick_device

    device = pick_device(args.device)
    names = snapshots(args.run)
    if not names:
        raise SystemExit(
            f"{args.run} has no snapshots yet (training puts one aside every --eval-every updates)"
        )
    make_env = lambda seed: Env(
        args.pool, args.envs, 9_000_000 + seed, threads=args.threads
    )
    ratings = Ratings(args.run)
    for k, name in enumerate(names):
        net = load_snapshot(args.run, name, device)
        ratings.add(
            measure(
                args.run,
                name,
                net,
                make_env,
                device,
                args.games,
                args.rivals,
                seed=100 * k,
            )
        )
        print(f"{name}: {ratings.ratings[name]:.0f}", flush=True)
    print(table(ratings.ratings, ratings.results))


class TrainingLeague:
    """Episode-stable opponents. Frozen/scripted rows never enter PPO updates."""

    def __init__(
        self,
        run,
        envs,
        device,
        seed=1,
        weights=(0.5, 0.4, 0.1),
        cache_size=4,
        state=None,
        selection=None,
        pool_size=None,
    ):
        from collections import OrderedDict

        if any(w < 0 for w in weights) or abs(sum(weights) - 1) > 1e-6:
            raise ValueError("league weights must be nonnegative and sum to one")
        if cache_size < 1:
            raise ValueError("league cache size must be positive")
        self.selection = selection or (state or {}).get("selection", "legacy")
        self.pool_size = (
            pool_size if pool_size is not None else (state or {}).get("pool_size", 12)
        )
        if self.selection not in ("legacy", "broad") or self.pool_size < 4:
            raise ValueError(
                "league selection must be legacy/broad and pool size at least four"
            )
        self.results = {k: dict(v) for k, v in (state or {}).get("results", {}).items()}
        self.update_results = {}
        self.run, self.device, self.weights = run, device, tuple(weights)
        self.cache_size, self.cache = cache_size, OrderedDict()
        self.rng = np.random.default_rng(seed)
        if state:
            self.rng.bit_generator.state = state["rng"]
            if tuple(state["weights"]) != self.weights:
                raise ValueError("league weights changed on resume")
        self.assigned = ["current"] * envs
        self.side = np.arange(envs) % 2
        if state and not set(state["snapshots"]).issubset(set(snapshots(run))):
            raise ValueError(
                "a saved league snapshot is missing; restore the run snapshot directory"
            )
        self.allowed = set(state["snapshots"]) if state else None
        self.initialized = False
        self.counts = (
            state["counts"].copy()
            if state
            else {"current": 0, "frozen": 0, "scripted": 0}
        )

    def assign(self, done):
        names = [
            n for n in snapshots(self.run) if self.allowed is None or n in self.allowed
        ]
        # A bounded diverse candidate pool: recent plus geometrically older.
        candidates = list(
            dict.fromkeys(
                names[-2:] + ([names[len(names) // 2], names[0]] if names else [])
            )
        )
        probabilities = None
        if self.selection == "broad" and names:
            # Spread the historical budget across the full saved history; retain
            # the two newest opponents. No outcome-based bias in this first experiment.
            historical = names[:-2]
            indices = np.linspace(
                0, max(0, len(historical) - 1), self.pool_size - 2, dtype=int
            )
            candidates = list(
                dict.fromkeys(
                    names[-2:] + [historical[j] for j in indices]
                    if historical
                    else names
                )
            )
            recent = min(2, len(names))
            if len(candidates) > recent:
                probabilities = [0.5 / recent] * recent + [
                    0.5 / (len(candidates) - recent)
                ] * (len(candidates) - recent)
        for game in np.flatnonzero(done):
            kind = int(self.rng.choice(3, p=self.weights))
            opponent = (
                "current"
                if kind == 0 or (kind == 1 and not candidates)
                else "snapshot:" + str(self.rng.choice(candidates, p=probabilities))
                if kind == 1
                else str(self.rng.choice(("random", "greedy")))
            )
            self.assigned[game] = opponent
            self.side[game] = int(self.rng.integers(2))
            self.counts[
                "current"
                if opponent == "current"
                else "frozen"
                if opponent.startswith("snapshot:")
                else "scripted"
            ] += 1
        self.initialized = True

    def record_results(self, done, reward):
        """Record terminal outcomes before finished slots receive new opponents."""
        for game in np.flatnonzero(done):
            name = self.assigned[game]
            # Current self-play has two learner sides; use assigned side as the
            # reporting perspective, rather than count both sides as separate games.
            result = float(reward[game, self.side[game]])
            for records in (self.results, self.update_results):
                row = records.setdefault(name, dict(games=0, wins=0, losses=0, ties=0))
                row["games"] += 1
                row["wins" if result > 0 else "losses" if result < 0 else "ties"] += 1

    def drain_results(self):
        records, self.update_results = self.update_results, {}
        return records

    def learner_rows(self):
        learn = np.ones(2 * len(self.assigned), dtype=bool)
        for g, name in enumerate(self.assigned):
            if name != "current":
                learn[2 * g + 1 - self.side[g]] = False
        return torch.as_tensor(learn, device=self.device)

    @torch.no_grad()
    def replace_actions(self, env, f, i, mask, actions, history=None):
        import time

        self.scripted_seconds = 0.0
        if not self.initialized:
            self.assign(np.ones(env.envs, dtype=bool))
        for name in sorted(set(self.assigned) - {"current"}):
            games = np.array([g for g, n in enumerate(self.assigned) if n == name])
            rows = torch.as_tensor(2 * games + 1 - self.side[games], device=self.device)
            if name.startswith("snapshot:"):
                if name not in self.cache:
                    net = load_snapshot(self.run, name.split(":", 1)[1], self.device)
                    net.requires_grad_(False)
                    self.cache[name] = net
                    while len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
                self.cache.move_to_end(name)
                net = self.cache[name]
                hist = (
                    None
                    if history is None or not net.config.history
                    else tuple(h[rows] for h in history)
                )
                actions[rows] = net.act(f[rows], i[rows], mask[rows], history=hist)[0]
            else:
                started = time.perf_counter()
                baseline = torch.as_tensor(
                    env.baseline(name), device=self.device, dtype=torch.long
                )
                self.scripted_seconds += time.perf_counter() - started
                actions[rows] = baseline[rows]
        return self.learner_rows()

    def register_snapshot(self, name):
        if self.allowed is not None:
            self.allowed.add(name)

    def state_dict(self):
        # Live episodes intentionally restart; RNG restored before new assignments.
        return {
            "selection": self.selection,
            "pool_size": self.pool_size,
            "results": self.results,
            "rng": self.rng.bit_generator.state,
            "weights": self.weights,
            "snapshots": [
                n
                for n in snapshots(self.run)
                if self.allowed is None or n in self.allowed
            ],
            "counts": self.counts,
        }


if __name__ == "__main__":
    main()
