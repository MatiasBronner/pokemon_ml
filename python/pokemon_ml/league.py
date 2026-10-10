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
from . import ppo
from .env import Env

SCRIPTED = ("random", "greedy", "lookahead")
START = "000000"


@torch.no_grad()
def play_models(env, first, second, games, device):
    """The share of at least `games` games that `first` wins against `second`, ties counting as
    half. `first` takes the first side of even-numbered games and the second of odd ones."""
    first.eval(), second.eval()
    side = torch.arange(env.envs) % 2
    mine = (2 * torch.arange(env.envs) + side).numpy()
    theirs = (2 * torch.arange(env.envs) + 1 - side).numpy()
    points = played = 0.0
    actions = torch.zeros(2 * env.envs, 2, dtype=torch.long)
    while played < games:
        f, i, mask = ppo.to_device(env, device)
        actions[mine] = first.act(f[mine], i[mine], mask[mine])[0].cpu()
        actions[theirs] = second.act(f[theirs], i[theirs], mask[theirs])[0].cpu()
        env.step(actions.numpy())
        over = env.done.astype(bool)
        result = env.reward[over, side.numpy()[over]]
        points += float((result > 0).sum()) + 0.5 * float((result == 0).sum())
        played += float(over.sum())
    return points / played, int(played)


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


def scripted_results(make_env, games, seed=0):
    """The scripted players against each other, so that their ratings do not rest on how
    badly early networks lose to each of them."""
    pairs = [(a, b) for k, a in enumerate(SCRIPTED) for b in SCRIPTED[:k]]
    return [(a, b, *play_scripted(make_env(seed + k), a, b, games)) for k, (a, b) in enumerate(pairs)]


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
                rating[name] += max(-200.0, min(200.0, slope / curve * 400.0 / math.log(10)))
    shift = rating.get(anchor, 0.0)
    return {name: r - shift for name, r in rating.items()}


def snapshot_path(run, name):
    return os.path.join(run, "snapshots", name + ".pt")


def snapshots(run):
    """The names of a run's snapshots, oldest first."""
    folder = os.path.join(run, "snapshots")
    return sorted(f[:-3] for f in os.listdir(folder) if f.endswith(".pt")) if os.path.isdir(folder) else []


def save_snapshot(run, update, model):
    name = f"{update:06d}"
    os.makedirs(os.path.join(run, "snapshots"), exist_ok=True)
    path = snapshot_path(run, name)
    torch.save({"model": model.state_dict(), "model_config": model.save_config()}, path + ".tmp")
    os.replace(path + ".tmp", path)
    return name


def load_snapshot(run, name, device):
    saved = torch.load(snapshot_path(run, name), map_location="cpu", weights_only=False)
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
        with open(self.path + ".tmp", "w") as file:
            json.dump({"results": self.results, "ratings": self.ratings}, file, indent=1)
        os.replace(self.path + ".tmp", self.path)


def measure(run, name, model, make_env, device, games, rivals=4, seed=0):
    """Plays the snapshot `name` (the network `model`) against the three scripted players and
    some earlier snapshots. Returns the results, as `fit` takes them. With a run's first
    snapshot come the scripted players' games against each other."""
    results = [] if snapshots(run)[:1] != [name] else scripted_results(make_env, 2 * games, seed + 50)
    for k, kind in enumerate(SCRIPTED):
        share = ppo.play_baseline(make_env(seed + k), model, kind, games, device)
        results.append((name, kind, share, games))
    before = [s for s in snapshots(run) if s < name]
    for k, rival in enumerate(earlier(before, rivals)):
        share, played = play_models(make_env(seed + 10 + k), model, load_snapshot(run, rival, device), games, device)
        results.append((name, rival, share, played))
    return results


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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="the run whose snapshots to rate")
    p.add_argument("--pool", required=True, help="team pool to play with")
    p.add_argument("--games", type=int, default=400, help="games for each pair of versions that meet")
    p.add_argument("--rivals", type=int, default=6, help="earlier versions each one plays")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args(argv)
    from .train import pick_device
    device = pick_device(args.device)
    names = snapshots(args.run)
    if not names:
        raise SystemExit(f"{args.run} has no snapshots yet (training puts one aside every --eval-every updates)")
    make_env = lambda seed: Env(args.pool, args.envs, 9_000_000 + seed, threads=args.threads)
    ratings = Ratings(args.run)
    for k, name in enumerate(names):
        net = load_snapshot(args.run, name, device)
        ratings.add(measure(args.run, name, net, make_env, device, args.games, args.rivals, seed=100 * k))
        print(f"{name}: {ratings.ratings[name]:.0f}", flush=True)
    print(table(ratings.ratings, ratings.results))


if __name__ == "__main__":
    main()
