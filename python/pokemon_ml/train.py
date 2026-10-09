"""Trains a network by self-play.

    python -m pokemon_ml.train --pool teams/2027-frankfurt.json --run runs/first

Plays `--envs` games at once with the network on both sides, updates it with
PPO every `--steps` decisions, and every `--eval-every` updates measures it
against the random and the greedy player. Everything needed to carry on is
saved to `<run>/checkpoint.pt` every `--save-every` updates, when `--hours`
of wall-clock time are up, and on Ctrl-C; the same command picks up from
there. The log is printed and appended to `<run>/log.jsonl`.
"""
import argparse
import json
import os
import signal
import time
from dataclasses import asdict

import torch

from . import model as model_lib
from . import ppo
from .env import Env


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", required=True, help="team pool to draw teams from (teams/<name>.json)")
    p.add_argument("--run", default="runs/first", help="directory for the checkpoint and the log")
    p.add_argument("--envs", type=int, default=512, help="games played at once")
    p.add_argument("--updates", type=int, default=1000, help="stop after this many updates in all")
    p.add_argument("--hours", type=float, default=None, help="stop (and save) after this much wall-clock time")
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--threads", type=int, default=0, help="threads stepping the games; 0 for one per core")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--open-sheets", type=float, default=0.5, help="share of games played with open team sheets")
    p.add_argument("--no-vary", action="store_true", help="play the pool's teams exactly as they are")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-games", type=int, default=500)
    p.add_argument("--save-every", type=int, default=25)
    for name, value in {**asdict(model_lib.Config()), **asdict(ppo.Config())}.items():
        p.add_argument("--" + name.replace("_", "-"), type=type(value), default=None)
    args = p.parse_args(argv)

    device = pick_device(args.device)
    os.makedirs(args.run, exist_ok=True)
    path = os.path.join(args.run, "checkpoint.pt")
    saved = torch.load(path, map_location="cpu", weights_only=False) if os.path.exists(path) else None

    # Sizes come from the checkpoint when there is one, so that a run carries on as it began.
    given = lambda config: {k: getattr(args, k) for k in asdict(config()) if getattr(args, k) is not None}
    net_config = model_lib.Config(**{**(saved["model_config"] if saved else {}), **given(model_lib.Config)})
    ppo_config = ppo.Config(**{**(saved["ppo_config"] if saved else {}), **given(ppo.Config)})
    torch.manual_seed(args.seed + (saved["update"] if saved else 0))
    model = model_lib.Model(net_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.lr, eps=1e-5)
    totals = {"update": 0, "decisions": 0, "games": 0, "seconds": 0.0}
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        totals = {k: saved[k] for k in totals}
        for group in optimizer.param_groups:
            group["lr"] = ppo_config.lr

    # A different seed each time a run is picked up, so that it does not replay the same games.
    env_seed = args.seed + 1000 * totals["update"]
    make = lambda envs, seed: Env(args.pool, envs, seed, args.open_sheets, not args.no_vary, threads=args.threads)
    env = make(args.envs, env_seed)
    roll = ppo.Rollout(ppo_config.steps, 2 * args.envs, device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{parameters:,} parameters on {device}; {args.envs} games at once on {env.raw.threads} threads; "
          f"{'resuming at' if saved else 'starting at'} update {totals['update']}", flush=True)

    def save():
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "model_config": asdict(net_config), "ppo_config": asdict(ppo_config), **totals}
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)  # never a half-written checkpoint

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    began = time.time()
    while totals["update"] < args.updates and not stop["now"]:
        t0 = time.time()
        games_before = env.stats()["games"]
        ppo.collect(env, model, roll)
        t1 = time.time()
        log = ppo.update(model, optimizer, roll, ppo_config)
        t2 = time.time()
        stats = env.stats()
        totals["update"] += 1
        totals["decisions"] += log["decisions"]
        totals["games"] += stats["games"] - games_before
        totals["seconds"] += t2 - t0
        log.update(update=totals["update"], games=totals["games"], hours=totals["seconds"] / 3600,
                   per_second=ppo_config.steps * args.envs / (t2 - t0), playing=(t1 - t0) / (t2 - t0),
                   turns=stats["turns"] / max(stats["games"], 1), ties=stats["ties"] / max(stats["games"], 1))
        if totals["update"] % args.eval_every == 0 or totals["update"] == args.updates:
            for kind in ("random", "greedy"):
                rival = make(min(args.envs, 256), 7_000_000 + totals["update"])
                log["vs_" + kind] = ppo.play_baseline(rival, model, kind, args.eval_games, device)
        line = (f"update {log['update']:5d}  games {log['games']:9,d}  {log['per_second']:7.0f} game-steps/s "
                f"({100 * log['playing']:.0f}% playing)  value explains {100 * log['explained']:5.1f}%  "
                f"entropy {log['entropy']:.2f}  kl {log['kl']:.4f}  {log['turns']:.1f} turns")
        if "vs_random" in log:
            line += f"  | wins {100 * log['vs_random']:.1f}% vs random, {100 * log['vs_greedy']:.1f}% vs greedy"
        print(line, flush=True)
        with open(os.path.join(args.run, "log.jsonl"), "a") as out:
            out.write(json.dumps(log) + "\n")
        out_of_time = args.hours is not None and time.time() - began > args.hours * 3600
        if totals["update"] % args.save_every == 0 or out_of_time:
            save()
        if out_of_time:
            print(f"{args.hours} hours are up", flush=True)
            break
    save()
    print(f"saved {path} at update {totals['update']}", flush=True)


if __name__ == "__main__":
    main()
