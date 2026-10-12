"""Fixed-seed, history-aware checkpoint evaluation and calibrated reports."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .env import Env, PREVIEW, part, NAMES
from .features import History
from . import ppo


def interval(wins, losses, ties):
    """Wilson interval for fractional score, draws=half; approximate with draws."""
    n = wins + losses + ties
    if not n:
        return [0.0, 1.0]
    p, z = (wins + 0.5 * ties) / n, 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0, center - radius), min(1, center + radius)]


def outcome_summary(outcomes):
    wins, losses, ties = (sum(x == v for x in outcomes) for v in (1, -1, 0))
    n = len(outcomes)
    return dict(
        games=n,
        wins=wins,
        losses=losses,
        ties=ties,
        score=(wins + 0.5 * ties) / max(n, 1),
        score_interval_95=interval(wins, losses, ties),
    )


def composition(ids):
    return "-".join(map(str, sorted(int(x) for x in ids)))


def calibration(predictions, outcomes, bins=10):
    p = np.clip((np.asarray(predictions) + 1) / 2, 0, 1)
    y = (np.asarray(outcomes) + 1) / 2
    if not len(p):
        return {"samples": 0, "brier": None, "ece": None, "bins": []}
    rows, ece = [], 0.0
    for b in range(bins):
        take = (p >= b / bins) & (p < (b + 1) / bins if b < bins - 1 else p <= 1)
        if take.any():
            confidence, score = float(p[take].mean()), float(y[take].mean())
            count = int(take.sum())
            ece += count / len(p) * abs(confidence - score)
            rows.append(dict(bin=b, count=count, predicted=confidence, actual=score))
    return dict(
        samples=len(p),
        brier=float(np.mean((p - y) ** 2)),
        ece=ece,
        bins=rows,
        interpretation="expected outcome score; draws count half; repeated decisions within a game are correlated",
    )


@torch.no_grad()
def play(
    env, first, second, games, device, greedy=False, trace_games=0, archetypes=None
):
    if games < 1:
        raise ValueError("games must be positive")
    first.eval()
    if not isinstance(second, str):
        second.eval()
    side = np.arange(env.envs) % 2
    mine_np, theirs_np = (
        2 * np.arange(env.envs) + side,
        2 * np.arange(env.envs) + 1 - side,
    )
    mine = torch.as_tensor(mine_np, device=device)
    theirs = torch.as_tensor(theirs_np, device=device)
    histories = [History(2 * env.envs, first.config.history, device)]
    if not isinstance(second, str):
        histories.append(History(2 * env.envs, second.config.history, device))
    pending = [[] for _ in range(env.envs)]
    traces = [[] for _ in range(env.envs)]
    teams, opponents = [""] * env.envs, [""] * env.envs
    results, predictions, labels, records = [], [], [], []
    groups, matchups, styles = defaultdict(list), defaultdict(list), defaultdict(list)
    initial_seed = env.seed
    batch_number = 0
    active = np.arange(env.envs) < min(env.envs, games)
    completed_turns, timeouts = 0, 0
    while len(results) < games:
        f, i, mask = ppo.to_device(env, device)
        info = part(i, "i", "info")
        roster = part(i, "i", "mon_ids")
        for g in range(env.envs):
            if active[g] and int(info[mine[g], 0]) == PREVIEW:
                teams[g] = composition(roster[mine[g], :6, 0].cpu())
                opponents[g] = composition(roster[mine[g], 6:, 0].cpu())
        h = histories[0].tensors(mine) if first.config.history else None
        chosen, _, value, _, _ = first.act(
            f[mine], i[mine], mask[mine], greedy=greedy, history=h
        )
        if isinstance(second, str):
            actions = torch.as_tensor(
                env.baseline(second), device=device, dtype=torch.long
            )
        else:
            actions = torch.zeros(2 * env.envs, 2, dtype=torch.long, device=device)
            h = histories[1].tensors(theirs) if second.config.history else None
            actions[theirs] = second.act(f[theirs], i[theirs], mask[theirs], history=h)[
                0
            ]
        actions[mine] = chosen
        for g in range(env.envs):
            if active[g] and int(info[mine[g], 3]) == 1:
                pending[g].append(float(value[g]))
                if g < trace_games:
                    traces[g].append(
                        {
                            "turn": int(info[mine[g], 2]),
                            "phase": int(info[mine[g], 0]),
                            "actions": chosen[g].cpu().tolist(),
                            "value": float(value[g]),
                            "f": f[mine[g]].cpu().tolist(),
                            "i": i[mine[g]].cpu().tolist(),
                        }
                    )
        for hist in histories:
            hist.append(
                f,
                i,
                (info[:, 3] == 1)
                & torch.as_tensor(active, device=device).repeat_interleave(2),
            )
        env.step(actions.cpu().numpy())
        over = torch.as_tensor(env.done, device=device).bool().repeat_interleave(2)
        for hist in histories:
            hist.reset(over)
        episode_info = env.episode_info()
        for g in np.flatnonzero(env.done.astype(bool) & active):
            active[g] = False
            if len(results) >= games:
                continue
            completed_turns += int(episode_info[g, 0])
            timeouts += int(episode_info[g, 1])
            result = int(env.reward[g, side[g]])
            results.append(result)
            predictions.extend(pending[g])
            labels.extend([result] * len(pending[g]))
            pending[g] = []
            groups[teams[g]].append(result)
            matchups[teams[g] + " vs " + opponents[g]].append(result)
            style = (
                archetypes if archetypes is not None else getattr(env, "archetypes", {})
            ).get(teams[g], "unlabelled")
            styles[style].append(result)
            if g < trace_games and len(records) < trace_games:
                records.append(
                    dict(
                        team=teams[g],
                        opponent=opponents[g],
                        outcome=result,
                        decisions=traces[g],
                    )
                )
            traces[g] = []
        # Count each initial episode exactly once, then reseed the next batch.
        # Auto-reset episodes in already-finished slots cannot bias the sample.
        if not active.any() and len(results) < games:
            batch_number += 1
            env = env.restart(initial_seed + batch_number)
            active[:] = np.arange(env.envs) < min(env.envs, games - len(results))
            for hist in histories:
                hist.reset(torch.ones(2 * env.envs, dtype=torch.bool, device=device))
    return {
        **outcome_summary(results),
        "seed": initial_seed,
        "schedule": "first episode per slot; batch seed increments by one",
        "open_sheets": env.open_sheets,
        "max_turns": env.max_turns,
        "per_team": {k: outcome_summary(v) for k, v in groups.items()},
        "matchups": {k: outcome_summary(v) for k, v in matchups.items()},
        "archetypes": {k: outcome_summary(v) for k, v in styles.items()},
        "calibration": calibration(predictions, labels),
        "traces": records,
        "timeouts": timeouts,
        "mean_turns": completed_turns / len(results),
        "team_names": {
            k: [NAMES["species"][int(x)] for x in k.split("-")] for k in groups
        },
        "privileged_opponent": second == "lookahead"
        if isinstance(second, str)
        else False,
    }


def load(path, device):
    from .model import Model, Config
    from .checkpoint import verify_contract

    saved = torch.load(path, map_location="cpu", weights_only=False)
    verify_contract(saved)
    model = Model(Config(**saved["model_config"])).to(device).eval()
    model.load_state_dict(saved["model"])
    return model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--opponent",
        default="greedy",
        help="random, greedy, lookahead or checkpoint path",
    )
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed", type=int, default=9000001)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--open-sheets", type=float, default=0.5)
    p.add_argument("--trace-games", type=int, default=2)
    p.add_argument("--archetypes", help="JSON mapping composition keys to labels")
    a = p.parse_args(argv)
    from .train import pick_device
    from .checkpoint import file_hash

    device = pick_device(a.device)
    first = load(a.checkpoint, device)
    second = (
        a.opponent
        if a.opponent in ("random", "greedy", "lookahead")
        else load(a.opponent, device)
    )
    env = Env(a.pool, a.envs, a.seed, a.open_sheets, False, threads=a.threads)
    report = play(
        env,
        first,
        second,
        a.games,
        device,
        trace_games=a.trace_games,
        archetypes=json.loads(Path(a.archetypes).read_text()) if a.archetypes else None,
    )
    report.update(
        seed=a.seed,
        pool_sha256=file_hash(a.pool),
        checkpoint_sha256=file_hash(a.checkpoint),
        opponent=a.opponent,
        open_sheets=a.open_sheets,
        opponent_sha256=file_hash(a.opponent) if not isinstance(second, str) else None,
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ("games", "score", "score_interval_95", "timeouts")}
        )
    )


if __name__ == "__main__":
    main()
