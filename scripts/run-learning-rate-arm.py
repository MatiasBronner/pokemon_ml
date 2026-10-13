#!/usr/bin/env python3
"""Resume a prepared learning-rate arm with identical non-LR training settings."""

import argparse
import json
from pathlib import Path
from pokemon_ml.train import main as train


def training_args(root, arm, device="cuda"):
    root = Path(root)
    manifest = json.loads((root / "experiment.json").read_text())
    if manifest.get("experiment") != "learning-rate" or arm not in manifest["arms"]:
        raise ValueError("not a prepared learning-rate arm")
    a = manifest["source_args"]
    args = [
        "--run",
        str(root / arm),
        "--pool",
        a["pool"],
        "--updates",
        str(manifest["target_update"]),
        "--lr",
        str(manifest["learning_rates"][arm]),
        "--device",
        device,
        "--precision",
        "fp32",
        "--attention",
        "sdpa",
        "--deferred-metrics",
        "--no-fused-adam",
        "--league-selection",
        "legacy",
        "--league-cache-size",
        "32",
    ]
    if device == "cuda":
        args.append("--compile-encoder")
    else:
        args.append("--no-compile-encoder")
    for key in (
        "eval_pool",
        "envs",
        "threads",
        "seed",
        "eval_seed",
        "open_sheets",
        "max_turns",
        "eval_every",
        "eval_games",
        "save_every",
        "snapshot_every",
        "league_current",
        "league_frozen",
        "league_scripted",
        "archetypes",
    ):
        if a.get(key) is not None:
            args.extend(["--" + key.replace("_", "-"), str(a[key])])
    for key in ("no_vary", "no_lookahead_eval"):
        if a.get(key):
            args.append("--" + key.replace("_", "-"))
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--arm", choices=("control", "low-lr"), required=True)
    args = parser.parse_args()
    train(training_args(args.experiment, args.arm))
