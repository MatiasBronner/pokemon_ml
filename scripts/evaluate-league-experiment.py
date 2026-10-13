#!/usr/bin/env python3
"""Evaluate each continuation against the same immutable historical panel."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--arm", choices=["source", "legacy", "broad"], required=True)
    parser.add_argument("--games", type=int, default=1500)
    args = parser.parse_args()
    root = Path(args.experiment)
    manifest = json.loads((root / "experiment.json").read_text())
    config = manifest["source_args"]
    model_path = (
        root / "source-checkpoint.pt"
        if args.arm == "source"
        else root / args.arm / "checkpoint.pt"
    )
    out = root / "evaluation" / args.arm
    out.mkdir(parents=True, exist_ok=True)
    # Use copied immutable snapshots, including for source evaluations.
    for rival in manifest["panel"] + ["greedy"]:
        opponent = (
            rival
            if rival == "greedy"
            else str(root / "legacy" / "snapshots" / (rival + ".pt"))
        )
        print(f"Evaluating {args.arm} against {rival}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pokemon_ml.evaluation",
                "--checkpoint",
                str(model_path),
                "--opponent",
                opponent,
                "--pool",
                config.get("eval_pool") or config["pool"],
                "--games",
                str(args.games),
                "--envs",
                "128",
                "--threads",
                str(config["threads"]),
                "--open-sheets",
                str(config["open_sheets"]),
                "--max-turns",
                str(config["max_turns"]),
                "--seed",
                "9702347",
                "--device",
                "cuda",
                "--trace-games",
                "0",
                "--out",
                str(out / (rival + ".json")),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
