#!/usr/bin/env python3
"""Clone a finished training run into matched legacy/broad league continuations."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil

import torch


def prepare(
    source,
    out,
    updates=1000,
    experiment="league",
    expected_update=None,
    control_lr=None,
    low_lr=0.0001,
):
    source, out = Path(source).resolve(), Path(out).resolve()
    if updates < 1:
        raise ValueError("additional updates must be positive")
    config = json.loads((source / "config.json").read_text())
    saved = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=False)
    if expected_update is not None and saved["update"] != expected_update:
        raise ValueError(
            f"expected source update {expected_update}, found {saved['update']}"
        )
    if experiment not in ("league", "learning-rate"):
        raise ValueError("unknown experiment")
    arms = ("legacy", "broad") if experiment == "league" else ("control", "low-lr")
    rates = {}
    if experiment == "learning-rate":
        source_lr = saved["ppo_config"]["lr"]
        control_lr = source_lr if control_lr is None else control_lr
        if control_lr != source_lr:
            raise ValueError(f"control LR must match source LR {source_lr}")
        if (
            not all(math.isfinite(x) and x > 0 for x in (control_lr, low_lr))
            or low_lr >= control_lr
        ):
            raise ValueError(
                "learning rates must be finite, positive, and low LR below control LR"
            )
        rates = {"control": control_lr, "low-lr": low_lr}
    if not (source / "snapshots").is_dir():
        raise ValueError("source snapshot history is required")
    if not saved.get("league"):
        raise ValueError(
            "a resumable training checkpoint with league state is required"
        )
    names = saved["league"]["snapshots"]
    if not names:
        raise ValueError("the source checkpoint must have historical opponents")
    out.mkdir(parents=True, exist_ok=False)
    for arm in arms:
        target = out / arm
        target.mkdir()
        # Mutable files are copied, never linked. Snapshots are immutable, so
        # linking them avoids duplicating gigabytes of historical opponents.
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, target / path.name)
        (target / "snapshots").mkdir()
        for path in (source / "snapshots").glob("*.pt"):
            try:
                os.link(path, target / "snapshots" / path.name)
            except OSError:
                shutil.copy2(path, target / "snapshots" / path.name)
    shutil.copy2(source / "checkpoint.pt", out / "source-checkpoint.pt")
    import hashlib

    # Prefer earlier opponents so a final-update snapshot does not duplicate
    # the explicit frozen-source comparison.
    names = [name for name in names if int(name) < saved["update"]] or names
    panel = list(
        dict.fromkeys(
            [names[0], names[len(names) // 2], names[3 * len(names) // 4], names[-1]]
        )
    )
    manifest = dict(
        source=str(source),
        experiment=experiment,
        arms=list(arms),
        learning_rates=rates,
        panel=panel,
        source_update=saved["update"],
        source_sha256=hashlib.sha256(
            (source / "checkpoint.pt").read_bytes()
        ).hexdigest(),
        target_update=saved["update"] + updates,
        source_args=config["args"],
    )
    (out / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Prepared {out}; both arms continue update {saved['update']} to {manifest['target_update']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument(
        "--experiment", choices=("league", "learning-rate"), default="league"
    )
    parser.add_argument("--expected-update", type=int)
    parser.add_argument(
        "--control-lr", type=float, help="must match source LR; defaults to source LR"
    )
    parser.add_argument("--low-lr", type=float, default=0.0001)
    args = parser.parse_args()
    prepare(
        args.source,
        args.out,
        args.updates,
        args.experiment,
        args.expected_update,
        args.control_lr,
        args.low_lr,
    )
