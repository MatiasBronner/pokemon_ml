"""Run controlled multi-seed ablations; report strength against allocated training time.

The runner prints commands unless --execute is explicitly supplied.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = {
    "baseline": [],
    "no-speed": ["--no-speed"],
    "semantic-actions": ["--semantic-actions"],
    "joint-preview": ["--joint-preview"],
    "opponent-prediction": ["--opponent-prediction"],
    "opponent-feedback": ["--opponent-prediction", "--prediction-feedback"],
    "history": ["--history", "16"],
    "hidden-prediction": ["--hidden-prediction"],
    "no-league": [
        "--league-current",
        "1",
        "--league-frozen",
        "0",
        "--league-scripted",
        "0",
    ],
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True)
    p.add_argument("--eval-pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument(
        "--experiments", nargs="+", choices=EXPERIMENTS, default=["baseline"]
    )
    p.add_argument(
        "--sizes", nargs="+", choices=["small", "medium", "large"], default=["small"]
    )
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--hours", type=float)
    p.add_argument("--execute", action="store_true")
    a, extra = p.parse_known_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for experiment in a.experiments:
        for size in a.sizes:
            for seed in a.seeds:
                run = out / f"{experiment}-{size}-{seed}"
                cmd = [
                    sys.executable,
                    "-m",
                    "pokemon_ml.train",
                    "--pool",
                    a.pool,
                    "--eval-pool",
                    a.eval_pool,
                    "--run",
                    str(run),
                    "--model-size",
                    size,
                    "--seed",
                    str(seed),
                    "--updates",
                    str(a.updates),
                ]
                if a.hours:
                    cmd += ["--hours", str(a.hours)]
                cmd += EXPERIMENTS[experiment] + extra
                manifest.append(
                    dict(experiment=experiment, size=size, seed=seed, command=cmd)
                )
    (out / "experiments.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for row in manifest:
        print(json.dumps(row), flush=True)
        if a.execute:
            subprocess.run(row["command"], check=True)


if __name__ == "__main__":
    main()
