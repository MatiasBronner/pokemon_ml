"""Split tournament pools by species composition, keeping spread variants together."""

import argparse
import hashlib
import json
import random
from pathlib import Path


def key(team):
    return tuple(
        sorted(
            "".join(c for c in p["species"].lower() if c.isalnum())
            for p in team["team"]
        )
    )


def assert_disjoint(train, evaluation):
    overlap = {key(t) for t in train["teams"]} & {key(t) for t in evaluation["teams"]}
    if overlap:
        raise ValueError(
            f"training/evaluation pools overlap in {len(overlap)} species compositions"
        )


def split(pool, fraction=0.2, seed=1):
    if not 0 < fraction < 1:
        raise ValueError("evaluation fraction must lie in (0,1)")
    groups = sorted({key(t) for t in pool["teams"]})
    if len(groups) < 2:
        raise ValueError("at least two species compositions are needed")
    random.Random(seed).shuffle(groups)
    test = set(groups[: min(len(groups) - 1, max(1, round(len(groups) * fraction)))])
    train, evaluation = dict(pool), dict(pool)
    train["teams"] = [t for t in pool["teams"] if key(t) not in test]
    evaluation["teams"] = [t for t in pool["teams"] if key(t) in test]
    assert_disjoint(train, evaluation)
    return train, evaluation


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)
    source = Path(a.pool)
    pool = json.loads(source.read_text())
    train, evaluation = split(pool, a.fraction, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train), ("eval", evaluation)]:
        (out / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n")
    (out / "split.json").write_text(
        json.dumps(
            dict(
                seed=a.seed,
                fraction=a.fraction,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                grouping="identical species multiset; spread, move and item variants stay together",
                train_teams=len(train["teams"]),
                eval_teams=len(evaluation["teams"]),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
