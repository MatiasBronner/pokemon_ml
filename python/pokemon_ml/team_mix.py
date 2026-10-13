"""Combine validated tournament pools with explicit source sampling weights.

Can also run directly: python3 python/pokemon_ml/team_mix.py --help
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def composition(team):
    return tuple(
        sorted(
            "".join(c for c in p["species"].lower() if c.isalnum())
            for p in team["team"]
        )
    )


def reweight(pool):
    """Recompute group/tournament/composition masses after filtering or splitting."""
    settings = pool.get("sampling")
    if not settings:
        return pool
    worlds = settings["worlds_fraction"]
    blend = settings["composition_balance"]
    if not 0 <= worlds <= 1 or not 0 <= blend <= 1:
        raise ValueError("sampling fractions must be between zero and one")
    events = defaultdict(lambda: defaultdict(list))
    for team in pool["teams"]:
        group = team["sampling_group"]
        if group not in ("current", "worlds"):
            raise ValueError(f"unknown sampling group: {group}")
        events[group][(team["source"], team["event"])].append(team)
    masses = {"current": 1 - worlds, "worlds": worlds}
    total = sum(masses[g] for g in events)
    if total <= 0:
        raise ValueError("available groups have zero sampling mass")
    for group, tournaments in events.items():
        event_mass = masses[group] / total / len(tournaments)
        for teams in tournaments.values():
            counts = Counter(composition(t) for t in teams)
            for team in teams:
                team["sampling_weight"] = event_mass * (
                    (1 - blend) / len(teams)
                    + blend / len(counts) / counts[composition(team)]
                )
    return pool


def combine(current, worlds, worlds_fraction=0.1, composition_balance=0.5):
    if not current:
        raise ValueError("at least one current-format pool is required")
    formats = {p["format"] for p in current + worlds}
    if len(formats) != 1:
        raise ValueError("all pools must be validated against the same target format")
    teams = []
    seen_events = set()
    for group, pools in [("current", current), ("worlds", worlds)]:
        for pool in pools:
            identity = (pool["source"], pool["event"])
            if identity in seen_events:
                raise ValueError(f"tournament supplied more than once: {identity}")
            seen_events.add(identity)
            for original in pool["teams"]:
                team = dict(original)
                team.update(
                    source=pool["source"], event=pool["event"], sampling_group=group
                )
                teams.append(team)
    result = dict(
        source="Multiple tournaments",
        event="Weighted tournament mixture",
        format=formats.pop(),
        teams=teams,
        sampling=dict(
            worlds_fraction=worlds_fraction, composition_balance=composition_balance
        ),
    )
    return reweight(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", nargs="+", required=True)
    parser.add_argument("--worlds", nargs="*", default=[])
    parser.add_argument("--worlds-fraction", type=float, default=0.1)
    parser.add_argument("--composition-balance", type=float, default=0.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    def read(paths):
        return [json.loads(Path(p).read_text()) for p in paths]

    pool = combine(
        read(args.current),
        read(args.worlds),
        args.worlds_fraction,
        args.composition_balance,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pool, indent=2) + "\n")
    for group in ("current", "worlds"):
        teams = [t for t in pool["teams"] if t["sampling_group"] == group]
        print(
            f"{group}: {len(teams)} teams, sampling mass {sum(t['sampling_weight'] for t in teams):.1%}"
        )


if __name__ == "__main__":
    main()
