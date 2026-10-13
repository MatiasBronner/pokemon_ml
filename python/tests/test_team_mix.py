import copy

import pytest

from pokemon_ml.team_mix import combine
from pokemon_ml.team_splits import assert_disjoint, split


def pool(event, species):
    return dict(
        source=event,
        event=event,
        format="target",
        teams=[dict(team=[dict(species=s)]) for s in species],
    )


def test_group_event_and_composition_weights():
    mixed = combine(
        [pool("a", ["A"] * 9 + ["B"]), pool("b", ["C"])], [pool("worlds", ["D"] * 100)]
    )

    def mass(event):
        return sum(t["sampling_weight"] for t in mixed["teams"] if t["event"] == event)

    assert mass("a") == pytest.approx(0.45)
    assert mass("b") == pytest.approx(0.45)
    assert mass("worlds") == pytest.approx(0.1)
    a = [t for t in mixed["teams"] if t["event"] == "a"]
    assert a[-1]["sampling_weight"] == pytest.approx(0.45 * (0.05 + 0.25))


def test_split_reweights_and_keeps_compositions_disjoint():
    mixed = combine([pool("a", list("ABCDEFGH"))], [pool("w", list("IJKLMNOP"))])
    train, evaluation = split(copy.deepcopy(mixed), seed=23)
    assert_disjoint(train, evaluation)
    for part in (train, evaluation):
        assert sum(t["sampling_weight"] for t in part["teams"]) == pytest.approx(1)
        if {t["sampling_group"] for t in part["teams"]} == {"current", "worlds"}:
            assert sum(
                t["sampling_weight"]
                for t in part["teams"]
                if t["sampling_group"] == "worlds"
            ) == pytest.approx(0.1)


def test_reject_duplicate_events_and_bad_configuration():
    a = pool("a", ["A"])
    with pytest.raises(ValueError):
        combine([a, a], [])
    with pytest.raises(ValueError):
        combine([a], [], worlds_fraction=2)
    with pytest.raises(ValueError):
        combine([a], [dict(pool("w", ["B"]), format="other")])
    mixed = combine([a], [])
    assert mixed["teams"][0]["sampling_weight"] == 1


def test_legacy_split_does_not_enable_weighting():
    train, evaluation = split(pool("a", list("ABCD")))
    assert all("sampling_weight" not in t for t in train["teams"] + evaluation["teams"])


def test_split_does_not_mutate_source_weights():
    mixed = combine([pool("a", list("ABCDEFGH"))], [pool("w", list("IJKLMNOP"))])
    before = copy.deepcopy(mixed)
    split(mixed)
    assert mixed == before


def test_standalone_commands_write_separate_evaluation_pools(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    a, w = tmp_path / "current.json", tmp_path / "worlds.json"
    a.write_text(json.dumps(pool("a", list("ABCDEFGH"))))
    w.write_text(json.dumps(pool("w", list("IJKLMNOP"))))
    merged = tmp_path / "mix.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "python/pokemon_ml/team_mix.py"),
            "--current",
            str(a),
            "--worlds",
            str(w),
            "--out",
            str(merged),
        ],
        check=True,
    )
    out = tmp_path / "split"
    subprocess.run(
        [
            sys.executable,
            str(root / "python/pokemon_ml/team_splits.py"),
            "--pool",
            str(merged),
            "--out",
            str(out),
            "--seed",
            "23",
        ],
        check=True,
    )
    for group in ("current", "worlds"):
        data = json.loads((out / f"eval-{group}.json").read_text())
        assert {t["sampling_group"] for t in data["teams"]} == {group}
        assert sum(t["sampling_weight"] for t in data["teams"]) == pytest.approx(1)
