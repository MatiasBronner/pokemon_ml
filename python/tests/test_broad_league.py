import numpy as np
import pytest

from pokemon_ml import league


def fake_history(monkeypatch, count=100):
    monkeypatch.setattr(
        league, "snapshots", lambda run: [f"{i:06d}" for i in range(count)]
    )


def test_broad_reaches_history_without_changing_episode_opponents(monkeypatch):
    fake_history(monkeypatch)
    broad = league.TrainingLeague(
        "unused", 2000, "cpu", weights=(0, 1, 0), selection="broad"
    )
    broad.assign(np.ones(2000, dtype=bool))
    names = set(broad.assigned)
    assert len(names) == 12
    assert {"snapshot:000000", "snapshot:000098", "snapshot:000099"} <= names
    assert "snapshot:000010" in names
    recent = sum(n in ("snapshot:000098", "snapshot:000099") for n in broad.assigned)
    assert 850 < recent < 1150
    assigned = list(broad.assigned)
    broad.assign(np.zeros(2000, dtype=bool))
    assert broad.assigned == assigned
    legacy = league.TrainingLeague("unused", 2000, "cpu", weights=(0, 1, 0))
    legacy.assign(np.ones(2000, dtype=bool))
    assert set(legacy.assigned) == {
        "snapshot:000000",
        "snapshot:000050",
        "snapshot:000098",
        "snapshot:000099",
    }


def test_terminal_results_use_learner_side_and_resume(monkeypatch):
    fake_history(monkeypatch, 3)
    players = league.TrainingLeague("unused", 3, "cpu")
    players.assigned = ["snapshot:000001", "greedy", "current"]
    players.side = np.array([1, 0, 1])
    players.record_results(
        np.array([True, True, False]), np.array([[-1, 1], [-1, 1], [0, 0]])
    )
    records = players.drain_results()
    assert records["snapshot:000001"]["wins"] == 1
    assert records["greedy"]["losses"] == 1
    assert players.drain_results() == {}
    players.record_results(np.array([False, False, True]), np.zeros((3, 2)))
    assert players.results["current"]["ties"] == 1
    state = players.state_dict()
    resumed = league.TrainingLeague("unused", 3, "cpu", state=state)
    assert resumed.results == players.results
    assert resumed.selection == "legacy"
    broad = league.TrainingLeague("unused", 3, "cpu", state=state, selection="broad")
    assert broad.selection == "broad"
    assert broad.results == players.results
    with pytest.raises(ValueError):
        league.TrainingLeague("unused", 1, "cpu", pool_size=2)


def test_prepared_arms_resume_from_identical_checkpoint(pool, tmp_path):
    import runpy
    from pathlib import Path
    import torch
    from pokemon_ml.train import main

    torch.set_num_threads(1)
    source = tmp_path / "source"
    shared = [
        "--pool",
        pool,
        "--device",
        "cpu",
        "--envs",
        "2",
        "--threads",
        "1",
        "--steps",
        "8",
        "--epochs",
        "1",
        "--minibatch",
        "4",
        "--eval-games",
        "2",
        "--max-turns",
        "3",
        "--no-lookahead-eval",
    ]
    main(
        shared
        + [
            "--run",
            str(source),
            "--updates",
            "1",
            "--width",
            "32",
            "--layers",
            "1",
            "--heads",
            "2",
            "--ff",
            "64",
        ]
    )
    before = (source / "checkpoint.pt").read_bytes()
    root = Path(__file__).resolve().parents[2]
    prepare = runpy.run_path(str(root / "scripts/prepare-league-experiment.py"))[
        "prepare"
    ]
    experiment = tmp_path / "experiment"
    prepare(source, experiment, updates=1)
    for arm in ("legacy", "broad"):
        assert (experiment / arm / "checkpoint.pt").read_bytes() == before
        main(
            shared
            + [
                "--run",
                str(experiment / arm),
                "--updates",
                "2",
                "--league-selection",
                arm,
                "--league-cache-size",
                "32",
            ]
        )
        saved = torch.load(experiment / arm / "checkpoint.pt", weights_only=False)
        assert saved["update"] == 2
        assert saved["league"]["selection"] == arm
        assert saved["league"]["results"]
    assert (source / "checkpoint.pt").read_bytes() == before
    with pytest.raises(FileExistsError):
        prepare(source, experiment)
