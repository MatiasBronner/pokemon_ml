"""Checkpoints from outside a run in its training league (`--league-fixed`)."""
import numpy as np
import pytest
import torch

from pokemon_ml import league
from pokemon_ml.expert_train import save
from pokemon_ml.model import Config, Model
from pokemon_ml.train import main


def test_a_fixed_opponent_is_drawn_with_its_weight_and_never_learned_from(monkeypatch):
    monkeypatch.setattr(league, "snapshots", lambda run: ["000000"])
    players = league.TrainingLeague("unused", 4000, "cpu", weights=(0.5, 0.2, 0.1), fixed=[("teacher.pt", 0.2)])
    players.assign(np.ones(4000, dtype=bool))
    share = np.mean([n == "fixed:teacher.pt" for n in players.assigned])
    assert 0.17 < share < 0.23 and players.counts["fixed"] == sum(n.startswith("fixed:") for n in players.assigned)
    learn = players.learner_rows().numpy()
    for g, name in enumerate(players.assigned):
        if name.startswith("fixed:"):
            assert learn[2 * g + players.side[g]] and not learn[2 * g + 1 - players.side[g]]
    # Saved and resumed with it; never quietly with another one.
    state = players.state_dict()
    assert state["fixed"] == [["teacher.pt", 0.2]]
    league.TrainingLeague("unused", 4, "cpu", weights=(0.5, 0.2, 0.1), fixed=[("teacher.pt", 0.2)], state=state)
    with pytest.raises(ValueError, match="fixed"):
        league.TrainingLeague("unused", 4, "cpu", weights=(0.5, 0.2, 0.1), fixed=[("other.pt", 0.2)], state=state)
    with pytest.raises(ValueError, match="sum to one"):
        league.TrainingLeague("unused", 4, "cpu", weights=(0.5, 0.4, 0.1), fixed=[("teacher.pt", 0.2)])


def test_without_fixed_opponents_the_draws_are_what_they_were(monkeypatch):
    monkeypatch.setattr(league, "snapshots", lambda run: ["000000", "000001"])
    players = league.TrainingLeague("unused", 500, "cpu", seed=3)
    players.assign(np.ones(500, dtype=bool))
    rng = np.random.default_rng(3)  # the draws as they were made before fixed opponents existed
    expected = []
    for _ in range(500):
        kind = int(rng.choice(3, p=(0.5, 0.4, 0.1)))
        expected.append("current" if kind == 0 else "snapshot:" + str(rng.choice(["000000", "000001"]))
                        if kind == 1 else str(rng.choice(("random", "greedy"))))
        rng.integers(2)
    assert players.assigned == expected


def test_training_plays_the_fixed_checkpoint(pool, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    teacher = Model(Config(width=32, layers=1, heads=2, ff=64))
    save(tmp_path / "teacher.pt", teacher, {})
    student = Model(Config(width=48, layers=1, heads=2, ff=64))
    save(tmp_path / "student.pt", student, {})  # as distillation writes it
    run = tmp_path / "run"
    args = ["--pool", pool, "--run", str(run), "--init-checkpoint", str(tmp_path / "student.pt"),
            "--league-current", "0.5", "--league-frozen", "0", "--league-scripted", "0",
            "--league-fixed", f"{tmp_path / 'teacher.pt'}:0.5", "--device", "cpu", "--envs", "8",
            "--threads", "1", "--updates", "1", "--steps", "8", "--epochs", "1", "--minibatch", "8",
            "--eval-games", "2", "--max-turns", "3", "--no-lookahead-eval"]
    main(args)
    saved = torch.load(run / "checkpoint.pt", weights_only=False)
    assert saved["model_config"]["width"] == 48
    assert saved["league"]["fixed"] == [[str(tmp_path / "teacher.pt"), 0.5]]
    assert saved["league"]["counts"]["fixed"] > 0
    assert any(k.startswith("fixed:") for k in saved["league"]["results"])
    bad = [a for a in args if a != "--init-checkpoint" and a != str(tmp_path / "student.pt")]
    bad[bad.index("--run") + 1] = str(tmp_path / "other")
    bad[bad.index("--league-fixed") + 1] = f"{tmp_path / 'missing.pt'}:0.5"
    with pytest.raises(SystemExit):
        main(bad)
