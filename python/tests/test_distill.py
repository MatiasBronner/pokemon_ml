"""Distillation (`pokemon_ml.distill`)."""
import json

import numpy as np
import pytest
import torch

from pokemon_ml.distill import Games, divergence, favourite, held_out_positions, main, policy, train
from pokemon_ml.env import Env, N_ACTIONS, PREVIEW, part
from pokemon_ml.evaluation import load
from pokemon_ml.expert_train import save
from pokemon_ml.model import Config, Model
from pokemon_ml.search import joint_log_policy


def tiny(seed, **sizes):
    torch.manual_seed(seed)
    return Model(Config(**{**dict(width=32, layers=1, heads=2, ff=64), **sizes})).eval()


@pytest.fixture(scope="module")
def positions(pool):
    """Positions with a choice to make, Team Preview and battle both."""
    env = Env(pool, envs=8, seed=3, open_sheets=0.5, threads=2, max_turns=30)
    f, i, mask = held_out_positions(env, tiny(1), 400, "cpu")
    phase = part(i, "i", "info")[:, 0]
    assert (phase == PREVIEW).any() and (phase != PREVIEW).any()
    return f, i, mask


def exact_kl(teacher, student, f, i, mask):
    """KL(teacher || student) the slow way: over the full table of pairs, or Team Preview's choices."""
    with torch.no_grad():
        jt, js = joint_log_policy(teacher, f, i, mask), joint_log_policy(student, f, i, mask)
        pt = torch.log_softmax(teacher.preview_logits(teacher.encode(f, i)[1]), -1)
        ps = torch.log_softmax(student.preview_logits(student.encode(f, i)[1]), -1)
    legal = torch.isfinite(jt)
    battle = (jt.exp() * torch.where(legal, jt - js, torch.zeros_like(jt))).flatten(1).sum(-1)
    preview = (pt.exp() * (pt - ps)).sum(-1)
    return torch.where(part(i, "i", "info")[:, 0] == PREVIEW, preview, battle), jt


def test_the_divergence_is_the_kl_over_the_whole_choice(positions):
    teacher, student = tiny(1), tiny(2)
    with torch.no_grad():
        t, s = policy(teacher, *positions), policy(student, *positions)
        fast = divergence(t, s)
    slow, table = exact_kl(teacher, student, *positions)
    assert torch.allclose(fast, slow, atol=1e-4, rtol=1e-4)
    assert (fast > -1e-6).all() and (fast > 1e-3).float().mean() > 0.8  # 0 only where one choice is legal
    # Against itself it is nothing.
    with torch.no_grad():
        assert float(divergence(t, policy(teacher, *positions)).abs().max()) < 1e-5
    # The likeliest choice: in battle, the likeliest pair of the full table.
    battle = part(positions[1], "i", "info")[:, 0] != PREVIEW
    pick = favourite(t)
    assert (pick[battle] == table[battle].flatten(1).argmax(-1)).all()
    assert (pick[~battle] < 90).all()


def test_games_keep_who_plays_a_side_for_the_whole_game(pool):
    env = Env(pool, envs=6, seed=4, open_sheets=0.5, threads=2, max_turns=20)
    games = Games(env, tiny(1), tiny(2), 0.5, np.random.default_rng(0), "cpu")
    before = games.by_student.copy()
    changed_mid_game = False
    for _ in range(60):
        was = games.by_student.copy()
        games.step()
        over = np.repeat(env.done.astype(bool), 2)
        changed_mid_game |= bool((games.by_student != was)[~over].any())
    assert not changed_mid_game and games.games > 0 and games.positions > 0
    assert before.any() and not before.all()


def test_the_student_learns_the_teacher(pool, positions, tmp_path):
    teacher, student = tiny(1), tiny(2)
    env = Env(pool, envs=8, seed=5, open_sheets=0.5, threads=2, max_turns=30)
    summary = train(teacher, student, env, positions, tmp_path, "cpu", steps=300, lr=3e-3, batch=128,
                    buffer=4096, reuse=4, warmup=20, eval_every=100, seed=0)
    assert summary["final"]["policy_kl"] < 0.3 * summary["baseline"]["policy_kl"]
    assert summary["final"]["value_mse"] < 0.3 * summary["baseline"]["value_mse"]
    assert summary["final"]["agreement"] > summary["baseline"]["agreement"]
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert log[0]["baseline"] and log[-1]["step"] == 300 and log[-1]["positions"] > 0
    for name in ("best.pt", "checkpoint.pt"):
        loaded = load(str(tmp_path / name), "cpu")
        assert loaded.config == student.config


def test_the_command_line_makes_a_bigger_student_with_the_teacher_s_features(pool, tmp_path):
    teacher = tiny(1, use_speed=False)
    save(tmp_path / "teacher.pt", teacher, {})
    out = tmp_path / "run"
    main(["--teacher", str(tmp_path / "teacher.pt"), "--model-size", "small", "--pool", pool, "--out", str(out),
          "--steps", "4", "--batch", "32", "--buffer", "256", "--envs", "4", "--threads", "2", "--eval-every", "2",
          "--held-out", "64", "--warmup", "1", "--device", "cpu"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["steps"] == 4
    student = load(str(out / "checkpoint.pt"), "cpu")
    assert student.config.width == 128 and student.config.use_speed is False
    config = json.loads((out / "config.json").read_text())
    assert config["teacher_sha256"] and config["parameters"] > 0
    with pytest.raises(SystemExit):  # never into a directory that holds a run
        main(["--teacher", str(tmp_path / "teacher.pt"), "--model-size", "small", "--pool", pool, "--out", str(out)])
