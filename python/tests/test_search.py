"""Search at play time (`pokemon_ml.search`): the equilibrium solver, the
network's odds over pairs of actions, copies of a game played a turn ahead,
and whole games with search on one side."""
import numpy as np
import pytest
import torch

from pokemon_ml import ppo
from pokemon_ml.env import N_ACTIONS, OBS_F, OBS_I, OBS_M, PREVIEW, Env, part
from pokemon_ml.model import Config as ModelConfig, Model
from pokemon_ml.search import Config, duel, joint_log_policy, search, solve


def test_the_solver_finds_known_equilibria():
    # Rock, paper, scissors: a third each, worth nothing.
    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], float)
    x, (y,), value, gap = solve([rps], 3000)
    assert np.allclose(x, 1 / 3, atol=0.02) and np.allclose(y, 1 / 3, atol=0.02)
    assert abs(value) < 0.01 and 0 <= gap < 0.01
    # A lopsided guessing game: x = (1/3, 2/3), worth 1/3.
    x, _, value, gap = solve([np.array([[3.0, -1.0], [-1.0, 1.0]])], 3000)
    assert np.allclose(x, [1 / 3, 2 / 3], atol=0.02) and abs(value - 1 / 3) < 0.01 and gap < 0.01
    # A dominated row is never played.
    x, _, _, _ = solve([np.array([[1.0, 1.0], [0.0, 0.0]])], 200)
    assert x[0] > 0.99


def test_the_solver_lets_the_other_side_know_which_world_it_is_in():
    # Row 0 is good in the first world and bad in the second, but in the second the
    # other side has a reply that makes it worse still; row 1 is safe in both.
    first = np.array([[2.0], [0.0]])
    second = np.array([[-1.0, 5.0], [0.0, 0.0]])
    x, (y1, y2), value, gap = solve([first, second], 2000)
    assert x[0] > 0.99 and abs(value - 0.5) < 0.01 and gap < 0.01
    assert y1.shape == (1,) and y2[0] > 0.99


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return Model(ModelConfig(width=32, layers=1, heads=2, ff=64)).eval()


def to_battle(env, steps=6):
    """Plays a few random steps so that every game is past Team Preview."""
    for _ in range(steps):
        env.step(env.baseline("random"))
    return env


def test_the_joint_odds_are_the_odds_act_chooses_with(pool):
    model = tiny_model(1)
    env = to_battle(Env(pool, envs=16, seed=11, open_sheets=1.0, threads=2))
    f, i, mask = ppo.to_device(env, "cpu")
    battle = (part(i, "i", "info")[:, 0] != PREVIEW).nonzero().squeeze(1)
    with torch.no_grad():
        joint = joint_log_policy(model, f[battle], i[battle], mask[battle])
        actions, logp, *_ = model.act(f[battle], i[battle], mask[battle])
    total = torch.logsumexp(joint.flatten(1), 1)
    assert torch.allclose(total, torch.zeros_like(total), atol=1e-4)
    rows = torch.arange(len(battle))
    assert torch.allclose(joint[rows, actions[:, 0], actions[:, 1]], logp, atol=1e-4)


def test_copies_of_a_game_are_played_a_turn_ahead(pool):
    env = to_battle(Env(pool, envs=8, seed=12, open_sheets=1.0, threads=2))
    info = part(env.i, "i", "info")
    g = next(g for g in range(env.envs) if info[2 * g, 0] != PREVIEW and info[2 * g, 3] == 1)
    worlds = env.raw.worlds(g, 0, 3, 99)
    assert worlds.count == 3 and worlds.view == 0
    # The searching side sees each copy exactly as it sees the game.
    f, i, m = np.zeros((3, OBS_F), np.float32), np.zeros((3, OBS_I), np.int16), np.zeros((3, OBS_M), np.uint8)
    worlds.observe(0, f, i, m)
    assert (f == env.f[2 * g]).all() and (i == env.i[2 * g]).all() and (m == env.mask[2 * g]).all()
    own = np.asarray(worlds.legal(0, 0)[:2], np.int32)
    theirs = np.asarray(worlds.legal(0, 1)[:2], np.int32)
    n = len(own) * len(theirs) * 2

    def expand(seed):
        out = np.zeros((n, OBS_F), np.float32), np.zeros((n, OBS_I), np.int16), np.zeros((n, OBS_M), np.uint8), np.zeros(n, np.float32)
        worlds.expand(0, own, theirs, 2, seed, *out)
        return out

    a, b = expand(5), expand(5)
    assert all((x == y)[~np.isnan(x)].all() if x.dtype.kind == "f" else (x == y).all() for x, y in zip(a, b))
    outcome = a[3]
    assert np.isin(outcome[~np.isnan(outcome)], [-1, 0, 1]).all()
    # A turn later the side is still the one observed, and at a decision of the battle.
    assert (part(a[1], "i", "info")[:, 0] != PREVIEW).all()
    with pytest.raises(Exception):
        bad = np.full((1, 2), N_ACTIONS - 1, np.int32)
        worlds.expand(0, bad, bad, 1, 0, *[x[:1] for x in expand(5)])


def test_search_plays_a_legal_pair_and_whole_games(pool):
    model = tiny_model(2)
    config = Config(candidates=3, replies=3, worlds=2, dice=1, iterations=50)
    env = to_battle(Env(pool, envs=8, seed=13, open_sheets=1.0, threads=2))
    info = part(env.i, "i", "info")
    rng = np.random.default_rng(0)
    searched = 0
    for g in range(env.envs):
        row = 2 * g + 1
        if info[row, 0] == PREVIEW or info[row, 3] != 1:
            continue
        pair, found = search(model, env, g, 1, env.f[row], env.i[row], env.mask[row], config, 7, rng, "cpu")
        legal = {tuple(p) for p in env.raw.worlds(g, 1, 1, 0).legal(0, 1)}
        assert tuple(int(a) for a in pair) in legal
        if found["searched"]:
            searched += 1
            assert 1 <= found["candidates"] <= 3 and found["gap"] >= -1e-9 and -1 <= found["value"] <= 1
    assert searched > 0
    for privileged in (False, True):
        env = Env(pool, envs=4, seed=14, open_sheets=0.5, threads=2, max_turns=30)
        report = duel(env, model, tiny_model(3), 6, Config(2, 2, 1, 1, 20, privileged=privileged), "cpu", seed=1)
        assert report["games"] == 6 and 0 <= report["score"] <= 1 and report["searched_decisions"] > 0
    report = duel(Env(pool, envs=4, seed=15, threads=2, max_turns=30), model, "greedy", 4, config, "cpu")
    assert report["games"] == 4
