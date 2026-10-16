"""Search at play time (`pokemon_ml.search`): the equilibrium solver, the
network's odds over pairs of actions, copies of a game played a turn ahead,
and whole games with search on one side."""
import numpy as np
import pytest
import torch

from pokemon_ml import ppo
from pokemon_ml.env import N_ACTIONS, OBS_F, OBS_I, OBS_M, PREVIEW, Env, part
from pokemon_ml.model import Config as ModelConfig, Model
from pokemon_ml.search import Config, choose, duel, joint_log_policy, ranked, search, solve


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


def test_without_exploration_the_policy_s_favourites_are_searched():
    rng = np.random.default_rng(0)
    logp = np.log(rng.dirichlet(np.ones(N_ACTIONS * N_ACTIONS))).reshape(N_ACTIONS, N_ACTIONS)
    legal = [(a, b) for a in range(0, 40, 3) for b in range(1, 40, 5)]
    pairs, score, explored = choose(logp, legal, 8, rng=rng)
    top, top_score = ranked(logp, legal, 8)
    assert (pairs == top).all() and np.allclose(score, top_score) and not explored.any()


def test_exploration_adds_to_the_policy_s_favourites_and_never_displaces_them():
    rng = np.random.default_rng(1)
    logp = np.full((N_ACTIONS, N_ACTIONS), -np.inf)
    legal = [(a, b) for a in range(0, 40, 3) for b in range(1, 40, 5)]
    # A policy sure of a few pairs and nearly sure the rest are worthless.
    for k, (a, b) in enumerate(legal):
        logp[a, b] = -0.1 * k if k < 8 else -30.0 - k
    favourites = [tuple(p) for p in ranked(logp, legal, 8)[0]]
    picked_with_noise, picked_uniformly = set(), set()
    for _ in range(200):
        for noise, seen in ((0.25, picked_with_noise), (0.0, picked_uniformly)):
            pairs, _, explored = choose(logp, legal, 8, extra=2, noise=noise, rng=rng)
            as_tuples = [tuple(p) for p in pairs]
            # The policy's eight, in order, then two more: nothing taken away.
            assert as_tuples[:8] == favourites and len(set(as_tuples)) == 10 and set(as_tuples) <= set(legal)
            assert list(explored) == [False] * 8 + [True] * 2
            seen.update(as_tuples[8:])
    # Both ways reach well beyond the favourites, the noise included.
    assert len(picked_with_noise) > 20 and len(picked_uniformly) > 20
    # Noise with nothing extra to choose changes nothing.
    pairs, _, explored = choose(logp, legal, 8, extra=0, noise=0.25, rng=rng)
    assert [tuple(p) for p in pairs] == favourites and not explored.any()
    # More extras than there are pairs left: every legal pair, once.
    pairs, _, _ = choose(logp, legal[:5], 3, extra=10, rng=rng)
    assert sorted(map(tuple, pairs)) == sorted(legal[:5])
    with pytest.raises(ValueError):
        Config(noise=1.5, explore=2)
    with pytest.raises(ValueError):
        Config(noise=0.25)


def test_search_and_whole_games_with_exploration(pool):
    model = tiny_model(4)
    config = Config(candidates=3, replies=3, worlds=2, dice=1, iterations=50, noise=0.25, explore=2, explore_replies=2)
    env = to_battle(Env(pool, envs=8, seed=16, open_sheets=1.0, threads=2))
    info = part(env.i, "i", "info")
    rng = np.random.default_rng(3)
    searched = 0
    for g in range(env.envs):
        row = 2 * g
        if info[row, 0] == PREVIEW or info[row, 3] != 1:
            continue
        pair, found = search(model, env, g, 0, env.f[row], env.i[row], env.mask[row], config, 9, rng, "cpu")
        legal = {tuple(p) for p in env.raw.worlds(g, 0, 1, 0).legal(0, 0)}
        assert tuple(int(a) for a in pair) in legal
        if found["searched"]:
            searched += 1
            assert found["candidates"] <= 5 and 0 <= found["explored_weight"] <= 1 + 1e-9
    assert searched > 0
    report = duel(Env(pool, envs=4, seed=17, open_sheets=1.0, threads=2, max_turns=30), model, tiny_model(5), 4, config, "cpu")
    assert report["games"] == 4 and report["mean_explored_weight"] is not None


def random_game(rng, worlds=4, rows=10, cols=10):
    payoffs = [rng.uniform(-1, 1, (rows, cols - int(rng.integers(0, 3)))) for _ in range(worlds)]
    prior = rng.dirichlet(np.full(rows, 0.3))
    priors = [rng.dirichlet(np.full(a.shape[1], 0.3)) for a in payoffs]
    return payoffs, prior, priors


def test_the_anchored_solution_is_the_policy_reweighted_by_payoff():
    # Against a single reply per world, the anchored best answer has a closed form: prior · exp(u / λ).
    rng = np.random.default_rng(5)
    payoffs = [rng.uniform(-1, 1, (6, 1)) for _ in range(3)]
    prior = rng.dirichlet(np.ones(6))
    for lam in (0.03, 0.3, 3.0):
        x, (y1, y2, y3), _, gap = solve(payoffs, 1000, lam, prior, [np.ones(1)] * 3)
        u = np.mean([a[:, 0] for a in payoffs], 0)
        floored = 0.999 * prior + 0.001 / 6
        expected = floored * np.exp((u - u.max()) / lam)
        assert np.allclose(x, expected / expected.sum(), atol=1e-6) and gap < 1e-6


def test_the_anchored_solver_converges_across_strengths():
    rng = np.random.default_rng(6)
    for lam in (0.01, 0.03, 0.1, 0.3, 1.0):
        for _ in range(10):
            payoffs, prior, priors = random_game(rng)
            x, ys, value, gap = solve(payoffs, 1000, lam, prior, priors)
            assert -1e-12 <= gap < 1e-4, (lam, gap)
            assert abs(x.sum() - 1) < 1e-9 and all(abs(y.sum() - 1) < 1e-9 for y in ys) and -1 <= value <= 1


def test_the_anchor_runs_from_the_plain_equilibrium_to_the_policy():
    rng = np.random.default_rng(7)
    payoffs, prior, priors = random_game(rng)
    plain = solve(payoffs, 3000)
    weak = solve(payoffs, 1000, 0.01, prior, priors)
    strong = solve(payoffs, 1000, 100.0, prior, priors)
    assert abs(weak[2] - plain[2]) < 0.05
    assert np.allclose(strong[0], 0.999 * prior + 0.001 / len(prior), atol=0.01)
    # Off, it is the plain solver exactly.
    assert all(np.array_equal(a, b) for a, b in zip(solve(payoffs, 500)[0], solve(payoffs, 500, 0.0, prior, priors)[0]))


def test_the_anchor_does_not_chase_a_small_edge_the_policy_doubts():
    # The policy is sure of row 0; row 3 looks one point better, which noise alone could do.
    payoffs = [np.array([[0.0], [-0.2], [-0.2], [0.01]])]
    prior = np.array([0.97, 0.01, 0.01, 0.01])
    plain_x = solve(payoffs, 1000)[0]
    anchored_x = solve(payoffs, 1000, 0.03, prior, [np.ones(1)])[0]
    assert plain_x[3] > 0.99
    assert anchored_x[0] > 0.9 and anchored_x[3] < 0.05
    # A clear edge still wins it over.
    payoffs = [np.array([[0.0], [-0.2], [-0.2], [0.3]])]
    assert solve(payoffs, 1000, 0.03, prior, [np.ones(1)])[0][3] > 0.99


def test_search_and_whole_games_with_the_anchor(pool):
    model = tiny_model(6)
    config = Config(candidates=3, replies=3, worlds=2, dice=1, iterations=200, anchor=0.03, explore=1, explore_replies=1)
    env = to_battle(Env(pool, envs=8, seed=18, open_sheets=1.0, threads=2))
    info = part(env.i, "i", "info")
    rng = np.random.default_rng(4)
    searched = 0
    for g in range(env.envs):
        row = 2 * g
        if info[row, 0] == PREVIEW or info[row, 3] != 1:
            continue
        pair, found = search(model, env, g, 0, env.f[row], env.i[row], env.mask[row], config, 11, rng, "cpu")
        assert tuple(int(a) for a in pair) in {tuple(p) for p in env.raw.worlds(g, 0, 1, 0).legal(0, 0)}
        if found["searched"]:
            searched += 1
            assert -1e-9 <= found["gap"] < 1e-3
    assert searched > 0
    report = duel(Env(pool, envs=4, seed=19, open_sheets=1.0, threads=2, max_turns=30), model, tiny_model(7), 4, config, "cpu")
    assert report["games"] == 4
    with pytest.raises(ValueError):
        Config(anchor=-1)


def test_a_strong_anchor_plays_what_the_policy_would(pool):
    # The search hands the solver the policy's odds: held hard enough to them, it plays the policy's favourite.
    model = tiny_model(8)
    config = Config(candidates=4, replies=3, worlds=2, dice=1, iterations=300, anchor=1000.0, pick="best", explore=2)
    env = to_battle(Env(pool, envs=8, seed=20, open_sheets=1.0, threads=2))
    info = part(env.i, "i", "info")
    rng = np.random.default_rng(5)
    searched = 0
    for g in range(env.envs):
        row = 2 * g
        if info[row, 0] == PREVIEW or info[row, 3] != 1:
            continue
        _, found = search(model, env, g, 0, env.f[row], env.i[row], env.mask[row], config, 12, rng, "cpu")
        if found["searched"]:
            searched += 1
            assert found["policy_top"] and not found["played_explored"]
    assert searched > 0
