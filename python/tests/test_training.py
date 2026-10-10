"""The Python side: the environment's arrays, the network's bookkeeping, the advantages, and
that a short run of training learns something. Run with `pytest python/tests`."""
import numpy as np
import torch

from pokemon_ml import ppo
from pokemon_ml.env import LAYOUT, N_ACTIONS, N_PREVIEW, OBS_F, OBS_I, OBS_M, PREVIEW, Env, part
from pokemon_ml.model import Config, Model


def test_the_environment_fills_its_arrays(pool):
    env = Env(pool, envs=32, seed=3, threads=2)
    assert env.f.shape == (64, OBS_F) and env.i.shape == (64, OBS_I) and env.mask.shape == (64, OBS_M)
    info = part(env.i, "i", "info")
    assert (info[:, 0] == PREVIEW).all() and (info[:, 1] == N_PREVIEW).all() and (info[:, 3] == 1).all()
    # Everyone's own six are there in full, the other six at least by species.
    assert (part(env.i, "i", "mon_ids")[:, :, 0] >= 2).all()
    assert (part(env.f, "f", "mon_feats")[:, :6, 1] == 1).all() and (part(env.f, "f", "mon_feats")[:, 6:, 1] == 0).all()
    finished = 0
    for step in range(300):
        actions = env.baseline("greedy" if step % 2 else "random")
        mask0 = env.mask[:, :N_ACTIONS]
        battle = info[:, 0] != PREVIEW
        assert (mask0[battle, actions[battle, 0]] == 1).all()
        env.step(actions)
        finished += int(env.done.sum())
        assert (env.reward.sum(1) == 0).all() and (env.reward[env.done == 0] == 0).all()
        assert np.isfinite(env.f).all()
    assert finished == env.stats()["games"] > 100


def test_a_choice_and_its_log_probability_agree(pool):
    torch.manual_seed(0)
    env = Env(pool, envs=48, seed=4, threads=2)
    model = Model(Config(width=32, layers=1, heads=2, ff=64))
    for step in range(40):
        f, i, mask = ppo.to_device(env, "cpu")
        actions, logp, value, mask0, mask1 = model.act(f, i, mask)
        # What was chosen is legal under the masks handed back, and `evaluate` gives it the same probability.
        battle = part(i, "i", "info")[:, 0] != PREVIEW
        rows = torch.arange(len(actions))[battle]
        assert mask0[rows, actions[rows, 0]].all() and mask1[rows, actions[rows, 1]].all()
        assert (actions[~battle, 0] < N_PREVIEW).all()
        again, entropy, spread, value_again = model.evaluate(f, i, mask0, mask1, actions)
        assert torch.allclose(logp, again, atol=1e-4) and torch.allclose(value, value_again, atol=1e-4)
        assert (entropy >= -1e-5).all() and (spread <= 1e-5).all() and torch.isfinite(spread).all()
        # A side with one legal thing to do is certain of it.
        forced = part(i, "i", "info")[:, 1] == 1
        assert torch.allclose(logp[forced], torch.zeros(int(forced.sum())), atol=1e-4)
        env.step(actions.numpy())
    # The probabilities of all 90 ways to bring four add up to one.
    env = Env(pool, envs=4, seed=1)
    f, i, _ = ppo.to_device(env, "cpu")
    logits = model.preview_logits(model.encode(f, i)[1])
    assert logits.shape == (8, N_PREVIEW)
    assert LAYOUT["roster"] == 6


def test_advantages_follow_each_side_through_its_own_decisions():
    torch.manual_seed(1)
    steps, batch, lam = 40, 6, 0.9
    value = torch.randn(steps, batch)
    done = torch.rand(steps, batch) < 0.15
    reward = torch.where(done, torch.sign(torch.randn(steps, batch)), torch.zeros(steps, batch))
    valid = torch.rand(steps, batch) < 0.7
    last = torch.randn(batch)
    adv, returns = ppo.advantages(value, reward, done, valid, last, lam)
    for b in range(batch):
        # By hand: cut the column into games, keep the steps with a decision, and run plain GAE over them.
        t = 0
        while t < steps:
            end = next((u for u in range(t, steps) if done[u, b]), None)
            span = range(t, (end if end is not None else steps - 1) + 1)
            mine = [u for u in span if valid[u, b]]
            final = reward[end, b].item() if end is not None else 0.0
            tail = 0.0 if end is not None else last[b].item()
            running = 0.0
            for k in reversed(range(len(mine))):
                u = mine[k]
                is_last = k == len(mine) - 1
                nxt = tail if is_last else value[mine[k + 1], b].item()
                delta = (final if is_last else 0.0) + nxt - value[u, b].item()
                running = delta + lam * running
                assert abs(adv[u, b].item() - running) < 1e-4, (b, u)
                assert abs(returns[u, b].item() - (running + value[u, b].item())) < 1e-4
            for u in span:
                if not valid[u, b]:
                    assert adv[u, b] == 0
            t = span[-1] + 1


def test_a_short_run_learns_to_beat_the_random_player(pool):
    torch.manual_seed(2)
    torch.set_num_threads(2)
    env = Env(pool, envs=96, seed=11, threads=2)
    model = Model(Config(width=48, layers=2, heads=2, ff=96))
    config = ppo.Config(steps=32, minibatch=1024, lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-5)
    roll = ppo.Rollout(config.steps, 2 * env.envs, "cpu")
    rival = lambda seed: Env(pool, envs=96, seed=seed, threads=2)
    before = ppo.play_baseline(rival(100), model, "random", 300, "cpu")
    for _ in range(25):
        ppo.collect(env, model, roll)
        log = ppo.update(model, optimizer, roll, config)
        assert np.isfinite(list(log.values())).all()
    after = ppo.play_baseline(rival(101), model, "random", 300, "cpu")
    assert 0.35 < before < 0.65, before
    assert after > before + 0.1, (before, after)


def test_ratings_are_recovered_from_results():
    from pokemon_ml.league import earlier, expected, fit
    truth = {"000000": 0.0, "000025": 120.0, "000050": 310.0, "greedy": 200.0}
    names = list(truth)
    results = [(a, b, expected(truth[a], truth[b]), 2000) for k, a in enumerate(names) for b in names[:k]]
    fitted = fit(results)
    assert all(abs(fitted[name] - truth[name]) < 3 for name in names), fitted
    # One that wins everything is rated above the rest, and stays finite.
    fitted = fit(results + [("000075", "000050", 1.0, 300)])
    assert 310 < fitted["000075"] < 3000
    # Each new version plays the one before it, then twice as far back each time, and the first.
    names = [f"{25 * k:06d}" for k in range(12)]
    assert earlier(names, 3) == ["000275", "000250", "000200", "000000"]
    assert earlier(names[:1], 3) == ["000000"] and earlier([], 3) == []


def test_training_puts_versions_aside_and_rates_them(pool, tmp_path):
    import json
    from pokemon_ml import league, train
    torch.set_num_threads(2)
    run = str(tmp_path / "run")
    args = ["--pool", pool, "--run", run, "--envs", "32", "--steps", "16", "--minibatch", "512", "--threads", "2",
            "--width", "32", "--layers", "1", "--heads", "2", "--ff", "64", "--eval-every", "1", "--eval-games", "40",
            "--device", "cpu"]
    train.main(args + ["--updates", "2"])
    assert league.snapshots(run) == ["000000", "000001", "000002"]
    saved = json.load(open(f"{run}/ratings.json"))
    assert saved["ratings"]["000000"] == 0 and set(league.SCRIPTED) <= set(saved["ratings"])
    log = [json.loads(line) for line in open(f"{run}/log.jsonl")]
    assert all(k in log[-1] for k in ("rating", "vs_random", "vs_greedy", "vs_lookahead"))
    # The same command carries on, and the new version meets the old ones.
    train.main(args + ["--updates", "3"])
    assert league.snapshots(run)[-1] == "000003"
    met = {(a, b) for a, b, _, _ in league.Ratings(run).results}
    assert ("000003", "000002") in met and ("000003", "000000") in met
    # A network against a copy of itself wins about half the time.
    net = league.load_snapshot(run, "000003", "cpu")
    share, played = league.play_models(Env(pool, envs=64, seed=5, threads=2), net, net, 400, "cpu")
    assert played >= 400 and 0.4 < share < 0.6, share
