import json
from dataclasses import asdict
import numpy as np
import pytest
import torch
from pokemon_ml import ppo, league, auxiliary, checkpoint
from pokemon_ml.env import Env, part
from pokemon_ml.features import History, policy_features
from pokemon_ml.model import Model, Config, preset
from pokemon_ml.team_splits import split, assert_disjoint
from pokemon_ml.evaluation import play, calibration, interval


def small(**kwargs):
    return Config(width=32, layers=1, heads=2, ff=64, **kwargs)


def test_presets_and_speed_ablation_preserve_known_stats(pool):
    assert preset().width == 128
    assert preset("medium").layers == 4 and preset("large").width == 384
    env = Env(pool, 4, threads=2)
    f, i, m = ppo.to_device(env, "cpu")
    removed = policy_features(f, False)
    assert torch.equal(
        part(removed, "f", "mon_feats")[..., :46], part(f, "f", "mon_feats")[..., :46]
    )
    assert not part(removed, "f", "mon_feats")[..., -6:].any()
    assert not part(removed, "f", "act_feats")[..., -6:].any()
    assert torch.equal(f, torch.from_numpy(env.f))
    model = Model(small(use_speed=False)).eval()
    altered = f.clone()
    part(altered, "f", "mon_feats")[..., -6:] = torch.randn_like(
        part(altered, "f", "mon_feats")[..., -6:]
    )
    part(altered, "f", "act_feats")[..., -6:] = torch.randn_like(
        part(altered, "f", "act_feats")[..., -6:]
    )
    assert torch.allclose(model.encode(f, i)[0], model.encode(altered, i)[0])


@pytest.mark.parametrize(
    "options",
    [
        {"semantic_actions": True},
        {"joint_preview": True},
        {"history": 4},
        {"opponent_prediction": True},
        {"opponent_prediction": True, "prediction_feedback": True},
        {"hidden_prediction": True},
    ],
)
def test_experimental_policies_have_matching_action_likelihoods(pool, options):
    env = Env(pool, 4, threads=2)
    model = Model(small(**options)).eval()
    history = History(8, model.config.history, "cpu")
    for _ in range(5):
        f, i, m = ppo.to_device(env, "cpu")
        h = history.tensors() if model.config.history else None
        actions, lp, value, m0, m1 = model.act(f, i, m, history=h)
        again, _, _, v = model.evaluate(f, i, m0, m1, actions, history=h)
        assert torch.allclose(lp, again, atol=1e-5) and torch.allclose(
            value, v, atol=1e-5
        )
        history.append(f, i)
        env.step(actions.numpy())
        history.reset(torch.as_tensor(env.done).bool().repeat_interleave(2))


def test_history_is_player_visible_episode_local_and_masks_padding(pool):
    env = Env(pool, 2, threads=2)
    f, i, _ = ppo.to_device(env, "cpu")
    hist = History(4, 3, "cpu")
    hist.append(f, i)
    assert hist.valid.sum() == 4
    roster = part(i.long(), "i", "mon_ids")[..., 0]
    species = torch.cat([torch.zeros_like(roster[:, :1]), roster], 1)
    expected = species.gather(1, part(i.long(), "i", "act_ids")[..., 0])
    assert torch.equal(hist.i[:, -1, ::2], expected)
    # History construction never receives the opposite view or private labels.
    old = hist.f.clone()
    hist.reset(torch.tensor([True, True, False, False]))
    assert not hist.valid[:2].any() and torch.equal(hist.f[2:], old[2:])
    model = Model(small(history=3)).eval()
    a = model.encode(f, i, hist.tensors())[0]
    hist.f[~hist.valid] = 999
    hist.i[~hist.valid] = 1
    b = model.encode(f, i, hist.tensors())[0]
    assert torch.allclose(a, b, atol=1e-5)


def test_training_labels_do_not_enter_policy_and_use_move_identity(pool):
    env = Env(pool, 4, threads=2)
    env.step(env.baseline("greedy"))
    f, i, m = ppo.to_device(env, "cpu")
    actions = torch.tensor(env.baseline("greedy"), dtype=torch.long)
    target, valid, private = auxiliary.labels(i, actions)
    assert target.shape == (8, 2) and private.shape == (8, 6, 8)
    assert torch.equal(private[0], part(i.long(), "i", "mon_ids")[1, :6])
    model = Model(small(opponent_prediction=True, hidden_prediction=True)).eval()
    outputs = model.auxiliary(f, i)
    losses = auxiliary.loss(outputs, i, target, valid, private)
    sum(losses.values()).backward()
    assert model.opponent_out.weight.grad.abs().sum() > 0
    assert torch.isfinite(torch.stack(list(losses.values()))).all()


def test_league_keeps_opponents_and_excludes_off_policy_rows(pool, tmp_path):
    torch.set_num_threads(2)
    run = str(tmp_path)
    model = Model(small())
    league.save_snapshot(run, 0, model)
    env = Env(pool, 8, threads=2)
    opponents = league.TrainingLeague(run, 8, "cpu", weights=(0, 1, 0))
    roll = ppo.Rollout(4, 16, "cpu")
    ppo.collect(env, model, roll, opponents)
    assert sum(roll.valid[0]) == 8
    assert all(x == "snapshot:000000" for x in opponents.assigned)
    before = opponents.assigned.copy()
    opponents.assign(np.zeros(8, dtype=bool))
    assert before == opponents.assigned
    cfg = ppo.Config(steps=4, minibatch=64)
    log = ppo.update(model, torch.optim.Adam(model.parameters()), roll, cfg)
    assert 0 < log["decisions"] <= 32 and log["kl"] >= -1e-7
    state = opponents.state_dict()
    resumed = league.TrainingLeague(run, 8, "cpu", weights=(0, 1, 0), state=state)
    opponents.assign(np.ones(8, dtype=bool))
    resumed.assign(np.ones(8, dtype=bool))
    assert opponents.assigned == resumed.assigned and np.array_equal(
        opponents.side, resumed.side
    )


def test_all_experiments_train_with_a_league(pool, tmp_path):
    torch.set_num_threads(2)
    model = Model(
        small(
            history=3,
            semantic_actions=True,
            joint_preview=True,
            opponent_prediction=True,
            hidden_prediction=True,
            prediction_feedback=True,
        )
    )
    run = str(tmp_path)
    league.save_snapshot(run, 0, model)
    env = Env(pool, 8, threads=2)
    opponents = league.TrainingLeague(run, 8, "cpu", weights=(0, 0, 1))
    roll = ppo.Rollout(8, 16, "cpu", history=3, auxiliary_labels=True)
    ppo.collect(env, model, roll, opponents)
    log = ppo.update(
        model,
        torch.optim.Adam(model.parameters()),
        roll,
        ppo.Config(steps=8, minibatch=64),
    )
    assert log["opponent_loss"] > 0 and log["hidden_loss"] > 0
    assert np.isfinite(list(log.values())).all()


def test_spread_variants_stay_together_and_overlap_is_rejected(pool):
    data = json.load(open(pool))
    variant = json.loads(json.dumps(data["teams"][0]))
    variant["team"][0]["item"] = "Leftovers"
    data["teams"].append(variant)
    a, b = split(data, seed=4)
    assert_disjoint(a, b)
    assert (
        data["teams"][0] in a["teams"]
        if variant in a["teams"]
        else data["teams"][0] in b["teams"]
    )
    with pytest.raises(ValueError):
        assert_disjoint(a, a)


def test_evaluation_uses_actual_game_counts_and_reports_calibration(pool):
    model = Model(small(history=2)).eval()
    env = Env(pool, 4, seed=22, max_turns=1, threads=2)
    report = play(env, model, "random", 5, "cpu", trace_games=1)
    assert report["games"] >= 5
    assert sum(x["games"] for x in report["per_team"].values()) == report["games"]
    assert report["calibration"]["samples"] > 0 and len(report["traces"]) == 1
    assert report["timeouts"] == report["games"]
    perfect = calibration([-1, 1, 0], [-1, 1, 0])
    assert perfect["brier"] == 0 and perfect["ece"] == 0
    assert interval(50, 50, 0)[0] < 0.5 < interval(50, 50, 0)[1]


def test_checkpoint_rng_and_contract_are_restored():
    torch.manual_seed(5)
    state = checkpoint.rng_state()
    next_value = torch.rand(4)
    checkpoint.restore_rng(state)
    assert torch.equal(next_value, torch.rand(4))
    checkpoint.verify_contract({"contract": checkpoint.contract()})
    with pytest.raises(ValueError):
        checkpoint.verify_contract({"contract": {"version": 1, "sha256": "bad"}})


def test_matched_evaluation_keeps_initial_team_pairings_across_policies(pool):
    torch.set_num_threads(2)
    first, second = Model(small()).eval(), Model(small()).eval()
    a = play(Env(pool, 4, seed=17, max_turns=2, threads=2), first, "random", 11, "cpu")
    b = play(Env(pool, 4, seed=17, max_turns=2, threads=2), second, "greedy", 11, "cpu")
    assert a["games"] == b["games"] == 11
    assert {k: v["games"] for k, v in a["matchups"].items()} == {
        k: v["games"] for k, v in b["matchups"].items()
    }


def test_showdown_history_is_isolated_per_room(pool, tmp_path):
    from pokemon_ml.showdown import ModelPolicy

    env = Env(pool, 1, threads=2)
    model = Model(small(history=3)).eval()
    path = tmp_path / "model.pt"
    torch.save(
        dict(
            model=model.state_dict(),
            model_config=asdict(model.config),
            contract=checkpoint.contract(),
        ),
        path,
    )
    shared = ModelPolicy(path)
    a, b = shared.fork(), shared.fork()
    a.act(env.f[0], env.i[0], env.mask[0], True)
    assert a.net is b.net and a.history.valid.sum() == 1 and b.history.valid.sum() == 0


def test_benchmark_reports_positive_elapsed_times(pool, tmp_path):
    from pokemon_ml import benchmark

    torch.set_num_threads(2)
    out = tmp_path / "benchmark.json"
    benchmark.main(
        [
            "--pool",
            pool,
            "--out",
            str(out),
            "--sizes",
            "small",
            "--envs",
            "2",
            "--threads",
            "2",
            "--steps",
            "2",
            "--device",
            "cpu",
        ]
    )
    report = json.loads(out.read_text())["experiments"][0]
    assert (
        report["rollout_seconds"] > 0
        and report["update_seconds"] > 0
        and report["learner_decisions_per_second"] > 0
    )


def test_resumed_league_uses_saved_manifest_until_new_snapshot_registered(
    pool, tmp_path
):
    model = Model(small()).eval()
    run = str(tmp_path)
    league.save_snapshot(run, 0, model)
    original = league.TrainingLeague(run, 4, "cpu", weights=(0, 1, 0))
    saved = original.state_dict()
    # A snapshot left behind by work after the saved checkpoint must not enter resume.
    league.save_snapshot(run, 1, model)
    resumed = league.TrainingLeague(run, 4, "cpu", weights=(0, 1, 0), state=saved)
    resumed.assign(np.ones(4, dtype=bool))
    assert resumed.assigned == ["snapshot:000000"] * 4
    assert resumed.state_dict()["snapshots"] == ["000000"]
    resumed.register_snapshot("000001")
    assert resumed.state_dict()["snapshots"] == ["000000", "000001"]
