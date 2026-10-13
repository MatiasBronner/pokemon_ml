import numpy as np
import pytest
import torch
from pokemon_ml import ppo
from pokemon_ml.env import Env
from pokemon_ml.model import Model, Config
from pokemon_ml.runtime import Runtime, configure
from pokemon_ml.tune_learner import validation, candidates


def test_runtime_keeps_checkpoint_keys_and_eager_likelihoods(pool):
    torch.set_num_threads(2)
    env = Env(pool, 4, threads=2)
    model = Model(Config(width=32, layers=1, heads=2, ff=64)).eval()
    keys = set(model.state_dict())
    reference = Model(model.config).eval()
    reference.load_state_dict(model.state_dict())
    configure(model, Runtime(attention="sdpa", deferred_metrics=True))
    assert set(model.state_dict()) == keys
    assert validation(model, env, reference)["likelihood_error"] < 1e-4
    env.step(env.baseline("greedy"))
    assert (
        validation(model, env, reference)["fp32_log_probability_max_difference"] < 1e-3
    )


def test_deferred_metrics_preserve_parameter_updates(pool):
    torch.set_num_threads(2)
    torch.manual_seed(4)
    env = Env(pool, 4, threads=2)
    first = Model(Config(width=32, layers=1, heads=2, ff=64)).eval()
    second = Model(first.config).eval()
    second.load_state_dict(first.state_dict())
    cfg = ppo.Config(steps=4, minibatch=8)
    roll = ppo.Rollout(4, 8, "cpu")
    ppo.collect(env, first, roll)
    configure(second, Runtime(deferred_metrics=True))
    a = torch.optim.Adam(first.parameters(), lr=cfg.lr)
    b = torch.optim.Adam(second.parameters(), lr=cfg.lr)
    torch.manual_seed(91)
    log_a = ppo.update(first, a, roll, cfg)
    torch.manual_seed(91)
    log_b = ppo.update(second, b, roll, cfg)
    for x, y in zip(first.parameters(), second.parameters()):
        assert torch.equal(x, y)
    for k in log_a:
        assert np.isclose(log_a[k], log_b[k], rtol=1e-5, atol=1e-6), (
            k,
            log_a[k],
            log_b[k],
        )


def test_unsupported_precision_is_rejected_and_cpu_candidates_are_safe():
    model = Model(Config(width=32, layers=1, heads=2, ff=64))
    with pytest.raises(ValueError, match="Ampere"):
        configure(model, Runtime(precision="bf16"))
    assert all(
        option.precision == "fp32"
        for _, _, option in candidates(torch.device("cpu"), [32, 64], False)
    )


def test_tuning_worker_does_not_change_source_checkpoint(pool, tmp_path):
    import argparse, json
    from pokemon_ml import checkpoint
    from pokemon_ml.tune_learner import worker

    model = Model(Config(width=32, layers=1, heads=2, ff=64))
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "source.pt"
    torch.save(
        dict(
            model=model.state_dict(),
            model_config=model.save_config(),
            ppo_config=ppo.Config(steps=2, minibatch=4).__dict__,
            optimizer=optimizer.state_dict(),
            contract=checkpoint.contract(),
        ),
        path,
    )
    before = checkpoint.file_hash(path)
    args = argparse.Namespace(
        device="cpu",
        threads=2,
        seed=1,
        worker_out=str(tmp_path / "results"),
        runtime_json=json.dumps(dict(deferred_metrics=True)),
        checkpoint=str(path),
        pool=pool,
        envs=2,
        steps=2,
        minibatch=4,
        league_run=None,
        league_manifest=None,
        repeats=1,
        profile=False,
    )
    report = worker(args)
    assert checkpoint.file_hash(path) == before and report["status"] == "passed"
    assert report["median_decisions_per_second"] > 0


def test_diagnostics_distinguish_value_drift_without_weakening_gate(pool, monkeypatch):
    env = Env(pool, 2, threads=2)
    model = Model(Config(width=32, layers=1, heads=2, ff=64)).eval()
    reference = Model(model.config).eval()
    reference.load_state_dict(model.state_dict())
    original = model.evaluate

    def shifted(*args, **kwargs):
        lp, ent, spread, value = original(*args, **kwargs)
        return lp, ent, spread, value + 0.01

    monkeypatch.setattr(model, "evaluate", shifted)
    with pytest.raises(ValueError, match="max_value_error="):
        validation(model, env, reference)
    result = validation(model, env, reference, strict=False)
    assert not result["consistency_passed"]
    assert (
        result["value_consistency_error"] > 0.009 and result["likelihood_error"] < 1e-4
    )
