"""Expert iteration's training step (`pokemon_ml.expert_train`)."""
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from pokemon_ml.env import Env
from pokemon_ml.evaluation import load
from pokemon_ml.expert_data import generate, parse_opponent
from pokemon_ml.expert_train import load_data, main, pair_log_probs, split_games, to_batch, train
from pokemon_ml.model import Config as ModelConfig, Model
from pokemon_ml.search import Config, duel, joint_log_policy


def tiny(seed):
    torch.manual_seed(seed)
    return Model(ModelConfig(width=32, layers=1, heads=2, ff=64)).eval()


@pytest.fixture(scope="module")
def data_dir(pool, tmp_path_factory):
    """A little data from a tiny network with search, against itself and the greedy player."""
    out = tmp_path_factory.mktemp("expert")
    model = tiny(1)
    league = [parse_opponent(x) for x in ("self+search:0.5", "greedy:0.5")]
    config = Config(candidates=4, replies=3, worlds=2, dice=1, iterations=100, explore=2, noise=0.25)
    env = Env(pool, envs=8, seed=9, open_sheets=1.0, threads=2, max_turns=40)
    torch.manual_seed(0)
    totals = generate(env, model, league, {"self": model}, 40, config, out, "cpu", seed=2, shard_size=400)
    (out / "manifest.json").write_text(json.dumps(dict(totals, checkpoint_sha256="test")))
    torch.save(dict(model_config=asdict(model.config), model=model.state_dict()), out / "learner.pt")
    return out


def test_the_probability_of_a_searched_pair_is_the_one_act_chooses_with(data_dir):
    data, _ = load_data([data_dir])
    model = tiny(1)
    rows = np.flatnonzero(data["searched"])[:64]
    b = to_batch(data, rows, "cpu")
    with torch.no_grad():
        mine = pair_log_probs(model, b["f"], b["i"], b["mask"], b["candidates"])
        table = joint_log_policy(model, b["f"], b["i"], b["mask"])
    for r in range(len(rows)):
        for k, (a0, a1) in enumerate(b["candidates"][r].tolist()):
            if a0 >= 0:
                assert abs(float(mine[r, k]) - float(table[r, a0, a1])) < 1e-4


def test_held_out_games_are_whole_games(data_dir):
    data, manifests = load_data([data_dir])
    assert manifests[0]["games"] == 40 and len(data["f"]) == manifests[0]["rows"]
    held = split_games(data["game"], 0.25, seed=1)
    assert 0 < held.sum() < len(held)
    assert not set(data["game"][held]) & set(data["game"][~held])
    # Deterministic, and two directories never share a game id.
    assert (split_games(data["game"], 0.25, seed=1) == held).all()
    both, _ = load_data([data_dir, data_dir])
    assert len(np.unique(both["game"])) == 2 * len(np.unique(data["game"]))


def test_training_learns_the_search_s_choices_and_the_outcomes(data_dir, tmp_path):
    # On data this small the network can learn it outright: what it fits must improve a lot.
    data, _ = load_data([data_dir])
    model = load(str(data_dir / "learner.pt"), "cpu")
    summary = train(model, data, tmp_path, "cpu", lr=3e-3, epochs=60, batch=256, holdout=0.2, warmup=10,
                    eval_every=5, seed=0)
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert log[0]["baseline"] and log[-1]["step"] == summary["steps"] and len(log) > 3
    first, last = log[1], log[-1]
    # Training rows: the policy closes most of its distance to the search (cross-entropy cannot
    # fall below the search's own entropy, so the distance is the KL divergence), and the value
    # learns the outcomes.
    assert last["train_policy_kl"] < 0.4 * first["train_policy_kl"]
    assert last["train_policy_kl"] <= last["train_policy_loss"] + 1e-6
    assert last["train_value_loss"] < 0.3 * first["train_value_loss"]
    assert last["held_out"]["policy_kl"] is not None and last["held_out"]["policy_kl"] >= -1e-6
    # Checkpoints load where any checkpoint does, and play.
    for name in ("checkpoint.pt", "best.pt"):
        net = load(str(tmp_path / name), "cpu")
        assert net.config == model.config


def test_the_new_network_with_search_plays_the_old_one_with_search(data_dir, pool, tmp_path):
    new, old = tiny(2), tiny(3)
    config = Config(candidates=3, replies=3, worlds=2, dice=1, iterations=50)
    env = Env(pool, envs=4, seed=10, open_sheets=1.0, threads=2, max_turns=30)
    report = duel(env, new, old, 4, config, "cpu", seed=1, opponent_search=True)
    assert report["games"] == 4 and report["searched_decisions"] > 0
    with pytest.raises(ValueError):
        duel(Env(pool, envs=2, seed=11, threads=2), new, "greedy", 2, config, "cpu", opponent_search=True)


def test_the_command_line_trains_and_writes_a_summary(data_dir, tmp_path):
    out = tmp_path / "run"
    args = ["--data", str(data_dir), "--init", str(data_dir / "learner.pt"), "--out", str(out),
            "--lr", "1e-3", "--epochs", "1", "--batch", "128", "--holdout", "0.2", "--device", "cpu", "--eval-every", "5"]
    main(args)
    summary = json.loads((out / "summary.json").read_text())
    config = json.loads((out / "config.json").read_text())
    assert summary["steps"] > 0 and summary["final"]["policy_ce"] is not None and config["init"]
    saved = torch.load(out / "checkpoint.pt", weights_only=False)
    assert "contract" in saved and saved["expert"]["step"] == summary["steps"]
    with pytest.raises(SystemExit):
        main(args)  # the directory is no longer empty
    with pytest.raises(SystemExit):
        main(["--data", str(data_dir), "--out", str(tmp_path / "x")])  # neither --init nor --model-size
