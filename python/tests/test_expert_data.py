"""Games to learn from (`pokemon_ml.expert_data`): the league, and what is recorded."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from pokemon_ml.env import N_ACTIONS, PASS, PREVIEW, Env, part
from pokemon_ml.expert_data import generate, main, parse_opponent
from pokemon_ml.model import Config as ModelConfig, Model
from pokemon_ml.search import Config


def tiny(seed):
    torch.manual_seed(seed)
    return Model(ModelConfig(width=32, layers=1, heads=2, ff=64)).eval()


def test_opponents_are_read_from_their_specs():
    o = parse_opponent("self+search:0.3")
    assert (o.target, o.search, o.weight, o.name) == ("self", True, 0.3, "self+search")
    o = parse_opponent("runs/a/old.pt")
    assert (o.target, o.search, o.weight) == ("runs/a/old.pt", False, 1.0)
    o = parse_opponent("runs/a/old.pt+search:2")
    assert (o.target, o.search, o.weight) == ("runs/a/old.pt", True, 2.0)
    assert parse_opponent("greedy:0.1").target == "greedy"
    for bad in ("greedy+search", "self:0", "self:-1", ":0.5"):
        with pytest.raises(ValueError):
            parse_opponent(bad)


def load_rows(out, totals):
    shards = [np.load(Path(out) / s["file"]) for s in totals["shards"]]
    return {k: np.concatenate([d[k] for d in shards]) for k in shards[0].files}


def test_what_is_recorded_is_what_a_network_can_learn_from(pool, tmp_path):
    model, old = tiny(1), tiny(2)
    league = [parse_opponent(x) for x in ("self+search:0.3", "self:0.2", "old+search:0.2", "old:0.1", "greedy:0.2")]
    config = Config(candidates=4, replies=3, worlds=2, dice=1, iterations=100, explore=2, noise=0.25)
    env = Env(pool, envs=8, seed=5, open_sheets=1.0, threads=2, max_turns=40)
    torch.manual_seed(0)
    totals = generate(env, model, league, {"self": model, "old": old}, 24, config, tmp_path, "cpu", seed=3, shard_size=150)
    rows = load_rows(tmp_path, totals)
    n = len(rows["f"])
    assert totals["games"] == 24 and totals["rows"] == n == sum(s["rows"] for s in totals["shards"])
    assert len(totals["shards"]) > 1 and all(s["rows"] <= 150 for s in totals["shards"])
    assert sum(sum(r.values()) for r in totals["results"].values()) == 24
    assert totals["searched_rows"] == int(rows["searched"].sum()) > 0
    # Every row is a decision with a choice, of one of the 24 games, which ended one way for all its rows.
    assert (part(rows["i"], "i", "info")[:, 3] == 1).all()
    assert len(np.unique(rows["game"])) == 24 and np.isin(rows["outcome"], [-1, 0, 1]).all()
    for g in np.unique(rows["game"]):
        mine = rows["game"] == g
        assert len(np.unique(rows["outcome"][mine])) == 1 and len(np.unique(rows["opponent"][mine])) == 1
    assert np.isin(rows["opponent"], range(len(league))).all()
    # Searched rows: a mixed strategy over legal pairs, as many as the search had; the rest, none.
    s = rows["searched"]
    assert (part(rows["i"][s], "i", "info")[:, 0] != PREVIEW).all()
    assert np.allclose(rows["target"][s].sum(1), 1, atol=1e-6) and (rows["target"] >= 0).all()
    assert np.isfinite(rows["value"][s]).all() and np.isnan(rows["value"][~s]).all()
    assert (rows["candidates"][~s] == -1).all() and (rows["target"][~s] == 0).all()
    for f_mask, cands, target, explored in zip(rows["mask"][s], rows["candidates"][s], rows["target"][s], rows["explored"][s]):
        mask0 = f_mask[:N_ACTIONS].astype(bool)
        mask1 = f_mask[N_ACTIONS:].reshape(N_ACTIONS, N_ACTIONS).astype(bool)
        real = cands[:, 0] >= 0
        assert (target[~real] == 0).all() and not explored[~real].any()
        assert explored.sum() <= config.explore and real.sum() <= config.candidates + config.explore
        pairs = [tuple(p) for p in cands[real]]
        assert len(set(pairs)) == len(pairs)
        for a0, a1 in pairs:
            assert mask0[a0] and (mask1[a0, a1] or (not mask1[a0].any() and a1 == PASS))


def test_the_same_seed_gives_the_same_data(pool, tmp_path):
    model = tiny(3)
    league = [parse_opponent(x) for x in ("self+search:0.5", "greedy:0.5")]
    config = Config(candidates=3, replies=3, worlds=2, dice=1, iterations=50, explore=1, noise=0.25)
    runs = []
    for k in range(2):
        torch.manual_seed(7)
        env = Env(pool, envs=4, seed=6, open_sheets=1.0, threads=2, max_turns=30)
        totals = generate(env, model, league, {"self": model}, 6, config, tmp_path / str(k), "cpu", seed=4)
        runs.append(load_rows(tmp_path / str(k), totals))
    for key in runs[0]:
        assert np.array_equal(runs[0][key], runs[1][key], equal_nan=True), key


def test_the_command_line_writes_a_manifest(pool, tmp_path):
    for name, seed in (("learner.pt", 4), ("old.pt", 5)):
        m = tiny(seed)
        torch.save(dict(model_config=asdict(m.config), model=m.state_dict()), tmp_path / name)
    out = tmp_path / "data"
    args = [
        "--checkpoint", str(tmp_path / "learner.pt"), "--pool", pool, "--out", str(out), "--games", "4",
        "--envs", "4", "--threads", "2", "--device", "cpu", "--max-turns", "30",
        "--candidates", "3", "--replies", "3", "--worlds", "2", "--dice", "1", "--iterations", "50",
        "--opponent", "self+search:0.5", "--opponent", f"{tmp_path / 'old.pt'}:0.3", "--opponent", "random:0.2",
    ]
    main(args)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["games"] == 4 and manifest["rows"] > 0 and manifest["vary"] is True
    assert manifest["search"]["explore"] == 2 and manifest["search"]["noise"] == 0.25 and manifest["search"]["anchor"] == 0.03
    assert [o["name"] for o in manifest["league"]] == ["self+search", str(tmp_path / "old.pt"), "random"]
    assert str(tmp_path / "old.pt") in manifest["opponent_sha256"]
    with pytest.raises(SystemExit):
        main(args)  # the directory is no longer empty


def test_outcomes_are_the_learner_s(pool, tmp_path):
    # An untrained network, even with search, loses most games to the look-ahead player,
    # which sees everything: recorded from the wrong side, it would seem to win them.
    model = tiny(6)
    league = [parse_opponent("lookahead")]
    config = Config(candidates=3, replies=3, worlds=2, dice=1, iterations=50)
    env = Env(pool, envs=8, seed=7, open_sheets=1.0, threads=2, max_turns=60)
    torch.manual_seed(1)
    totals = generate(env, model, league, {"self": model}, 32, config, tmp_path, "cpu", seed=5)
    rows = load_rows(tmp_path, totals)
    per_game = {g: o for g, o in zip(rows["game"], rows["outcome"])}
    score = (np.array(list(per_game.values())) + 1).mean() / 2
    assert len(per_game) == 32 and score < 0.35, score
    r = totals["results"]["lookahead"]
    assert (r["wins"] + 0.5 * r["ties"]) / 32 == pytest.approx(score)
