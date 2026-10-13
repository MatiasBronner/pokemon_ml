from dataclasses import asdict

import pytest
import torch

from pokemon_ml import checkpoint
from pokemon_ml.model import Config, Model
from pokemon_ml.team_mix import combine
from pokemon_ml.team_splits import split
from pokemon_ml.train import main


def test_warm_start_preserves_weights_but_resets_experiment(pool, tmp_path):
    torch.set_num_threads(1)
    model = Model(Config(width=32, layers=1, heads=2, ff=64))
    source = tmp_path / "source.pt"
    torch.save(
        dict(
            model=model.state_dict(),
            model_config=asdict(model.config),
            contract=checkpoint.contract(),
            update=900,
            games=123456,
            provenance={"pool_sha256": "different"},
            optimizer={"invalid": True},
            league={"invalid": True},
        ),
        source,
    )
    before = source.read_bytes()
    run = tmp_path / "new"
    args = [
        "--pool",
        pool,
        "--run",
        str(run),
        "--init-checkpoint",
        str(source),
        "--device",
        "cpu",
        "--envs",
        "2",
        "--threads",
        "1",
        "--updates",
        "1",
        "--steps",
        "2",
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
    main(args)
    start = torch.load(run / "snapshots/000000.pt", weights_only=False)
    assert all(torch.equal(v, start["model"][k]) for k, v in model.state_dict().items())
    saved = torch.load(run / "checkpoint.pt", weights_only=False)
    assert saved["update"] == 1
    assert saved["initialization"]["source_update"] == 900
    assert saved["provenance"]["pool_sha256"] == checkpoint.file_hash(pool)
    assert source.read_bytes() == before
    with pytest.raises(SystemExit):
        main(args)
    resumed = [x for x in args]
    index = resumed.index("--init-checkpoint")
    del resumed[index : index + 2]
    resumed[resumed.index("--updates") + 1] = "2"
    main(resumed)
    latest = torch.load(run / "checkpoint.pt", weights_only=False)
    assert latest["update"] == 2
    assert latest["initialization"] == saved["initialization"]


def test_seen_compositions_stay_out_of_new_evaluation():
    def pool(event, species):
        return dict(
            source=event,
            event=event,
            format="target",
            teams=[dict(team=[dict(species=s)]) for s in species],
        )

    mixed = combine([pool("a", list("ABCDEFGH"))], [pool("w", list("IJKLMNOP"))])
    train, evaluation = split(mixed, seen_pools=[pool("old", list("ABC"))])
    assert not {"A", "B", "C"} & {t["team"][0]["species"] for t in evaluation["teams"]}
    assert {"A", "B", "C"} <= {t["team"][0]["species"] for t in train["teams"]}
    assert sum(t["sampling_weight"] for t in train["teams"]) == pytest.approx(1)
    with pytest.raises(ValueError, match="no unseen"):
        split(mixed, seen_pools=[mixed])
