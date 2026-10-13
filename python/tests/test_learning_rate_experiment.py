import json
import runpy
from pathlib import Path

import pytest
import torch

from pokemon_ml.train import main


def test_lr_arms_restore_same_state_and_change_only_lr(pool, tmp_path):
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "source"
    main(
        [
            "--pool",
            pool,
            "--run",
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
            "8",
            "--epochs",
            "1",
            "--minibatch",
            "4",
            "--eval-games",
            "2",
            "--max-turns",
            "3",
            "--no-lookahead-eval",
            "--width",
            "32",
            "--layers",
            "1",
            "--heads",
            "2",
            "--ff",
            "64",
        ]
    )
    prepare = runpy.run_path(str(root / "scripts/prepare-league-experiment.py"))[
        "prepare"
    ]
    out = tmp_path / "ab"
    before = (source / "checkpoint.pt").read_bytes()
    with pytest.raises(ValueError, match="expected source update"):
        prepare(source, out, 2, "learning-rate", expected_update=6132)
    assert not out.exists()
    prepare(source, out, 2, "learning-rate", expected_update=1)
    manifest = json.loads((out / "experiment.json").read_text())
    assert manifest["target_update"] == 3
    build_args = runpy.run_path(str(root / "scripts/run-learning-rate-arm.py"))[
        "training_args"
    ]
    commands = {arm: build_args(out, arm, "cpu") for arm in manifest["arms"]}

    def values(command):
        return {
            k: command[i + 1]
            for i, k in enumerate(command)
            if k in ("--run", "--lr", "--updates", "--league-selection")
        }

    assert values(commands["control"])["--lr"] == "0.0003"
    assert values(commands["low-lr"])["--lr"] == "0.0001"
    for arm, command in commands.items():
        assert (out / arm / "checkpoint.pt").read_bytes() == before
        assert "--hours" not in command
        main(command)
        saved = torch.load(out / arm / "checkpoint.pt", weights_only=False)
        assert saved["update"] == 3
        assert saved["ppo_config"]["lr"] == manifest["learning_rates"][arm]
        assert all(
            g["lr"] == manifest["learning_rates"][arm]
            for g in saved["optimizer"]["param_groups"]
        )
        assert saved["league"]["selection"] == "legacy"
    assert (source / "checkpoint.pt").read_bytes() == before
