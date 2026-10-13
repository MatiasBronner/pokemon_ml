import json
import runpy
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from pokemon_ml import checkpoint
from pokemon_ml.model import Config, Model


def test_experiment_driver_calls_real_evaluator_with_turn_limit(
    pool, tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    model = Model(Config(width=32, layers=1, heads=2, ff=64))
    saved = dict(
        model=model.state_dict(),
        model_config=asdict(model.config),
        contract=checkpoint.contract(),
    )
    torch.save(saved, tmp_path / "source-checkpoint.pt")
    snapshots = tmp_path / "legacy" / "snapshots"
    snapshots.mkdir(parents=True)
    torch.save(saved, snapshots / "000000.pt")
    (tmp_path / "experiment.json").write_text(
        json.dumps(
            dict(
                panel=["000000"],
                source_args=dict(pool=pool, threads=1, open_sheets=0.5, max_turns=3),
            )
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate-league-experiment.py",
            "--experiment",
            str(tmp_path),
            "--arm",
            "source",
            "--games",
            "2",
        ],
    )
    actual_run = subprocess.run

    def run_cpu(command, **kwargs):
        command = list(command)
        command[command.index("--device") + 1] = "cpu"
        command[command.index("--envs") + 1] = "2"
        return actual_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run_cpu)
    runpy.run_path(
        str(root / "scripts/evaluate-league-experiment.py"), run_name="__main__"
    )
    for rival in ("000000", "greedy"):
        report = json.loads(
            (tmp_path / "evaluation" / "source" / f"{rival}.json").read_text()
        )
        assert report["games"] == 2
        assert report["max_turns"] == 3
