import numpy as np
import torch

from pokemon_ml import ppo
from pokemon_ml.env import Env, part
from pokemon_ml.evaluation import play
from pokemon_ml.model import Config, Model
from pokemon_ml.policy_audit import decision_details


def test_trace_details_are_legal_and_do_not_change_sampling(pool):
    torch.set_num_threads(1)
    model = Model(Config(width=32, layers=1, heads=2, ff=64)).eval()
    env = Env(pool, envs=2, seed=71, threads=1)
    for _ in range(3):
        f, i, mask = ppo.to_device(env, "cpu")
        state = torch.get_rng_state().clone()
        actions, logp, _, _, _ = model.act(f, i, mask)
        after = torch.get_rng_state().clone()
        details = decision_details(model, f[:1], i[:1], mask[:1], actions[:1])
        assert torch.equal(after, torch.get_rng_state())
        recovered = np.unpackbits(
            np.frombuffer(bytes.fromhex(details["mask_packed_hex"]), dtype=np.uint8),
            bitorder="little",
        )[: details["mask_bits"]]
        assert np.array_equal(recovered, env.mask[0])
        assert np.isclose(details["joint_probability"], float(logp[0].exp()), atol=1e-5)
        if "slot0" in details:
            assert all(mask[0, x["action"]] for x in details["slot0"]["top"])
        env.step(actions.numpy())
    torch.set_rng_state(state)
    a = play(
        Env(pool, envs=2, seed=72, max_turns=4, threads=1),
        model,
        "greedy",
        4,
        "cpu",
        trace_games=2,
    )
    torch.set_rng_state(state)
    b = play(
        Env(pool, envs=2, seed=72, max_turns=4, threads=1),
        model,
        "greedy",
        4,
        "cpu",
        trace_games=2,
        trace_policy_details=True,
    )
    assert (a["wins"], a["losses"], a["ties"]) == (b["wins"], b["losses"], b["ties"])
    assert b["trace_schema_version"] == 2
    assert all("policy" in d for game in b["traces"] for d in game["decisions"])


def test_benched_move_identity_pp_binding_is_lost_by_current_encoder(pool):
    torch.set_num_threads(1)
    env = Env(pool, envs=1, seed=73, threads=1)
    env.step(env.baseline("random"))
    f, ids, mask = ppo.to_device(env, "cpu")
    f, ids = f[:1].clone(), ids[:1].clone()
    active = set(int(k) - 1 for k in part(ids, "i", "act_ids")[0, :2, 0])
    bench = next(k for k in range(6) if k not in active)
    mon = part(ids, "i", "mon_ids")
    moves = mon[0, bench, 4:8]
    assert moves[0] != moves[1]
    part(f, "f", "mon_feats")[0, bench, 42:46] = torch.tensor([0.25, 0.75, 1.0, 1.0])
    swapped = ids.clone()
    part(swapped, "i", "mon_ids")[0, bench, 4:6] = moves[:2].flip(0)
    # These two observations bind different move identities to the same PP slots.
    # The observation retains that distinction; the current encoder pools it away.
    model = Model(Config(width=32, layers=1, heads=2, ff=64)).eval()
    with torch.no_grad():
        left, right = model.encode(f, ids), model.encode(f, swapped)
    assert not torch.equal(ids, swapped)
    assert all(torch.allclose(a, b, atol=2e-6) for a, b in zip(left, right))


def test_preview_head_has_additive_lead_reserve_constraint():
    from pokemon_ml.env import TABLES

    model = Model(Config(width=32, layers=1, heads=2, ff=64)).eval()
    choices = {
        tuple(int(x) for x in row): index for index, row in enumerate(TABLES["preview"])
    }
    logits = model.preview_logits(torch.randn(1, 12, 32))[0]
    # Four compatible lead/reserve combinations obey a forced cancellation.
    a, b, c, d = [
        choices[x] for x in [(0, 1, 3, 4), (0, 2, 3, 5), (0, 1, 3, 5), (0, 2, 3, 4)]
    ]
    assert torch.isclose(
        logits[a] + logits[b] - logits[c] - logits[d], torch.tensor(0.0), atol=1e-6
    )


def test_real_cli_exports_readable_diagnostic_traces(pool, tmp_path):
    import json
    import runpy
    import subprocess
    import sys
    from pathlib import Path
    from dataclasses import asdict
    from pokemon_ml import checkpoint

    model = Model(Config(width=32, layers=1, heads=2, ff=64))
    saved = tmp_path / "model.pt"
    torch.save(
        dict(
            model=model.state_dict(),
            model_config=asdict(model.config),
            contract=checkpoint.contract(),
        ),
        saved,
    )
    report_path = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pokemon_ml.evaluation",
            "--checkpoint",
            str(saved),
            "--pool",
            pool,
            "--out",
            str(report_path),
            "--opponent",
            "greedy",
            "--games",
            "2",
            "--envs",
            "2",
            "--threads",
            "1",
            "--max-turns",
            "3",
            "--device",
            "cpu",
            "--trace-games",
            "2",
            "--trace-policy-details",
        ],
        check=True,
    )
    root = Path(__file__).resolve().parents[2]
    render = runpy.run_path(str(root / "scripts/summarize-policy-traces.py"))["render"]
    report = json.loads(report_path.read_text())
    text = render(report)
    assert "chosen" not in text or "chosen slot-0" in text
    assert "probability" in text and "Public actives:" in text
    assert len(report["traces"]) == 2
