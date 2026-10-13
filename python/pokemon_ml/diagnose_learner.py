"""Read-only GPU diagnostics for acting versus gradient-enabled PPO evaluation."""

import argparse
import json
from pathlib import Path
import torch
from .evaluation import load
from .env import Env
from .runtime import Runtime, configure
from .tune_learner import validation, hardware


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=37)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    model = load(a.checkpoint, device)
    reference = load(a.checkpoint, device)
    env = Env(a.pool, a.envs, a.seed, threads=a.threads)
    reports = []
    # Same observations and RNG seed for each backend; each check evaluates its own sampled actions. No optimizer or updates.
    for phase in ["preview", "battle"]:
        if phase == "battle":
            env.step(env.baseline("greedy"))
        for attention in ["auto", "sdpa", "math"]:
            torch.manual_seed(a.seed)
            configure(model, Runtime(attention=attention))
            result = validation(model, env, reference, strict=False)
            reports.append(dict(phase=phase, attention=attention, **result))
            print(json.dumps(reports[-1]), flush=True)
    from .checkpoint import file_hash

    report = dict(
        hardware=hardware(device),
        checkpoint_sha256=file_hash(a.checkpoint),
        results=reports,
        interpretation="Numerical measurements only; original strict gates unchanged. No parameters updated.",
    )
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
