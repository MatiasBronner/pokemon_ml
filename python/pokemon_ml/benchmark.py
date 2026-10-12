"""Measure environment, inference and full PPO throughput on the allocated hardware."""

import argparse
import itertools
import json
import time
from pathlib import Path
import torch
from .env import Env
from .model import Model, preset
from . import ppo


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--sizes", nargs="+", choices=["small", "medium", "large"], default=["small"]
    )
    p.add_argument("--envs", nargs="+", type=int, default=[256, 512, 1024])
    p.add_argument("--threads", nargs="+", type=int, default=[2, 4])
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)
    if min(a.envs + a.threads + [a.steps]) < 1:
        p.error("counts must be positive")
    from .train import pick_device

    device = pick_device(a.device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    reports = []
    for size, envs, threads in itertools.product(a.sizes, a.envs, a.threads):
        torch.manual_seed(a.seed)
        env = Env(a.pool, envs, a.seed, threads=threads)
        model = Model(preset(size)).to(device).eval()
        cfg = ppo.Config(steps=a.steps)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        roll = ppo.Rollout(a.steps, 2 * envs, device)
        # Warm up inference separately; cold-start cost is not steady throughput.
        f, i, m = ppo.to_device(env, device)
        for _ in range(3):
            model.act(f, i, m)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync()
        start = time.perf_counter()
        for _ in range(a.steps):
            model.act(f, i, m)
        sync()
        inference = time.perf_counter() - start
        actions = env.baseline("random")
        start = time.perf_counter()
        for _ in range(a.steps):
            env.step(actions)
            actions = env.baseline("random")
        environment = time.perf_counter() - start
        sync()
        start = time.perf_counter()
        ppo.collect(env, model, roll)
        sync()
        collected = time.perf_counter() - start
        log = ppo.update(model, opt, roll, cfg)
        sync()
        elapsed = time.perf_counter() - start
        reports.append(
            dict(
                model_size=size,
                parameters=sum(p.numel() for p in model.parameters()),
                envs=envs,
                threads=threads,
                inference_observations_per_second=2 * envs * a.steps / inference,
                environment_game_steps_per_second=envs * a.steps / environment,
                environment_includes_random_action_selection=True,
                rollout_seconds=collected,
                update_seconds=elapsed - collected,
                learner_decisions_per_second=log["decisions"] / elapsed,
                full_game_steps_per_second=envs * a.steps / elapsed,
                cuda_peak_bytes=torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                **roll.timings,
            )
        )
        print(json.dumps(reports[-1]), flush=True)
        del model, opt, roll
        if device.type == "cuda":
            torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps(
            dict(
                device=str(device),
                torch_version=torch.__version__,
                seed=a.seed,
                device_name=torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else str(device),
                experiments=reports,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
