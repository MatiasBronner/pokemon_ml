"""Isolated learner execution experiments; never overwrite a training checkpoint."""

import argparse
import json
import os
import statistics
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
import torch
from . import ppo, checkpoint
from .env import Env, part, PREVIEW
from .model import Model, Config, preset
from .runtime import Runtime, configure
from .league import TrainingLeague


def hardware(device):
    data = dict(
        device=str(device),
        torch=torch.__version__,
        cuda_build=torch.version.cuda,
        python=sys.version,
        host=os.uname().nodename,
        slurm_job=os.environ.get("SLURM_JOB_ID"),
    )
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        data.update(
            gpu=prop.name,
            gpu_uuid=str(getattr(prop, "uuid", "unavailable")),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            capability=[prop.major, prop.minor],
            memory_bytes=prop.total_memory,
            bf16_native=prop.major >= 8,
        )
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        data["nvidia_smi"] = r.stdout.strip()
    return data


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def record_profile(out, name, operation, device):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as prof:
        operation()
        sync(device)
    prof.export_chrome_trace(str(out / (name + "-trace.json")))
    events = prof.key_averages()
    try:
        table = events.table(sort_by="self_device_time_total", row_limit=30)
    except AttributeError:
        table = events.table(sort_by="self_cpu_time_total", row_limit=30)
    (out / (name + "-operators.txt")).write_text(table)
    return [
        e.key
        for e in events
        if any(s in e.key.lower() for s in ("attention", "scaled_dot", "flash"))
    ]


def validation(model, env, reference):
    f, i, mask = ppo.to_device(env, next(model.parameters()).device)
    model.eval()
    reference.eval()
    with torch.no_grad():
        actions, lp, val, m0, m1 = model.act(f, i, mask)
    again, entropy, spread, value = model.evaluate(f, i, m0, m1, actions)
    legal = part(i, "i", "info")[:, 0] != PREVIEW
    rows = torch.arange(len(i), device=i.device)[legal]
    if not (m0[rows, actions[rows, 0]].all() and m1[rows, actions[rows, 1]].all()):
        raise ValueError("illegal action selected")
    if not all(
        torch.isfinite(x).all() for x in (lp, val, again, entropy, spread, value)
    ):
        raise ValueError("nonfinite policy/value output")
    err = float((lp - again).detach().abs().max())
    value_err = float((val - value).detach().abs().max())
    if err > 1e-4 or value_err > 1e-4:
        raise ValueError("collection/update likelihood mismatch")
    baseline_lp = reference.evaluate(f, i, m0, m1, actions)[0]
    delta = float((baseline_lp - lp).detach().abs().max())
    tolerance = (
        0.1 if getattr(model, "runtime", Runtime()).precision == "bf16" else 1e-3
    )
    if delta > tolerance:
        raise ValueError(f"FP32 reference log-probability drift {delta} > {tolerance}")
    return dict(
        likelihood_error=err,
        value_consistency_error=value_err,
        fp32_log_probability_max_difference=delta,
        log_probability_tolerance=tolerance,
    )


def worker(a):
    device = torch.device(a.device)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    out = Path(a.worker_out)
    out.mkdir(parents=True, exist_ok=True)
    options = Runtime(**json.loads(a.runtime_json))
    if a.checkpoint:
        saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        checkpoint.verify_contract(saved)
        config = Config(**saved["model_config"])
        cfg = ppo.Config(**saved["ppo_config"])
    else:
        saved = None
        config = preset(a.model_size)
        cfg = ppo.Config()
    cfg = replace(cfg, steps=a.steps, minibatch=a.minibatch)
    model = Model(config).to(device)
    reference = Model(config).to(device).eval()
    if saved:
        model.load_state_dict(saved["model"])
    reference.load_state_dict(model.state_dict())
    model.eval()
    env = Env(a.pool, a.envs, a.seed, threads=a.threads)
    roll = ppo.Rollout(
        a.steps,
        2 * a.envs,
        device,
        config.history,
        config.opponent_prediction or config.hidden_prediction,
    )
    opponents = (
        TrainingLeague(a.league_run, a.envs, device, a.seed) if a.league_run else None
    )
    if opponents and a.league_manifest:
        opponents.allowed = set(json.loads(Path(a.league_manifest).read_text()))
    # Validate on preview and a battle position before updating any parameters.
    configure(model, options)
    start = time.perf_counter()
    checks = [validation(model, env, reference)]
    env.step(env.baseline("greedy"))
    checks.append(validation(model, env, reference))
    sync(device)
    startup = time.perf_counter() - start
    del reference
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        eps=1e-5,
        fused=options.fused_adam if device.type == "cuda" else None,
    )
    if saved:
        for group in saved["optimizer"]["param_groups"]:
            group["fused"] = options.fused_adam if device.type == "cuda" else None
        opt.load_state_dict(saved["optimizer"])
        # Saved optimizer parameter groups can override execution settings.
        for group in opt.param_groups:
            group["fused"] = options.fused_adam if device.type == "cuda" else None
    # Warm both forward and backward shapes; exclude cold compile/optimizer allocation.
    start = time.perf_counter()
    ppo.collect(env, model, roll, opponents)
    ppo.update(model, opt, roll, cfg)
    sync(device)
    warmup = time.perf_counter() - start
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    records = []
    for _ in range(a.repeats):
        sync(device)
        start = time.perf_counter()
        ppo.collect(env, model, roll, opponents)
        sync(device)
        collected = time.perf_counter() - start
        log = ppo.update(model, opt, roll, cfg)
        sync(device)
        elapsed = time.perf_counter() - start
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("nonfinite parameters after PPO")
        numeric = [v for v in log.values() if isinstance(v, (int, float))]
        if not all(__import__("math").isfinite(v) for v in numeric):
            raise ValueError("nonfinite PPO metrics")
        records.append(
            dict(
                collection_seconds=collected,
                update_seconds=elapsed - collected,
                elapsed_seconds=elapsed,
                decisions=log["decisions"],
                decisions_per_second=log["decisions"] / elapsed,
                metrics=log,
            )
        )
    measured_peak = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    attention = []
    if a.profile:
        profile_roll = ppo.Rollout(
            min(a.steps, 4),
            2 * a.envs,
            device,
            config.history,
            config.opponent_prediction or config.hidden_prediction,
        )
        profile_cfg = replace(cfg, steps=profile_roll.steps)
        attention = record_profile(
            out,
            "collection",
            lambda: ppo.collect(env, model, profile_roll, opponents),
            device,
        )
        attention += record_profile(
            out,
            "ppo",
            lambda: ppo.update(model, opt, profile_roll, profile_cfg),
            device,
        )
    report = dict(
        status="passed",
        hardware=hardware(device),
        runtime=asdict(options),
        minibatch=a.minibatch,
        envs=a.envs,
        steps=a.steps,
        epochs=cfg.epochs,
        model=asdict(config),
        checks=checks,
        cold_validation_seconds=startup,
        warmup_update_seconds=warmup,
        median_decisions_per_second=statistics.median(
            r["decisions_per_second"] for r in records
        ),
        median_update_seconds=statistics.median(r["update_seconds"] for r in records),
        peak_memory_bytes=measured_peak,
        repeats=records,
        attention_ops=sorted(set(attention)),
        league=bool(opponents),
        weights_updated_only_in_memory=True,
    )
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def candidates(device, batches, compilation):
    fp32 = Runtime()
    optimized = Runtime(deferred_metrics=True)
    choices = [("baseline", batches[0], fp32)]
    for batch in batches:
        choices.append(("fp32-deferred", batch, optimized))
    if device.type == "cuda":
        choices.append(("fp32-sdpa", batches[0], replace(optimized, attention="sdpa")))
        choices.append(
            ("fp32-fused-adam", batches[0], replace(optimized, fused_adam=True))
        )
        if torch.cuda.get_device_capability(device)[0] >= 8:
            for batch in batches:
                choices.append(
                    (
                        "bf16",
                        batch,
                        replace(optimized, precision="bf16", attention="sdpa"),
                    )
                )
            choices.append(
                (
                    "bf16-flash",
                    batches[0],
                    replace(optimized, precision="bf16", attention="flash"),
                )
            )
            if compilation:
                choices.append(
                    (
                        "bf16-compile",
                        batches[0],
                        replace(
                            optimized, precision="bf16", attention="sdpa", compile=True
                        ),
                    )
                )
    if compilation:
        choices.append(
            (
                "fp32-compile",
                batches[0],
                replace(optimized, attention="sdpa", compile=True),
            )
        )
    return choices


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--league-run")
    p.add_argument(
        "--model-size", choices=["small", "medium", "large"], default="small"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--minibatches", nargs="+", type=int, default=[4096, 8192, 16384])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--include-compile", action="store_true")
    p.add_argument(
        "--timeout",
        type=int,
        default=480,
        help="seconds per isolated candidate, including compilation",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="profile baseline collection and PPO after timing",
    )
    p.add_argument("--league-manifest", help=argparse.SUPPRESS)
    p.add_argument("--worker-out", help=argparse.SUPPRESS)
    p.add_argument("--runtime-json", help=argparse.SUPPRESS)
    p.add_argument("--minibatch", type=int, default=4096, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if (
        min(a.envs, a.threads, a.steps, a.repeats, a.timeout) < 1
        or min(a.minibatches) < 2
    ):
        p.error("positive counts/minibatches>=2 required")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.worker_out:
        worker(a)
        return
    # Avoid overwriting reports from a previous hardware allocation.
    if (out / "summary.json").exists():
        p.error("output directory already has a report; choose a new --out")
    input_checkpoint = a.checkpoint
    if a.checkpoint:
        frozen = out / "source-checkpoint.pt"
        shutil.copyfile(a.checkpoint, frozen)
        a.checkpoint = str(frozen.resolve())
    if a.league_run:
        from .league import snapshots

        manifest = out / "league-manifest.json"
        manifest.write_text(json.dumps(snapshots(a.league_run)))
        a.league_manifest = str(manifest.resolve())
    device = torch.device(a.device)
    hw = hardware(device)
    provenance = dict(
        pool_sha256=checkpoint.file_hash(a.pool),
        input_checkpoint=input_checkpoint,
        checkpoint_sha256=checkpoint.file_hash(a.checkpoint) if a.checkpoint else None,
        seed=a.seed,
        steps=a.steps,
        envs=a.envs,
        threads=a.threads,
        league_manifest=json.loads(Path(a.league_manifest).read_text())
        if a.league_run
        else None,
    )
    reports = []
    for name, batch, options in candidates(device, a.minibatches, a.include_compile):
        folder = out / f"{name}-{batch}"
        folder.mkdir(exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "pokemon_ml.tune_learner",
            "--pool",
            a.pool,
            "--out",
            str(out),
            "--worker-out",
            str(folder),
            "--runtime-json",
            json.dumps(asdict(options)),
            "--device",
            a.device,
            "--envs",
            str(a.envs),
            "--threads",
            str(a.threads),
            "--steps",
            str(a.steps),
            "--minibatch",
            str(batch),
            "--repeats",
            str(a.repeats),
            "--seed",
            str(a.seed),
            "--model-size",
            a.model_size,
        ]
        if a.checkpoint:
            command += ["--checkpoint", a.checkpoint]
        if a.league_run:
            command += [
                "--league-run",
                a.league_run,
                "--league-manifest",
                a.league_manifest,
            ]
        if a.profile and name == "baseline":
            command += ["--profile"]
        print(f"Testing {name}, minibatch {batch}", flush=True)
        worker_started = time.perf_counter()
        try:
            with (folder / "worker.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, "TORCHINDUCTOR_COMPILE_THREADS": "1"},
                )
                try:
                    process.wait(timeout=a.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
                result = process
            if result.returncode:
                raise RuntimeError(
                    f"worker exit {result.returncode}; see {folder}/worker.log"
                )
            report = json.loads((folder / "result.json").read_text())
            report["name"] = name
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            report = dict(name=name, minibatch=batch, status="failed", error=str(error))
        report["worker_total_seconds"] = time.perf_counter() - worker_started
        reports.append(report)
        # Save partial results after every candidate, including failures/OOM/timeouts.
        (out / "summary.json").write_text(
            json.dumps(
                dict(hardware=hw, provenance=provenance, candidates=reports), indent=2
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "name",
                        "status",
                        "minibatch",
                        "median_decisions_per_second",
                        "error",
                    )
                    if k in report
                }
            ),
            flush=True,
        )
    passed = [r for r in reports if r["status"] == "passed"]
    baseline = next((r for r in passed if r["name"] == "baseline"), None)
    # Execution-only recommendations use the original minibatch size. Batch changes
    # alter optimizer step counts and require separate learning-quality experiments.
    compatible = [r for r in passed if r["minibatch"] == a.minibatches[0]]
    best = (
        max(compatible, key=lambda r: r["median_decisions_per_second"])
        if compatible
        else None
    )
    recommendation = dict(
        hardware=hw,
        best_same_minibatch=best["name"] if best else None,
        runtime=best["runtime"] if best else None,
        minibatch=a.minibatches[0],
        speedup=best["median_decisions_per_second"]
        / baseline["median_decisions_per_second"]
        if best and baseline
        else None,
        note="Short numerical/throughput checks only; confirm learning with matched seeds. No training settings changed.",
    )
    if best and baseline:
        saved_seconds_per_decision = (
            1 / baseline["median_decisions_per_second"]
            - 1 / best["median_decisions_per_second"]
        )
        extra_startup = max(
            0,
            best["cold_validation_seconds"]
            + best["warmup_update_seconds"]
            - baseline["cold_validation_seconds"]
            - baseline["warmup_update_seconds"],
        )
        recommendation["startup_break_even_decisions"] = (
            extra_startup / saved_seconds_per_decision
            if saved_seconds_per_decision > 0
            else None
        )
    if best:
        opt = best["runtime"]
        flags = [
            "--precision",
            opt["precision"],
            "--attention",
            opt["attention"],
            "--compile-encoder" if opt["compile"] else "--no-compile-encoder",
            "--deferred-metrics"
            if opt["deferred_metrics"]
            else "--no-deferred-metrics",
            "--fused-adam" if opt["fused_adam"] else "--no-fused-adam",
        ]
        recommendation["train_flags"] = flags
        (out / "train-flags.txt").write_text(" ".join(flags) + "\n")
    (out / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2) + "\n"
    )
    print(json.dumps(recommendation), flush=True)


if __name__ == "__main__":
    main()
