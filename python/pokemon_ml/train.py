"""Multi-team doubles PPO with episode-stable checkpoint opponents.

python -m pokemon_ml.train --pool teams/train.json --eval-pool teams/eval.json --run runs/small
"""

import argparse
import json
import os
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from . import league, model as model_lib, ppo, checkpoint
from .env import Env
from .team_splits import assert_disjoint
from .runtime import Runtime, configure


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True)
    p.add_argument(
        "--eval-pool",
        help="disjoint species-composition pool; default evaluates training distribution",
    )
    p.add_argument(
        "--archetypes",
        help="JSON composition-key to archetype mapping for held-out evaluation",
    )
    p.add_argument("--run", default="runs/first")
    p.add_argument(
        "--init-checkpoint",
        help="initialize weights in an empty run; reset optimizer, league, RNG and counters",
    )
    p.add_argument(
        "--model-size",
        choices=model_lib.PRESETS,
        default=None,
        help="small by default; resume uses saved sizes",
    )
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--hours", type=float)
    p.add_argument("--lr-decay-end", type=float)
    p.add_argument("--lr-decay-start-update", type=int)
    p.add_argument("--lr-decay-end-update", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", choices=["fp32", "bf16"], default=None)
    p.add_argument(
        "--attention", choices=["auto", "sdpa", "math", "flash"], default=None
    )
    p.add_argument(
        "--compile-encoder", action=argparse.BooleanOptionalAction, default=None
    )
    p.add_argument(
        "--deferred-metrics", action=argparse.BooleanOptionalAction, default=None
    )
    p.add_argument("--fused-adam", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-seed", type=int, default=7000000)
    p.add_argument("--open-sheets", type=float, default=0.5)
    p.add_argument("--no-vary", action="store_true")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-games", type=int, default=300)
    p.add_argument("--rivals", type=int, default=3)
    p.add_argument(
        "--no-lookahead-eval",
        action="store_true",
        help="omit expensive privileged lookahead evaluation during training",
    )
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=25,
        help="checkpoint opponent cadence independent of evaluation",
    )
    p.add_argument("--league-selection", choices=("legacy", "broad", "targeted"), default=None)
    p.add_argument("--league-pool-size", type=int, default=None)
    p.add_argument("--league-cache-size", type=int, default=4)
    p.add_argument("--league-current", type=float, default=0.5)
    p.add_argument("--league-frozen", type=float, default=0.4)
    p.add_argument("--league-scripted", type=float, default=0.1)
    p.add_argument(
        "--league-fixed",
        action="append",
        default=[],
        metavar="PATH:WEIGHT",
        help="a checkpoint from outside the run, always in the league with this weight (repeatable); "
        "all league weights must sum to one",
    )
    p.add_argument("--max-turns", type=int, default=100)
    for name, value in {**asdict(model_lib.Config()), **asdict(ppo.Config())}.items():
        if isinstance(value, bool):
            p.add_argument(
                "--" + name.replace("_", "-"),
                action=argparse.BooleanOptionalAction,
                default=None,
            )
        else:
            p.add_argument(
                "--" + name.replace("_", "-"), type=type(value), default=None
            )
    p.add_argument("--no-speed", dest="use_speed", action="store_false", default=None)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (
        min(
            args.envs,
            args.updates,
            args.eval_every,
            args.eval_games,
            args.save_every,
            args.snapshot_every,
            args.max_turns,
        )
        < 1
    ):
        p.error("counts and cadences must be positive")
    if not 0 <= args.open_sheets <= 1 or (args.hours is not None and args.hours <= 0):
        p.error("open-sheets must be in [0,1] and hours positive")
    device = pick_device(args.device)
    Path(args.run).mkdir(parents=True, exist_ok=True)
    path = Path(args.run) / "checkpoint.pt"
    saved = (
        torch.load(path, map_location="cpu", weights_only=False)
        if path.exists()
        else None
    )
    initial = None
    if args.init_checkpoint:
        if saved or any(Path(args.run).iterdir()):
            p.error("--init-checkpoint requires an empty --run; omit it to resume")
        initial = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        checkpoint.verify_contract(initial)
    architecture_source = saved or initial
    explicit = lambda config: {
        k: getattr(args, k) for k in asdict(config()) if getattr(args, k) is not None
    }
    overrides = explicit(model_lib.Config)
    if args.model_size:
        overrides = {**model_lib.PRESETS[args.model_size], **overrides}
    if architecture_source:
        checkpoint.verify_contract(architecture_source)
        old = asdict(model_lib.Config(**architecture_source["model_config"]))
        if any(old[k] != v for k, v in overrides.items()):
            p.error(
                "model/feature options differ from checkpoint; use a new --run for an ablation"
            )
        net_config = model_lib.Config(**old)
    else:
        net_config = model_lib.preset(
            args.model_size or "small", **explicit(model_lib.Config)
        )
    ppo_config = ppo.Config(
        **{**(saved["ppo_config"] if saved else {}), **explicit(ppo.Config)}
    )
    if (
        min(ppo_config.steps, ppo_config.epochs, ppo_config.minibatch) < 1
        or ppo_config.lr <= 0
    ):
        p.error("PPO counts and learning rate must be positive")
    schedule = saved.get("lr_schedule") if saved else None
    supplied = (args.lr_decay_end, args.lr_decay_start_update, args.lr_decay_end_update)
    if any(x is not None for x in supplied):
        if not all(x is not None for x in supplied):
            p.error("provide all three LR decay options together")
        candidate = dict(start_lr=ppo_config.lr, end_lr=args.lr_decay_end,
                         start_update=args.lr_decay_start_update, end_update=args.lr_decay_end_update)
        if schedule and candidate != schedule:
            p.error("cannot change a saved LR schedule")
        schedule = candidate
    if schedule:
        from .schedules import learning_rate
        learning_rate(schedule, schedule["start_update"])  # validate before training
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = model_lib.Model(net_config).to(device)
    if initial:
        model.load_state_dict(initial["model"], strict=True)
    runtime_values = saved.get("runtime", {}) if saved else {}
    for key, attr in [
        ("precision", "precision"),
        ("attention", "attention"),
        ("compile", "compile_encoder"),
        ("deferred_metrics", "deferred_metrics"),
        ("fused_adam", "fused_adam"),
    ]:
        if getattr(args, attr) is not None:
            runtime_values[key] = getattr(args, attr)
    execution = Runtime(**runtime_values)
    configure(model, execution)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ppo_config.lr,
        eps=1e-5,
        fused=execution.fused_adam if device.type == "cuda" else None,
    )
    if execution.fused_adam and device.type != "cuda":
        p.error("fused Adam experiment requires CUDA")
    totals = dict(update=0, decisions=0, games=0, seconds=0.0, wall_seconds=0.0)
    provenance = dict(
        pool_sha256=checkpoint.file_hash(args.pool),
        eval_pool_sha256=checkpoint.file_hash(args.eval_pool or args.pool),
        open_sheets=args.open_sheets,
        variation=not args.no_vary,
        seed=args.seed,
        max_turns=args.max_turns,
        envs=args.envs,
        eval_seed=args.eval_seed,
    )
    if args.eval_pool:
        assert_disjoint(
            json.loads(Path(args.pool).read_text()),
            json.loads(Path(args.eval_pool).read_text()),
        )
    if saved:
        if saved.get("provenance", provenance) != provenance:
            p.error("pool or rollout settings differ from saved run; start a new --run")
        model.load_state_dict(saved["model"])
        for group in saved["optimizer"]["param_groups"]:
            group["fused"] = execution.fused_adam if device.type == "cuda" else None
        optimizer.load_state_dict(saved["optimizer"])
        totals = {k: saved.get(k, 0) for k in totals}
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(schedule, totals["update"]) if schedule else ppo_config.lr
    env = Env(
        args.pool,
        args.envs,
        args.seed + 1000 * totals["update"],
        args.open_sheets,
        not args.no_vary,
        max_turns=args.max_turns,
        threads=args.threads,
    )
    roll = ppo.Rollout(
        ppo_config.steps,
        2 * args.envs,
        device,
        net_config.history,
        net_config.opponent_prediction or net_config.hidden_prediction,
    )
    weights = (args.league_current, args.league_frozen, args.league_scripted)
    fixed = []
    for spec in args.league_fixed:
        target, _, weight = spec.rpartition(":")
        try:
            fixed.append((target, float(weight)))
        except ValueError:
            p.error(f"--league-fixed takes PATH:WEIGHT, not {spec!r}")
        if not target or not Path(target).is_file():
            p.error(f"--league-fixed checkpoint not found: {target!r}")
        checkpoint.verify_contract(torch.load(target, map_location="cpu", weights_only=False))
    if abs(sum(weights) + sum(w for _, w in fixed) - 1) > 1e-6:
        p.error("league weights (current, frozen, scripted and every --league-fixed) must sum to one")
    training_league = league.TrainingLeague(
        args.run,
        args.envs,
        device,
        args.seed,
        weights,
        state=saved.get("league") if saved else None,
        selection=args.league_selection,
        pool_size=args.league_pool_size,
        cache_size=args.league_cache_size,
        fixed=fixed,
    )
    if saved and "rng" in saved:
        checkpoint.restore_rng(saved["rng"])
    print(
        f"{sum(p.numel() for p in model.parameters()):,} parameters on {device}; Speed={net_config.use_speed}; "
        f"{args.envs} games; league weights={weights}; fixed={fixed}; update={totals['update']}",
        flush=True,
    )
    initialization = saved.get("initialization") if saved else None
    if initial:
        initialization = dict(
            checkpoint=str(Path(args.init_checkpoint).resolve()),
            sha256=checkpoint.file_hash(args.init_checkpoint),
            source_update=initial.get("update"),
            source_games=initial.get("games"),
            source_provenance=initial.get("provenance"),
            reset="optimizer, league, RNG and counters",
        )
        print(
            f"Warm start from source update {initial.get('update')}; new run counters start at zero",
            flush=True,
        )
    configuration = dict(
        model=asdict(net_config),
        runtime=asdict(execution),
        ppo=asdict(ppo_config),
        lr_schedule=schedule,
        provenance=provenance,
        args=vars(args),
        contract=checkpoint.contract(),
        initialization=initialization,
        resume="live episodes restart; optimizer, RNG, league and counters restored",
    )
    (Path(args.run) / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n"
    )

    def save():
        state = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            model_config=asdict(net_config),
            runtime=asdict(execution),
            ppo_config=asdict(ppo_config),
            lr_schedule=schedule,
            contract=checkpoint.contract(),
            rng=checkpoint.rng_state(),
            league=training_league.state_dict(),
            provenance=provenance,
            initialization=initialization,
            **totals,
        )
        torch.save(state, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)

    ratings = league.Ratings(args.run)
    make_rival = lambda seed: Env(
        args.eval_pool or args.pool,
        min(args.envs, 128),
        args.eval_seed + seed,
        args.open_sheets,
        False,
        max_turns=args.max_turns,
        threads=args.threads,
    )
    archetypes = (
        json.loads(Path(args.archetypes).read_text()) if args.archetypes else None
    )

    def measure(log):
        state = checkpoint.rng_state()
        try:
            torch.manual_seed(args.eval_seed)
            name = league.save_snapshot(args.run, totals["update"], model)
            training_league.register_snapshot(name)
            results = league.measure(
                args.run,
                name,
                model,
                make_rival,
                device,
                args.eval_games,
                args.rivals,
                seed=0,
                archetypes=archetypes,
                include_lookahead=not args.no_lookahead_eval,
            )
            ratings.add(results)
            log["rating"] = ratings.ratings[name]
            report = json.loads(
                (Path(args.run) / "evaluation" / f"{name}-vs-greedy.json").read_text()
            )
            log["value_brier"] = report["calibration"]["brier"]
            log["value_ece"] = report["calibration"]["ece"]
            for _, rival, share, _ in results:
                if rival in league.SCRIPTED:
                    log["vs_" + rival] = share
        finally:
            checkpoint.restore_rng(state)

    began = time.perf_counter()
    previous_wall = totals["wall_seconds"]
    if not league.snapshots(args.run):
        measure({})
    stop = {"now": False}
    previous_handlers = {}
    for sig in (
        signal.SIGINT,
        signal.SIGTERM,
        getattr(signal, "SIGUSR1", signal.SIGTERM),
    ):
        previous_handlers[sig] = signal.signal(sig, lambda *_: stop.update(now=True))
    failed = False
    try:
        while totals["update"] < args.updates and not stop["now"]:
            t0 = time.perf_counter()
            before = env.stats()
            ppo.collect(env, model, roll, training_league)
            t1 = time.perf_counter()
            if schedule:
                rate = learning_rate(schedule, totals["update"] + 1)
                for group in optimizer.param_groups:
                    group["lr"] = rate
            log = ppo.update(model, optimizer, roll, ppo_config)
            log["lr"] = optimizer.param_groups[0]["lr"]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t2 = time.perf_counter()
            stats = env.stats()
            totals["update"] += 1
            totals["decisions"] += log["decisions"]
            totals["games"] += stats["games"] - before["games"]
            totals["seconds"] += t2 - t0
            games = stats["games"] - before["games"]
            log.update(
                update=totals["update"],
                games=totals["games"],
                hours=totals["seconds"] / 3600,
                per_second=ppo_config.steps * args.envs / (t2 - t0),
                learner_decisions_per_second=log["decisions"] / (t2 - t0),
                playing=(t1 - t0) / (t2 - t0),
                update_seconds=t2 - t1,
                turns=(stats["turns"] - before["turns"]) / max(games, 1),
                ties=(stats["ties"] - before["ties"]) / max(games, 1),
                timeouts=stats.get("timeouts", 0) - before.get("timeouts", 0),
                league_assignments=training_league.counts.copy(),
                league_results=training_league.drain_results(),
                league_selection=training_league.selection,
            )
            if device.type == "cuda":
                log["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
                    device
                )
                log["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                try:
                    log["gpu_utilization_percent"] = torch.cuda.utilization(device)
                except (AttributeError, ImportError, RuntimeError):
                    log["gpu_utilization_percent"] = None
            if totals["update"] % args.snapshot_every == 0:
                training_league.register_snapshot(
                    league.save_snapshot(args.run, totals["update"], model)
                )
            evaluation_start = time.perf_counter()
            out_of_time = (
                args.hours is not None
                and time.perf_counter() - began >= args.hours * 3600
            )
            if (
                not stop["now"]
                and not out_of_time
                and (
                    totals["update"] % args.eval_every == 0
                    or totals["update"] == args.updates
                )
            ):
                measure(log)
            log["evaluation_seconds"] = time.perf_counter() - evaluation_start
            totals["wall_seconds"] = previous_wall + time.perf_counter() - began
            log["wall_hours"] = totals["wall_seconds"] / 3600
            print(
                f"update {totals['update']} games {totals['games']:,} "
                f"learner decisions/s {log['learner_decisions_per_second']:.0f} KL {log['kl']:.4f}"
                + (f" rating {log['rating']:.0f}" if "rating" in log else ""),
                flush=True,
            )
            with open(Path(args.run) / "log.jsonl", "a") as f:
                f.write(json.dumps(log) + "\n")
            if totals["update"] % args.save_every == 0:
                save()
            if (
                args.hours is not None
                and time.perf_counter() - began >= args.hours * 3600
            ):
                break
    except BaseException:
        failed = True
        raise
    finally:
        if not failed:
            totals["wall_seconds"] = previous_wall + time.perf_counter() - began
            save()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    print(f"saved {path} at update {totals['update']}", flush=True)


if __name__ == "__main__":
    main()
