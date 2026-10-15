#!/usr/bin/env python3
"""Prepare and run staged, resumable plateau experiments on OSCAR."""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
import shutil
import torch

HERE = Path(__file__).resolve().parent


def prepare(source, out, updates, expected_update, stage):
    # Reuse source protection, snapshot preservation and LR/source guards.
    factory = runpy.run_path(str(HERE / 'prepare-league-experiment.py'))['prepare']
    saved = torch.load(Path(source) / 'checkpoint.pt', map_location='cpu', weights_only=False)
    rate = saved['ppo_config']['lr']
    if stage == 'compare':
        if rate != 0.00003:
            raise ValueError('comparison requires source LR 0.00003')
        state = saved.get('league') or {}
        if tuple(state.get('weights', ())) != (0.5, 0.4, 0.1) or state.get('selection', 'legacy') != 'legacy':
            raise ValueError('comparison requires the legacy 50/40/10 source league')
    if saved.get('lr_schedule'):
        raise ValueError('prepare from a constant-LR checkpoint; do not silently replace an existing schedule')
    factory(source, out, updates, 'learning-rate', expected_update, rate, rate / 3)
    root = Path(out)
    manifest = json.loads((root / 'experiment.json').read_text())
    if stage == 'diagnose':
        shutil.rmtree(root / 'low-lr')
        arms = ['control']
    else:
        (root / 'low-lr').rename(root / 'decay')
        def copy_file(src, dst):
            if Path(src).parent.name == 'snapshots':
                try:
                    import os
                    os.link(src, dst)
                    return dst
                except OSError:
                    pass
            return shutil.copy2(src, dst)
        shutil.copytree(root / 'control', root / 'targeted', copy_function=copy_file)
        arms = ['control', 'decay', 'targeted']
    manifest.update(stage=stage, arms=arms)
    print(f'Stage {stage}: prepared arms {arms}', flush=True)
    (root / 'experiment.json').write_text(json.dumps(manifest, indent=2) + '\n')


def run(root, arm, device):
    root = Path(root).resolve()
    m = json.loads((root / 'experiment.json').read_text())
    from pokemon_ml.checkpoint import file_hash
    if file_hash(root / 'source-checkpoint.pt') != m['source_sha256']:
        raise ValueError('frozen source checkpoint has changed')
    if arm not in m['arms']:
        raise ValueError('arm not prepared for this stage')
    # Existing helper forwards provenance, rollout, runtime and evaluation options.
    helper = runpy.run_path(str(HERE / 'run-learning-rate-arm.py'))['training_args']
    args = helper(root, arm, device)
    if arm == 'decay':
        args += ['--lr-decay-end', str(m['learning_rates']['low-lr']),
                 '--lr-decay-start-update', str(m['source_update']),
                 '--lr-decay-end-update', str(m['target_update'])]
    if arm == 'targeted':
        args[args.index('--league-selection') + 1] = 'targeted'
    from pokemon_ml.train import main
    main(args)
    saved = torch.load(root / arm / 'checkpoint.pt', map_location='cpu', weights_only=False)
    if saved['update'] != m['target_update']:
        raise SystemExit('Arm incomplete; resubmit the same job before evaluating.')
    source = torch.load(root / 'source-checkpoint.pt', map_location='cpu', weights_only=False)
    print(json.dumps(dict(arm=arm, update=saved['update'],
        learner_decisions=saved['decisions']-source['decisions'],
        training_seconds=saved['seconds']-source['seconds'],
        wall_seconds=saved.get('wall_seconds',0)-source.get('wall_seconds',0))), flush=True)


def evaluate(root, arm, games, device):
    root = Path(root).resolve()
    m = json.loads((root / 'experiment.json').read_text())
    from pokemon_ml.checkpoint import file_hash
    if file_hash(root / 'source-checkpoint.pt') != m['source_sha256']:
        raise ValueError('frozen source checkpoint has changed')
    if arm not in m['arms']:
        raise ValueError('arm not prepared')
    for name in ([arm, 'control'] if arm != 'control' else [arm]):
        state = torch.load(root / name / 'checkpoint.pt', map_location='cpu', weights_only=False)
        if state['update'] != m['target_update']:
            raise ValueError(f'{name} is incomplete; resume it first')
    a = m['source_args']
    if not a.get('eval_pool'):
        raise ValueError('a held-out eval_pool is required for plateau decisions')
    rivals = [('source', root / 'source-checkpoint.pt'), ('greedy', 'greedy')]
    rivals += [(n, root / 'control' / 'snapshots' / (n+'.pt')) for n in m['panel']]
    if arm != 'control':
        rivals.append(('control', root / 'control' / 'checkpoint.pt'))
    out = root / 'evaluation' / arm
    out.mkdir(parents=True, exist_ok=True)
    for seed in (10702347, 10802347):
        for label, opponent in rivals:
            subprocess.run([sys.executable, '-m', 'pokemon_ml.evaluation',
                '--checkpoint', str(root / arm / 'checkpoint.pt'), '--opponent', str(opponent),
                '--pool', a['eval_pool'], '--games', str(games), '--seed', str(seed),
                '--envs', '128', '--threads', str(a['threads']), '--device', device,
                '--open-sheets', str(a['open_sheets']), '--max-turns', str(a['max_turns']),
                '--trace-games', '0', '--out', str(out / f'{label}-{seed}.json')], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'train', 'evaluate'])
    p.add_argument('--experiment', required=True)
    p.add_argument('--source')
    p.add_argument('--expected-update', type=int)
    p.add_argument('--updates', type=int, default=1000)
    p.add_argument('--stage', choices=['diagnose', 'compare'], default='diagnose')
    p.add_argument('--arm', default='control')
    p.add_argument('--device', default='cuda')
    p.add_argument('--games', type=int, default=3000)
    a = p.parse_args()
    if a.action == 'prepare':
        if not a.source or a.expected_update is None:
            p.error('preparation requires --source and --expected-update')
        prepare(a.source, a.experiment, a.updates, a.expected_update, a.stage)
    elif a.action == 'train':
        run(a.experiment, a.arm, a.device)
    else:
        evaluate(a.experiment, a.arm, a.games, a.device)


if __name__ == '__main__':
    main()
