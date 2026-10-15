import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

from pokemon_ml import league
from pokemon_ml.schedules import learning_rate
from pokemon_ml.train import main

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = runpy.run_path(str(ROOT / 'scripts/plateau-experiment.py'))


def test_targeted_prefers_competitive_opponents_and_preserves_episodes(monkeypatch):
    monkeypatch.setattr(league, 'snapshots', lambda _: ['000000', '000001', '000002', '000003'])
    player = league.TrainingLeague('unused', 4000, 'cpu', weights=(0, 1, 0), selection='targeted')
    player.results = {'snapshot:' + n: dict(games=10000, wins=w, losses=10000-w, ties=0)
                      for n, w in [('000000', 10000), ('000001', 5000), ('000002', 0), ('000003', 5000)]}
    player.assign(np.ones(4000, dtype=bool))
    counts = {n: player.assigned.count('snapshot:'+n) for n in ['000000','000002','000003']}
    assert counts['000003'] > 5 * counts['000000']
    assert counts['000000'] > 100 and counts['000002'] > 100
    old = player.assigned.copy()
    player.assign(np.zeros(4000, dtype=bool))
    assert old == player.assigned
    restored = league.TrainingLeague('unused', 4000, 'cpu', weights=(0,1,0), state=player.state_dict())
    assert restored.selection == 'targeted' and restored.results == player.results


def test_staged_continuations_and_decay_resume(pool, tmp_path):
    torch.set_num_threads(1)
    source = tmp_path / 'source'
    args = ['--pool', pool, '--device', 'cpu', '--envs', '2', '--threads', '1',
            '--steps', '4', '--epochs', '1', '--minibatch', '4', '--eval-games', '2',
            '--max-turns', '3', '--no-lookahead-eval', '--width', '32', '--layers', '1',
            '--heads', '2', '--ff', '64', '--snapshot-every', '1', '--lr', '0.00003']
    main(args + ['--run', str(source), '--updates', '1'])
    before = (source/'checkpoint.pt').read_bytes()
    for stage in ['diagnose', 'compare']:
        out = tmp_path/stage
        WORKFLOW['prepare'](source, out, 2, 1, stage)
        manifest = json.loads((out/'experiment.json').read_text())
        assert manifest['arms'] == (['control'] if stage == 'diagnose' else ['control','decay','targeted'])
        for arm in manifest['arms']:
            assert (out/arm/'checkpoint.pt').read_bytes() == before
            if arm == 'decay':
                helper = runpy.run_path(str(ROOT / 'scripts/run-learning-rate-arm.py'))['training_args']
                partial = helper(out, arm, 'cpu')
                partial[partial.index('--updates') + 1] = '2'
                partial += ['--lr-decay-end', '0.00001', '--lr-decay-start-update', '1', '--lr-decay-end-update', '3']
                main(partial)
                midway = torch.load(out/arm/'checkpoint.pt', weights_only=False)
                assert midway['optimizer']['param_groups'][0]['lr'] == pytest.approx((3e-5*1e-5)**0.5)
            WORKFLOW['run'](out, arm, 'cpu')
            state = torch.load(out/arm/'checkpoint.pt', weights_only=False)
            assert state['update'] == 3
            assert state['league']['selection'] == ('targeted' if arm == 'targeted' else 'legacy')
            assert state['optimizer']['param_groups'][0]['lr'] == pytest.approx(1e-5 if arm == 'decay' else 3e-5)
        if stage == 'diagnose':
            # Synthetic same-pool fixture exercises the real evaluation CLI;
            # production requires the source's actual held-out pool.
            manifest['source_args']['eval_pool'] = pool
            (out/'experiment.json').write_text(json.dumps(manifest))
            WORKFLOW['evaluate'](out, 'control', 2, 'cpu')
            reports = list((out/'evaluation/control').glob('*.json'))
            assert len(reports) == 6
            assert all(json.loads(p.read_text())['games'] == 2 for p in reports)
        if stage == 'compare':
            WORKFLOW['run'](out, 'decay', 'cpu')
            state = torch.load(out/'decay/checkpoint.pt', weights_only=False)
            assert state['lr_schedule']['start_update'] == 1
            assert state['optimizer']['param_groups'][0]['lr'] == pytest.approx(1e-5)
    assert (source/'checkpoint.pt').read_bytes() == before


def test_bounded_schedule():
    s = dict(start_lr=3e-5, end_lr=1e-5, start_update=100, end_update=1100)
    assert learning_rate(s, 0) == 3e-5
    assert learning_rate(s, 600) == pytest.approx((3e-5 * 1e-5)**0.5)
    assert learning_rate(s, 2000) == pytest.approx(1e-5)
    with pytest.raises(ValueError):
        learning_rate(dict(s, end_lr=float('nan')), 100)
