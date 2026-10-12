#!/usr/bin/env python3
"""Run 100,000 fresh battles focused on historical observation bug families."""
import argparse
from pathlib import Path
import secrets
import subprocess
import sys

from showdown_stress_suite import load_suite

REPO = Path(__file__).resolve().parents[1]
SUITE = Path(__file__).with_name('showdown-observation-stress.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, help='base seed; omitted selects and prints a random one')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', type=Path, help='new results directory; defaults to ~/showdown-runs/observation-stress-SEED')
    parser.add_argument('--plan-only', action='store_true', help='validate and print coverage; do not build or run battles')
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else secrets.randbelow(2**32 - 1) + 1
    if not 0 < seed < 2**32 or args.workers < 1:
        parser.error('seed must be 1..4294967295 and workers must be positive')
    try:
        themes, rates = load_suite(SUITE, REPO / 'oracle/pool.json')
    except (OSError, ValueError) as error:
        parser.error(str(error))
    out = args.out or Path.home() / 'showdown-runs' / f'observation-stress-{seed}'
    print(f'100,000 fresh targeted battles; base seed {seed}; results {out.resolve()}', flush=True)
    print(f'{len(themes)} themes; 400 batches of 250; alternating closed/open sheets; Mega rates {rates}', flush=True)
    if args.plan_only:
        for name, _ in themes:
            print(name)
        return 0
    return subprocess.call([
        sys.executable, str(Path(__file__).with_name('compare-showdown.py')),
        '--battles', '100000', '--batch-size', '250', '--profile', 'regressions',
        '--themes-file', str(SUITE), '--seed', str(seed), '--workers', str(args.workers), '--out', str(out),
    ])


if __name__ == '__main__':
    sys.exit(main())
