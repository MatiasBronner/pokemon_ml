#!/usr/bin/env bash
# Plays random battles in Showdown and checks the engine reproduces every one.
#
#   scripts/fuzz.sh [battles] [seed]            compare state after every decision
#   TRACE=1 scripts/fuzz.sh [battles] [seed]    also compare every RNG draw (slower, large files)
#   REBUILD=1 scripts/fuzz.sh [battles] [seed]  also rebuild the battle from its exported position at every decision
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-2000}"
SEED="${2:-$RANDOM}"
OUT="$(mktemp -d)/cases.jsonl"
trap 'rm -rf "$(dirname "$OUT")"' EXIT

if [ ! -f oracle/pokemon-showdown/dist/sim/index.js ]; then
  echo "Showdown is not built yet; run scripts/setup-oracle.sh first." >&2
  exit 1
fi

echo "seed $SEED"
if [ -n "${TRACE:-}" ]; then
  node oracle/gen_cases.js --n "$N" --seed "$SEED" --out "$OUT" --trace
  cargo run --quiet --release --features trace --bin difftest -- "$OUT"
else
  node oracle/gen_cases.js --n "$N" --seed "$SEED" --out "$OUT" --stats oracle/last_fuzz_stats.json
  cargo run --quiet --release --bin difftest -- "$OUT" ${REBUILD:+--by-hand}
fi
