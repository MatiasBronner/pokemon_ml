#!/usr/bin/env bash
# Checks the engine's team validator against Pokémon Showdown's, both ways round.
#
#   scripts/check-teams.sh [teams] [seed] [format id]
#
# 1. Showdown judges random teams, most of them legal or one step from legal
#    (a move the Pokémon cannot learn, an item held twice, a stat point too
#    many); the engine must give every one the same verdict.
# 2. The engine makes random legal teams; Showdown must accept every one.
#
# The format defaults to the one the engine is built for. For another, generate
# its file first: node oracle/gen_format.js <format id>
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-5000}"
SEED="${2:-$RANDOM}"
FORMAT="${3:-}"
DIR="$(mktemp -d)"
trap 'rm -rf "$DIR"' EXIT

if [ ! -f oracle/pokemon-showdown/dist/sim/index.js ]; then
  echo "Showdown is not built yet; run scripts/setup-oracle.sh first." >&2
  exit 1
fi
JS_FORMAT=()
RS_FORMAT=()
if [ -n "$FORMAT" ]; then
  [ -f "formats/$FORMAT.json" ] || node oracle/gen_format.js "$FORMAT"
  JS_FORMAT=(--format "$FORMAT")
  RS_FORMAT=(--format "formats/$FORMAT.json")
fi

echo "seed $SEED"
cargo build --quiet --release --bin teamcheck
node oracle/gen_teams.js --n "$N" --seed "$SEED" --out "$DIR/judged.jsonl" "${JS_FORMAT[@]}" | head -n 1
./target/release/teamcheck "$DIR/judged.jsonl" "${RS_FORMAT[@]}"
./target/release/teamcheck --sample "$N" --seed "$((SEED % 65536))" "${RS_FORMAT[@]}" > "$DIR/sampled.jsonl"
node oracle/gen_teams.js --judge "$DIR/sampled.jsonl" "${JS_FORMAT[@]}"
