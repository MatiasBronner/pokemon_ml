#!/usr/bin/env bash
# Fetches and builds the pinned Pokémon Showdown commit the engine is checked against.
# Needs git and Node.js 22 or newer.
set -euo pipefail
cd "$(dirname "$0")/../oracle"

COMMIT=ad7ca5d51c03865e6829dbec3a7875df71480fb8   # 2026-10-08, has [Gen 9 Champions] VGC 2026 Reg M-C

if [ ! -d pokemon-showdown/.git ]; then
  mkdir -p pokemon-showdown
  git -C pokemon-showdown init -q
  git -C pokemon-showdown remote add origin https://github.com/smogon/pokemon-showdown.git
fi
git -C pokemon-showdown fetch -q --depth 1 origin "$COMMIT"
git -C pokemon-showdown checkout -q FETCH_HEAD
(cd pokemon-showdown && npm install --no-audit --no-fund && node build)
echo "Showdown $COMMIT built in oracle/pokemon-showdown"
