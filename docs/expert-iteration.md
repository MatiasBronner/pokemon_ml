# Expert iteration

Search makes the network play much better than its own policy: about 71%
against itself without search ([search.md](search.md)). Expert iteration
copies that improvement into the network. Train the policy towards what the
search chose and the value towards how games ended, then search again on
top of the better network.

It runs in rounds:

1. **Generate data** (`pokemon_ml.expert_data`, this page). The current
   network plays with search, against a league, and its decisions are
   recorded.
2. **Train** a network on that data. Not built yet.
3. **Check:** the new network with search against the old one with search.
   If that is clearly above 50%, the round worked, and the new network is
   the next round's learner.

## Generating data

The learner plays every game with search: the default anchored search
(4 dice, λ = 0.03) plus exploration (`--explore 2 --noise 0.25`). On Oscar,
exploration at this anchor cost nothing in strength. It lets pairs the
policy ranks below its top eight be tried, which is how the policy can
learn moves it currently misses.

Its opponent is drawn from a **league** for each game and kept for the
whole game. A network that only ever learns from games against itself
learns how to play itself, which is the "abnormal things in the wild" risk.
Frozen checkpoints and scripted players keep the data broader:

| `--opponent` | Opponent |
|---|---|
| `self+search:W` | the learner's own network, with search |
| `self:W` | the learner's own network, from its policy |
| `PATH+search:W` | a frozen checkpoint, with search |
| `PATH:W` | a frozen checkpoint, from its policy |
| `random:W`, `greedy:W`, `lookahead:W` | a scripted player |

`W` is the weight with which that opponent is drawn. Opponents never
explore, and only the learner's decisions are recorded.

## What is recorded

One row per decision of the learner that had more than one legal pair,
Team Preview included. Shards are `shard-00000.npz` and so on (compressed),
next to a `manifest.json`.

| Field | What it is | What it trains |
|---|---|---|
| `f`, `i`, `mask` | the learner's observation | — |
| `searched` | whether search made this decision. Team Preview and closed-sheet games are left to the policy. | — |
| `candidates` | the pairs searched, `[K, 2]`, padded with −1 | — |
| `target` | the search's mixed strategy over them, `[K]`, padded with 0 | the policy, on searched rows |
| `outcome` | 1, 0 or −1: how the game ended for the learner | the value, on every row |
| `value` | what the search expected, NaN if not searched | for analysis |
| `game`, `turn`, `opponent`, `explored` | bookkeeping: the game, the turn, the league member, and which candidates exploration added | for analysis |

`manifest.json` records:
- the search settings and the league;
- checkpoint and pool hashes;
- totals: games, rows, searched rows and time;
- results against each league member.

**Use the training pool.** Data generated on evaluation teams would leak
them into training. The job script refuses a pool with "eval" in its path.

## On Oscar

`scripts/oscar/expert-data.sbatch` uses these defaults:
- the preserved update-15342 checkpoint as the learner;
- `teams/multi-tournament-warmstart/train.json` as the pool;
- this league: `self+search:0.3`, `self:0.2`, update 13450 at 0.2,
  the Louisville checkpoint at 0.15 and `greedy:0.15`.

**Check the two frozen checkpoint paths exist first.** I took them from the
training findings; the script stops if one is missing. `EXPERT_CHECKPOINT`,
`EXPERT_POOL` and `EXPERT_LEAGUE` override the defaults.

Start with a smoke run, then size the real one from its time:

```bash
source ~/pokemon_env.sh && git pull --ff-only
ls -l runs/learning-rate-12450-ab/reference/low-lr-update-13450.pt runs/louisville-small/checkpoint.pt

EXPERT_OUT=runs/expert/round1-smoke EXPERT_GAMES=200 \
  sbatch $SLURM_OPTS --time=01:00:00 scripts/oscar/expert-data.sbatch
```

When it finishes:

```bash
python - <<'EOF'
import json
m = json.load(open("runs/expert/round1-smoke/manifest.json"))
print(m["games"], "games,", m["rows"], "rows,", m["searched_rows"], "searched,", round(m["seconds"]), "s")
for name, r in m["results"].items():
    print(f"  vs {name}: {r}")
EOF
du -sh runs/expert/round1-smoke
```

Scale `EXPERT_GAMES` and `--time` from those seconds and that size. A first
real round of about 10,000 games is a reasonable start: roughly 100,000
searched decisions.

**Speed and size are not measured on a GPU yet.** Opponents that search
cost about as much as the learner's own searches, so a league weighted
towards `+search` opponents runs slower.
