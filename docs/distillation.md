# Distillation, then self-play

The round 1 size test (`docs/expert-iteration-round1-review.md`) found that
a bigger network can't simply replace the small one. Trained from scratch on round 1's
50,000 games, large fits the search only a little better than small from scratch
(KL 0.856 against 0.927). It overfits after about 3,000 steps. And it is far behind
the small network fine-tuned from update 15342 (0.671). What 15342 learned in PPO
is worth more than the extra parameters, and round 1's data is too small to train
a large network from nothing.

So the large network first learns from 15342 itself, and only then from play:

1. **Distil** (`pokemon_ml.distill`). A fresh large network, with 15342's
   feature settings, is trained to choose like 15342 and rate positions like it.
   No search is needed, so positions are cheap, and they are generated during
   training rather than stored. The loss:
   - the exact KL divergence from the teacher's choice to the student's, over the
     whole choice: Team Preview, or the first position's action and the second's
     given every legal first one;
   - 0.5 × the squared gap between the two values.

   Half of the sides are played by the student (`--student-play`), so it also sees
   the positions its own mistakes lead to, with the teacher's answers there.
2. **Self-play from the distilled network** (`pokemon_ml.train --init-checkpoint`).
   15342 stays in the league as a fixed opponent (`--league-fixed PATH:WEIGHT`), next
   to the run's own snapshots and the scripted players. The league's results per
   update include `fixed:<path>`, so the score against 15342 can be followed during
   training.
3. **Expert iteration with the large network as the learner**, as in round 1
   (`docs/expert-iteration.md`), with 15342 with search as a frozen league member.

## On Oscar

```bash
source ~/pokemon_env.sh
DISTILL_OUT=runs/distill/large sbatch $SLURM_OPTS scripts/oscar/distill.sbatch
```

Defaults:
- teacher `runs/plateau-comparison-15342/source-checkpoint.pt`;
- student `large`;
- pool `teams/multi-tournament-warmstart/train.json`;
- 45,000 steps, with 3.5 hours as the time limit.

`DISTILL_SIZE`, `DISTILL_STEPS`, `DISTILL_HOURS` and `DISTILL_TEACHER` change these. The learning rate is lowered gradually over `DISTILL_STEPS`, so set it to what the time allows: a large student on an A5000 runs about 4 steps a second. Set too high, the run stops before the learning rate has come down. Progress is in
`runs/distill/large/log.jsonl`: `held_out.policy_kl` should fall towards 0 and
`held_out.agreement` should rise. The summary is in `summary.json`.

Whether it worked is decided by play. Run the student against the teacher without
search, then with search on both sides; about 50% means the student has caught up:

```bash
python -m pokemon_ml.evaluation --checkpoint runs/distill/large/best.pt \
  --opponent runs/plateau-comparison-15342/source-checkpoint.pt \
  --pool teams/multi-tournament-warmstart/eval-current.json \
  --out runs/distill/large-vs-15342.json --games 3000 --seed 9502347 --open-sheets 1.0 --device cuda
```

Then self-play from it:

```bash
FINETUNE_INIT=runs/distill/large/best.pt FINETUNE_RUN=runs/finetune/large \
  sbatch $SLURM_OPTS scripts/oscar/finetune.sbatch
```

- **League:** the current network 0.4, the run's snapshots 0.3, scripted players 0.1, and 15342 fixed at 0.2. Change these with `FINETUNE_CURRENT`, `FINETUNE_FROZEN`, `FINETUNE_SCRIPTED` and `FINETUNE_FIXED`; the last takes space-separated `PATH:WEIGHT`s, and all the weights must sum to one.
- **Resuming:** submit again with the same `FINETUNE_RUN` and it carries on from its last checkpoint.
- **Learning rate:** PPO's defaults apply, and other settings go after the script, like any extra argument.

A fixed opponent must stay the same for the whole run. Resuming with a different
`--league-fixed` is refused.
