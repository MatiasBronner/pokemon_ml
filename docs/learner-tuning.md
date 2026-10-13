# Oscar learner tuning

The tuner compares execution settings without changing the observer, model architecture, reward, PPO epochs, learning rate or league mixture. Minibatch sweeps are explicitly separate: changing minibatch size changes optimizer-step counts and can change learning quality. No settings are applied automatically.

## Run on Oscar

Finish the current training job before updating its checkout or benchmarking. Preserve its checkpoint and snapshots. After merging the tuning branch, on the login node:

```bash
cd /users/mcbronne/pokemon_ml
git switch main
git pull --ff-only origin main
module load python/3.12.8-zr3c
source .venv/bin/activate
export POLICY_REPO=/users/mcbronne/pokemon_ml
export POLICY_POOL="$POLICY_REPO/teams/louisville-split/train.json"
export POLICY_CHECKPOINT="$POLICY_REPO/runs/louisville-small/checkpoint.pt"
export POLICY_LEAGUE_RUN="$POLICY_REPO/runs/louisville-small"
sbatch scripts/oscar/tune-learner.sbatch
```

This requests one GPU, four CPU cores, 32 GB host RAM and two hours on the existing `gpu`/`default`/`norm-gpu` allocation. Software is already installed; the changes are Python-only, so no Rust extension rebuild is required. Nothing is run on the login node except submission. Use the returned job ID in `squeue -j <number>` and `tail -f tune-<number>.log`. Reports go to `runs/learner-tuning/job-<number>/`.

The tuner copies the source checkpoint once and captures a frozen snapshot manifest before testing candidates, so all workers start with the same policy and optimizer. It reads the league snapshots, but never writes model weights, optimizer state, checkpoints or training logs to the existing run. All updates are in-memory disposable experiments. The reports and copied checkpoint need disk space; profiler traces are bounded to four collection steps, and compiler caches use compute-node temporary storage in the batch template.

## Tested settings

- Existing FP32 execution, existing metric logging, minibatch 4096.
- FP32 with deferred scalar metric logging: device accumulators and one final transfer instead of many per-minibatch scalar synchronizations.
- Minibatches 4096, 8192 and 16384 with deferred logging.
- Explicit SDPA path (disable the inference MHA fastpath), permitting PyTorch's normal backend selection.
- Fused Adam on CUDA. Optimizer moments and hyperparameters are preserved; execution can introduce numerical differences.
- Native BF16 encoder computation on Ampere-or-newer GPUs. Policy/value heads, probabilities, ratios, normalization and losses remain FP32. No FP16 or gradient scaler is introduced.
- Forced Flash SDPA with BF16, to test actual support rather than assume it.
- `torch.compile` of the tensor encoder, with eager FP32 heads. Compilation requires a full encoder graph (no silent graph-break fallback), uses dynamic batch shapes, one compiler worker, and a per-candidate timeout.

Each candidate runs in a separate process. Failures, unsupported kernels, out-of-memory errors and timeouts are recorded; timeout cleanup kills the worker's process group including compiler children. Baseline profiling and candidate reports are persisted incrementally. Compilation/validation startup and warmup are reported separately from three measured collection/update repeats. Collection includes the existing stage-timing synchronization barriers. Deferred logging removes many PPO scalar synchronizations, not every synchronization in the program. Optional auxiliary classification metrics still synchronize when those model experiments are enabled.

Hardware records include GPU name, compute capability, memory, UUID/driver inventory, PyTorch/CUDA versions, host and Slurm job. Larger-model memory needs and optional policy heads are measured if `--checkpoint` supplies them. No fixed speedup is promised.

## Numerical checks and interpretation

Before updating weights, candidates check preview and battle actions for legality, finite output, and collection/gradient-enabled evaluation likelihood agreement (1e-4 maximum error). FP32 reference log-probability drift is gated at 1e-3 for FP32 execution and 0.1 for BF16. Thresholds are coarse numerical gates, not guarantees of unchanged playing strength. Gradient updates must leave finite parameters and finite reported losses/gradients/KL/clipping. CPU tests check unchanged state-dict keys and identical parameter updates with deferred logging.

`summary.json` contains per-candidate timings, memory, correctness checks, KL/clipping/gradients/losses and failed candidates. `baseline-4096/collection-operators.txt` and `ppo-operators.txt`, plus Chrome traces, show executed attention operations and CPU/device costs. Compilation can hide operator names under generated kernels; inspect the traces when needed.

`recommendation.json` ranks only configurations with the original minibatch size. `train-flags.txt` gives the corresponding precision/attention/compiler/logging/optimizer flags, but is not automatically executed. Check per-repeat variance and rerun close candidates; a short benchmark winner is provisional. Minibatch changes require a matched-seed learning experiment, rather than just accepting a higher decisions/sec number.

The standard `train-policy.sbatch` forwards its positional arguments to the trainer. To apply vetted flags, append them to the `exec bash .../train-policy.sbatch` line in your job wrapper; do not enable them solely because one short benchmark won.

Before adopting precision/compiler/optimizer changes for long runs, compare a short separately saved continuation against FP32 with the same starting checkpoint, data distribution and evaluation seeds. Judge strength per GPU-hour as well as throughput. The benchmark itself does not test competitive improvement.

## GPUs assigned by Slurm

A generic `--gres=gpu:1` request picks an eligible available GPU. Keep a report for each GPU type and software version; do not treat an RTX A6000 result as an H100, A40 or older Quadro result. Native BF16/Flash candidates are skipped when capability is below Ampere. FP32, deferred logging, minibatch tuning and compilation remain candidates; unsupported paths still fail explicitly.

For reproducible comparisons, list the regular partition's types:

```bash
sinfo -p gpu -o '%N %G'
```

If `nvidia_rtx_a6000` is listed and your allocation accepts it, request that type:

```bash
sbatch --gres=gpu:nvidia_rtx_a6000:1 scripts/oscar/tune-learner.sbatch
```

Command-line options override the script's generic GPU request. A specific type can wait longer. Do not request types/partitions outside your allocation or cancel/requeue repeatedly to hunt GPUs. For flexible production jobs, start with portable FP32 settings; only apply a BF16 result to a GPU with native support. Unsupported requested precision raises an error instead of silently changing the experiment.

## Production runtime flags

The trainer accepts `--precision fp32|bf16`, `--attention auto|sdpa|math|flash`, `--compile-encoder`, `--deferred-metrics` and `--fused-adam` (Boolean flags also have `--no-...` forms). Runtime options are saved separately from architecture in checkpoint/config metadata and restored on resume. Explicit overrides allow returning to FP32 or disabling compilation when the next GPU differs; model/optimizer weights remain compatible. Existing baseline checkpoints load unchanged. Frozen training opponents retain their existing FP32 eager execution, so league inference overhead is represented rather than hidden.

Local validation is CPU-only. Oscar CUDA, BF16/Flash/fused-Adam kernels, driver compatibility and competitive learning must be verified on the user's allocated hardware. CPU smoke timings are not GPU performance estimates.

## Validation

[Recorded validation](learner-tuning-validation.json): 27 targeted tests passed; CPU parent runs produced profiles and recommendations; a fullgraph compiled CPU encoder passed likelihood/reference checks. GPU execution remains to be tested on Oscar.

## Investigating a collection/update consistency failure

This check runs before any optimizer update. `auto` can use PyTorch's inference-only MHA fastpath while gradient-enabled evaluation uses SDPA. A failure is not evidence that Adam, the observer, or training checkpoint is broken. Both log-probability error **and value error** are gated; the error now prints them separately. Gates are unchanged.

After the tuner finishes, update the checkout and run this on an allocated GPU (not the login node):

```bash
python -m pokemon_ml.diagnose_learner \
  --checkpoint runs/learner-tuning/job-7337666/source-checkpoint.pt \
  --pool teams/louisville-split/train.json \
  --envs 512 --threads 4 --device cuda \
  --out runs/learner-tuning/job-7337666/diagnostics.json
```

The read-only diagnostic compares `auto`, `sdpa` and `math` on the same observations/seed, in preview and battle. It reports maximum log-probability error, value error and implied PPO-ratio deviation. No optimizer is created; no weights/checkpoints are written. The report may show failed numerical gates deliberately; it does not make those configurations eligible in the tuner. Use the measured differences to decide whether this is harmless floating-point drift or a material mismatch before modifying tolerances or production execution.
