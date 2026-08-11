# Current project status — 2026-08-11

This is the operational source of truth for the current Generals RL work. The
older files in `docs/` remain valuable experiment history, but statements in
them describing the r3 incumbent or the first two continuation pilots are not
the current run state.

## Verified champion

The currently accepted local champion is the 8-layer, 32-channel self-play
checkpoint:

```text
policy: /local/data/vng205/top3-v3/selfplay-stage4-1000-node1/selfplay.npz
critic: /local/data/vng205/top3-v3/selfplay-stage4-1000-node1/selfplay.critic.npz
policy SHA-256: 3b3b707fc064635ae0448f021e0e0bef8136d963369b09caffe0c1aed637c028
```

It was trained from stage 4 for 1,000 iterations and passed the final gate on
1,600 fresh competition-distance games:

```text
trained 0.588 vs initialization 0.500, required +0.050 -> ACCEPTED
```

The independent 2,000-game arena check against the previous incumbent produced
`1145W 22D 833L`, score `0.578`, and `+54.6 Elo` with a 95% interval of
`[+24.6, +85.5]`. This is a strong local result, not a ladder or top-three
claim. The previous incumbent remains a useful control and must not be
overwritten.

The pair is also available in shared home under the names
`$HOME/top3-v2-shared/weights/selfplay-champion-gen1.npz` and
`selfplay-champion-gen1.critic.npz`. Always verify the hashes on the node before
using a copy.

## Active capacity arm

The current experiment is a function-preserving width test, not a fresh model:

```text
grown policy: /local/data/vng205/top3-v3/champion-8x64.npz
grown critic: /local/data/vng205/top3-v3/champion-8x64.critic.npz
architecture: 8x64, policy 288,714 parameters, critic 329,346 parameters
source: accepted 8x32 pair above
```

`tools.grow` verified identical argmax decisions on 720 positions before
training. The active Node 1 run was launched with stage-4 start, 1,500 total
iterations, 256 games per iteration, stage-cap 500, and minibatch 4096. It is
not a champion until a fresh direct incumbent comparison and the full promotion
suite pass.

The L4 measurements observed during this run (`89%` GPU utilization,
`16,621/23,034 MiB` VRAM, `69.64 W`) are healthy. Keep this job running while
it is making progress. Do not start a second JAX trainer on the same L4.

## What the training actually does

- `learn.selfplay` trains the two seats with the current policy. Its `--opp`
  value is for `comp-eval`; it is **not** a training opponent.
- Stage 4 covers distance 17–24; stage 5 is the competition-distance tail.
  `stage-eval` compares against the policy saved at stage entry and controls
  curriculum promotion. `comp-eval` is the fixed competition-distance progress
  readout.
- `--init` and `--init-critic` warm-start policy and critic. This is not
  training from scratch.
- The terminal reward remains the default. The critic uses the matched
  checkpoint and `--critic-lr 1e-4`; do not substitute an uncalibrated critic.
- `learn.league --oracle net` is the separate archive-mixture arm. A `clone:`
  member affects training there, unlike `learn.selfplay --opp`.

## Monitoring commands on a compute node

Set these in every fresh shell:

```bash
export PY=/local/data/vng205/venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORMS
export RUN=/local/data/vng205/top3-v3/selfplay-8x64-stage4-1500-node1-g256
```

Use `tail -f "$RUN.log"` for live progress and these summaries after a shell
or Jupyter terminal reconnects:

```bash
grep -E '^iter |^comp-eval|^stage-eval|gate on|ACCEPTED|REJECTED|KILL|COLLAPSE|Traceback|RESOURCE_EXHAUSTED' "$RUN.log" | tail -n 40

nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,noheader
```

Healthy PPO output has nonzero policy updates after critic warmup, positive or
stable explained variance, finite KL/gradient values, and no `KILL`,
`COLLAPSE`, `Traceback`, or CUDA OOM lines. The self-play win rate should stay
near 0.5 because both seats use the same policy; it is not the progress metric.

## Memory and throughput policy

Games per iteration controls rollout/statistical throughput. Minibatch controls
GPU activation memory. Keep `--games 256 --minibatch 4096` while the 8x64 run is
stable. If a new run actually hits CUDA OOM, reduce only its minibatch to 2048
and use a new run directory; do not mutate or resume the current run with a
different memory shape. The current 89% GPU utilization means the trainer is
already compute-bound, so switching to the scan backend is an optional future
benchmark, not a required fix.

Use Node 2 for a separate seed/algorithm arm if more compute is available. Do
not run two trainers on one L4, and do not copy venvs or replay lakes through
shared home.

The next Node 2 arm should use the already-verified strategic module:
`tools.grow --strategy-hidden 64` adds global/3x3 region interactions and
residual spatial refinement while preserving the accepted 8x32 policy at
initialization. First train this pair with ordinary staged `learn.selfplay`
using the same curriculum and evaluation protocol as the accepted champion.
That isolates the architecture change. Only if that candidate passes the fresh
incumbent test should it enter `learn.league --oracle net` against the accepted
champion, previous incumbent, and generated hunter/greedy/expander anchors. In
that second phase set `--nn-sigma-floor 0.50`; raw support 1 means single-
opponent PPO and should not be trusted. Keep terminal reward and the matched
critic in both phases.

## Promotion rule

For every candidate, first run a fresh direct test against the accepted
champion with at least 2,000 paired games and zero faults. Then run pinned snipe
tests and the versioned promotion suite. A high score against `greedy`,
`hunter`, or `ours:configs/v16.json` alone is not evidence of a stronger bot.
Keep the accepted champion untouched until the candidate passes.

## Search and belief status

The repository contains an opt-in shallow `SearchPolicy` that reranks a small
number of visible one-action candidates with a gated critic. It is not the
shipped default. There is no learned hidden-state belief network, multi-ply
opponent search, or test-time tabular update yet. Those are future research
arms and must earn separate runtime and arena gates before deployment. See the
[Ataraxos paper](https://arxiv.org/pdf/2511.07312) for the research comparison;
do not describe the current bot as implementing that method.

## Installation invariant

An archive is not installed until all checks pass under `set -euo pipefail`:

```bash
test -f "$BUNDLE"
test "$(sha256sum "$BUNDLE" | awk '{print $1}')" = "$EXPECTED_SHA"
tar -tzf "$BUNDLE" >/dev/null
test -f "$ROOT/learn/selfplay.py"
"$PY" -m learn.selfplay --selfcheck
"$PY" -m learn.league --selfcheck
"$PY" -m learn.netoracle --selfcheck
```

Never print `*_OK` after a failed command. The shell must stop at the first
failure; this prevents an empty Node 2 directory from being mistaken for a
working codebase.
