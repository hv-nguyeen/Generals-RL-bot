# Top-3 v3 implementation and two-L4 runbook

Date: 2026-08-09

This iteration is implemented and locally verified. It does not claim that an
untrained architecture is stronger, and it does not call a checkpoint improved
until the independent promotion suite passes. The purpose of the first cluster
run is to answer one clean question: does global spatial strategy capacity improve
the current incumbent under the same data, seed, optimizer, and wall-clock budget?

## What changed

### Global strategy path

The policy and critic can now add a global spatial strategy module after the
convolutional trunk. It pools global mean/max features plus an ordered 3x3 board
partition, mixes all eleven tokens through an MLP, applies region-specific
scale/bias modulation, and performs two residual spatial refinement convolutions.

`tools.grow --strategy-hidden N` initializes every output back into the incumbent
function at exactly zero while initializing its incoming paths nonzero. Therefore
the migrated checkpoint chooses the same actions before training, but the new
paths receive useful gradients immediately. This removes fresh-initialization as
a confound.

For the current 8x32 policy and `--strategy-hidden 64`:

- parameters: 79,786 -> 158,314 (1.98x);
- one-thread NumPy forward on a 21x21 frame: 0.590 -> 0.833 ms;
- migration check: identical argmax on 720 positions, maximum logit difference
  `0.00e+00` for the direct 8x32 migration and `5.96e-08` across the grow
  self-check matrix.

### Strategic state

The observation is now C=43. Three new planes are appended, so C=24 and C=40
checkpoints and datasets remain function-preserving through zero extension:

1. a conservative enemy-general location prior: every still-possible hidden cell
   consistent with terrain visibility and the minimum general distance;
2. exact geodesic distance from the bot's original general, clipped and normalized;
3. the rules-exact castle construction cost at the current turn.

The NumPy and JAX encoders implement the same update. These features expose
general safety, expansion distance, fog uncertainty, and the time-dependent
castle economy without embedding a scripted strategy in the policy.

### Critic and sampling

- `--critic-replay-games` enables a bounded critic-only replay ring.
- Episodes are sampled uniformly, then turns uniformly, so long games do not
  dominate the value loss merely because they contain more rows.
- Neural league training also supports a source floor, preventing a large source
  from eliminating rare opponent styles from critic replay.
- Actor PPO batches remain strictly on-policy; historical actions never enter the
  policy loss.
- `--restart-frac` shifts a fraction of CPU rollouts toward midgame states through
  an exact policy and temporal-memory burn-in. It changes which prefix is recorded,
  not the on-policy action distribution. It currently saves sample bandwidth, not
  prefix compute, and cannot be combined with counterfactual forks.

### Population training

The neural oracle now has explicit `main`, `main-exploiter`, and
`league-exploiter` roles. Training can use Nash sampling or PFSP:

- `variance` emphasizes opponents near 50% win rate and is the main-policy default;
- `linear` and `squared` emphasize current weaknesses and are appropriate for
  exploiters;
- the original Nash mixture remains the evaluation and promotion distribution.

PFSP is resolved against the exact archived checkpoint being continued. Ambiguous
or missing archive identity fails closed rather than silently training against an
unrelated payoff row.

### Promotion contract

`tools.evaluate` now consumes a promotion attempt before playing games. Each full
attempt receives disjoint development and confirmation seed blocks; interrupted
or failed attempts remain consumed. The configured maximum number of attempts is
part of the family-wise correction.

A promotable result must pass both blocks, all statistical buckets, zero-fault
operation, and the runtime ceiling. The suite requires four behaviorally active
style buckets: rush, economy, castle, and fog. These are currently deterministic
style proxies. Replace them with accepted role-specialized neural checkpoints as
the population matures, but never remove the buckets.

## First two-node experiment: architecture only

Run the preparation once from a current source checkout. Confirm the actual
incumbent shape before using the commands; the current expected shape is 8x32.

```bash
export PY=/local/data/vng205/venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORMS

export ROOT=/local/data/vng205/generals-bot
export RUN=/local/data/vng205/top3-v3
export INCUMBENT=/path/to/accepted-policy.npz
export CRITIC=/path/to/matching-critic.npz
cd "$ROOT"
mkdir -p "$RUN"

$PY -m tools.grow --show "$INCUMBENT"
$PY -m tools.grow --net "$INCUMBENT" --out "$RUN/control-init.npz" \
  --layers 8 --channels 32
$PY -m tools.grow --net "$INCUMBENT" --out "$RUN/strategy-init.npz" \
  --layers 8 --channels 32 --strategy-hidden 64
$PY -m tools.grow --net "$CRITIC" --out "$RUN/control-critic.npz" \
  --layers 8 --channels 32
$PY -m tools.grow --net "$CRITIC" --out "$RUN/strategy-critic.npz" \
  --layers 8 --channels 32 --strategy-hidden 64
```

If the critic is inside a self-play `.resume.npz`, migrate it with `--prefix phi`
and pass the resulting file to `--init-critic`:

```bash
$PY -m tools.grow --net /path/to/run.resume.npz \
  --out "$RUN/strategy-critic.resume.npz" --prefix phi \
  --layers 8 --channels 32 --strategy-hidden 64
```

Use the same seed and flags on both nodes. Node 1 is the zero-extended architecture
control; node 2 differs only by the strategy module.

Node 1:

```bash
nohup $PY -m learn.selfplay \
  --init "$RUN/control-init.npz" --init-critic "$RUN/control-critic.npz" \
  --out "$RUN/control.npz" --backend cpu --workers 60 \
  --iters 400 --games 256 --epochs 1 --minibatch 4096 \
  --start-stage 4 --stage-cap 160 --stage-replay 0 \
  --lr 1e-4 --critic-lr 1e-4 --lam 1.0 --value-hidden 64 \
  --critic-replay-games 64 --critic-replay-frac 0.5 --augment \
  --opp ours:configs/v16.json,hunter:3,clone:"$INCUMBENT" \
  --seed 31001 > "$RUN/control.log" 2>&1 &
```

Node 2:

```bash
nohup $PY -m learn.selfplay \
  --init "$RUN/strategy-init.npz" --init-critic "$RUN/strategy-critic.npz" \
  --out "$RUN/strategy.npz" --backend cpu --workers 60 \
  --iters 400 --games 256 --epochs 1 --minibatch 4096 \
  --start-stage 4 --stage-cap 160 --stage-replay 0 \
  --lr 1e-4 --critic-lr 1e-4 --lam 1.0 --value-hidden 64 \
  --critic-replay-games 64 --critic-replay-frac 0.5 --augment \
  --opp ours:configs/v16.json,hunter:3,clone:"$INCUMBENT" \
  --seed 31001 > "$RUN/strategy.log" 2>&1 &
```

The CPU rollout backend is intentional: it is the exact, established path and
allows the parent JAX learner to use the L4. Do not add restart sampling to this
first comparison. Replay, augmentation, and the new observation are common to
both arms; only the global module differs.

Monitor `devices`, critic explained variance, policy-gradient activation, faults,
and stage/competition evaluations. A frozen critic, nonzero arena faults, or two
parents writing the same output invalidates the arm.

## Population follow-up

Only after selecting the architecture arm, create role policies from the selected
checkpoint and its matching critic. A main policy uses near-50% PFSP; exploiters
focus on weaknesses. Example main oracle:

```bash
nohup $PY -m learn.league --oracle net --nn-no-fallback \
  --nn-init /path/to/selected-policy.npz \
  --nn-init-critic /path/to/selected-critic.npz \
  --nn-role main --nn-sampling pfsp --nn-pfsp-weighting variance \
  --nn-critic-replay-games 128 --nn-critic-replay-frac 0.5 \
  --nn-critic-replay-source-floor 0.25 \
  --nn-iters 300 --nn-games 384 --nn-epochs 1 \
  --nn-lr 5e-5 --nn-critic-lr 1e-4 --nn-warm-evar 0.10 \
  --nn-sigma-floor 0.15 --workers 60 \
  --seeds 'clone:/path/to/selected-policy.npz,snipe:/path/to/selected-policy.npz,greedy,hunter,expander,ours:configs/v16.json' \
  --dir "$RUN/league-main" --out "$RUN/league-main/winner.json" \
  > "$RUN/league-main.log" 2>&1 &
```

For an exploiter, change the role to `main-exploiter` or `league-exploiter`, use
`--nn-pfsp-weighting squared`, and keep it out of the main slot unless it later
passes the complete promotion suite itself.

After the architecture decision, test restart sampling as its own paired ablation
with `--restart-frac 0.25 --restart-min-turn 80 --restart-max-turn 180` on one arm
and `--restart-frac 0` on the control.

## Acceptance sequence

1. Run a cheap smoke for construction, legal moves, runtime, and faults. A smoke
   can never promote.
2. Compare strategy versus control on fresh direct seeds. Reject the architecture
   if the paired result is inconclusive or negative, even if one diagnostic peaks.
3. Compare the winner versus the accepted incumbent.
4. Run the full registered suite once for the actual candidate. Do not tune on its
   confirmation block.

```bash
export JAX_PLATFORMS=cpu
$PY -m tools.evaluate --candidate /path/to/candidate.npz \
  --reference "$INCUMBENT" --suite evaluation/top3-v2.json \
  --registry evaluation/attempts.json --attempt-label top3-v3-arch \
  --out "$RUN/promotion-arch" --workers 60
```

Promotion means `summary.json` has `passed: true`; `.best.npz`, training Elo, or a
single favorable matchup is not a promotion decision.

## Local verification completed

- 52/52 repository integration tests;
- NumPy/JAX engine and encoder seam exact to `1.2e-07` with official-engine
  transition equality;
- temporal memory coverage includes all history and strategic planes;
- train, self-play, neural-oracle, league, grow, and vector-rollout self-checks;
- Python compilation and `git diff --check` clean.

The remaining evidence must come from the two L4 nodes: trained checkpoints,
fresh arena outcomes, and a registered promotion result. No software change can
guarantee that evidence in advance; the pipeline guarantees that regressions and
repeated test-set fishing are not mislabeled as progress.
