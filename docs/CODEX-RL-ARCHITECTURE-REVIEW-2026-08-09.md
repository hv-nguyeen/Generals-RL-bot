# Competitive RL architecture review — 2026-08-09

> **Execution update:** the Node 2 `ceiling-r2` neural-league pilot completed
> 300 PPO iterations but failed its fresh gate (`0.539` trained versus `0.504`
> init; `+0.071` required). It is negative evidence for that mixture and
> hyperparameter setting, not a candidate. Node 1 remains unaccepted; keep the
> immutable C=40/context incumbent as the reference.

## Decision

Do **not** spend the next two-L4 iteration on width alone or another blind
mirror-self-play run. The strongest measured path is:

1. finish or stop the remaining Node 1 arm and directly evaluate any viable
   checkpoint against the actual `cb8d...` incumbent;
2. test function-preserving dense global/regional context matrices, because
   the unresolved receptive-field hypothesis is now a P0 architecture question;
3. repair the critic's evaluation protocol and refresh it whenever the policy
   moves out of distribution;
4. replace weak/inert league members with learned exploiters and explicit style
   buckets;
5. close the raw-training versus TTA-serving symmetry gap;
6. only then add a small recurrent style state.

There is no defensible way to guarantee that one iteration will beat every bot.
The current historical gap is over 1,000 Elo to the leader. What this project can
guarantee is a promotion contract: no checkpoint ships unless it beats the
incumbent on fresh paired boards with a positive confidence-bound margin, meets
minimum scores against every measured playstyle, stays within the move-time
budget, and introduces no engine or observation regressions.

## What the evidence now says

The ladder brief establishes the externally visible failure: against the top six
the bot scored 0.014 (0W/1D/35L). The proposed mechanism is an economy deficit:
the 36-game diagnostic has point estimates of -33 army and about -1 castle at
turn 250, with castles preceding a >=20-army deficit in 11 of 14 applicable
games. Those derived means have no reported standard errors, and the ordering is
not causal proof: being weak can itself prevent a build because the heuristic's
gather threshold is 400 army. Treat the production story as the leading
hypothesis, not a confidence-qualified finding. The later stratum-specific
counterfactual castle experiment is the stronger evidence for castle value.

One important claim in the brief has since been falsified. Mirror self-play **can**
learn castle use when its critic is competent and calibrated:

- true counterfactual castle value is +0.117 in states where a competent builder
  would build, versus -0.129 in other build-legal states (6.6 sigma separation);
- temperature calibration raised the critic's relevant explained variance from
  -0.054 to +0.117;
- after the critic/value-loading fixes, `bld` rose from 0.60 to 2.29 in about 23
  iterations and `bldA` became positive.

Therefore the bottleneck is no longer “the reward contains no castle signal.” It
is distributional: the critic becomes stale as the actor changes, training sees
too narrow an opponent distribution, and the served policy is a different
function because it uses test-time symmetry averaging and a guard.

The current incumbent is also materially stronger than the ladder artifact:
`incumbent-refresh-onpolicy.npz` beat the archived incumbent by +106 Elo over
2,000 local games. It must be treated as the reference, not the old submitted
ZIP. Node 2's completed pilot failed its gate, while Node 1 remains an
experiment rather than a candidate until it passes a direct `ship:candidate`
versus `ship:incumbent` test.

## Ranked findings

| Priority | Finding | Expected value | Cost/risk | Decision |
|---|---|---:|---:|---|
| P0 | The plain 8-layer trunk has radius 8; the current one-shot context sends only pooled summaries after all local computation | High; long-range spatial coordination remains an open, directly evidenced architecture hypothesis | Medium | Test zero-initialized dense global/regional broadcast after each layer, preserving the incumbent function at initialization |
| P0 | One holdout is used for both critic checkpoint selection and the final gate | High; current critic readiness can be optimistic | Low | Add source-stratified train/selection/test games before another expensive run |
| P0 | Training uses the raw network while deployment uses TTA plus the general guard | High; measured TTA value is about +31 Elo | Medium | Train with symmetry augmentation; evaluate only the full `ship:` artifact |
| P0 | The promotion suite contains a nearly inert `snipe@120` bucket and saturated heuristic bots | High; a candidate can pass without demonstrating style robustness | High | A useful replacement must be a trained exploiter; retain the old bucket only for historical comparability |
| P1 | GAE lambda was fixed at 0.95 | Medium; it controls reliance on critic bootstraps versus Monte-Carlo advantage | Low | Exposed as a resumable parameter; compare 0.95, 0.99, and 1.0 after each material critic change |
| P1 | The critic head is linear after pooled board summaries | High; castle value depends on interactions among safety, timing, location, and opponent pressure | Low | Add a zero-initialized residual MLP while retaining the old linear skip |
| P1 | Sixteen hand-authored memory planes cannot infer a hidden opponent style or long temporal pattern | High for adaptation | High | Add a 64-unit recurrent global state after P0/P1 instrumentation is trustworthy |
| P1 | PASS pools over all 21x21 cells, including padding | Correctness/generalization across board sizes | Low but checkpoint-changing | Introduce as a versioned architecture flag and promote by A/B, never silently alter the incumbent |
| P2 | BC uses a full 3,970-way softmax and top-1 selection without legal masking | Medium; wastes probability mass and corrupts calibration | Medium schema migration | Build dataset schema v2 with legal masks and closed-loop evaluation |
| P2 | GPU rollout rewrites have repeatedly missed the actual wall-clock bottleneck | Low near-term value | High opportunity cost | Keep CPU rollouts; make one final `epochs=0` D2H timing measurement, then stop unless it shows a large movable fraction |
| Closed | Neutral-castle refog on competition maps | None: this ruleset strips every neutral castle before turn 1 | A serving change would create needless checkpoint mismatch | Do not change the incumbent feature; assert the map invariant instead |

## Current architecture: retain, repair, then widen

### Policy

The deployed plain 8x32 CNN is small (~79k parameters), fast, and already has
a legal action mask, temporal observation planes, whole-board/regional pooling,
TTA, and a general guard. Width alone is not the evidenced limit, but spatial
reach remains open: eight 3x3 layers have radius 8 while a 21x21 corner-to-corner
dependency is distance 20.

The current context operation means the output is not literally local-only: it
computes global and 3x3 regional channel means. But it multiplies each channel by
one scalar and injects the summaries only after all local convolutional layers.
It cannot perform repeated long-range spatial reasoning, and no channel can use
another channel's pooled statistic. This is the unresolved receptive-field
hypothesis, now P0 rather than a later capacity refinement.

The first architecture experiment should keep the current final mixer and add a
dense broadcast residual after every trunk layer:

```text
h_l' = h_l + W_global,l mean_valid(h_l) + W_region,l mean_region(h_l)
```

Initialize every new matrix to zero so the migrated checkpoint is exactly the
same function, while retaining the existing final diagonal scales. Dense CxC
global and regional matrices add 2,048 weights per injection at C=32. Compare it
to the unmodified policy at equal environment steps and wall-clock. A second arm
with dilation or a deeper residual trunk is justified only if this context-matrix
test fails; do not combine reach mechanisms in one arm.

The PASS head currently uses an unmasked spatial mean. A smaller board therefore
changes its PASS logit merely because it has more zero-padded cells. A new
checkpoint schema should use the same valid-cell mean already used by the
context path. This is a real defect, but changing it in the existing checkpoint
would change behavior without evidence; gate it like any other architecture.

### Memory and style adaptation

The deterministic temporal memory should remain the auditable source of map
facts. The proposed neutral-castle patch was retracted after a second audit:
competition generation strips every neutral castle, which is the premise that
makes a new post-turn-1 structure-in-fog an enemy-built castle. On a hypothetical
mode with neutral castles, a refogged neutral castle and the same castle captured
in fog emit the same local structure token; suppressing that token forever trades
a false positive for a false negative. The existing incumbent was trained on the
competition semantics, so changing this plane at serve time would be unjustified.
The code now preserves its semantics and tests the no-neutral-castles invariant.

For style adaptation, add a separate learned global recurrent state, not a
replacement for the deterministic planes:

- input: valid-masked global mean/max, scalar totals/deltas, previous action
  class, build event, incoming-threat summary, and visible opponent change;
- state: GRU-64;
- output: FiLM scale/bias applied to trunk channels plus an auxiliary opponent
  style prediction head;
- training: 32-turn burn-in, 64-turn gradient unrolls, exact hidden-state reset
  and checkpointing in every rollout branch;
- serving: one recurrent update per observation, comfortably below the current
  150 ms budget.

Start with a GRU rather than a Transformer. Recurrent replay already creates
parameter-lag and hidden-state-staleness failure modes; a Transformer adds a
larger optimization surface before the data pipeline can verify memory use. If
the GRU fails to improve partial-observation and style buckets, a gated
Transformer-XL is the next justified test.

Auxiliary style labels should be produced from outcomes, not hand-authored names:
early pressure rate, time-to-contact, build timing, expansion rate, army
concentration, and general-directed path mass. Cluster these trajectories, then
ask whether the hidden state predicts the cluster and whether policy behavior
changes conditionally. The auxiliary loss is useful only if terminal win rate
and worst-style score improve.

### Critic

The critic currently concatenates mean, max, and nine regional means and applies
one linear vector. Preserve that linear prediction and add a residual interaction
head:

```text
V(x) = tanh(w_old^T x + b_old + W2 relu(W1 x + b1))
```

Use 64 hidden units and initialize `W2=0`, giving an exact migration. The test is
not in-sample loss. It must improve game-balanced explained variance over the
scalar control on an untouched final test set, separately for:

- early turns `<100`;
- castle-eligible strategic states;
- active general-threat states;
- midgame `100–300`;
- late game `>300`;
- each data source and opponent-style cluster.

Use three disjoint, complete-game splits. Selection chooses the epoch and
temperature. The final test is evaluated once. Fit the scalar control with the
same game-balanced weights as the neural critic; otherwise the comparison is not
like-for-like.

A training-only omniscient critic is tempting, but a naive state-only critic can
bias policy gradients in a partially observed game. The safe first design is a
history-conditioned observation critic as the authoritative GAE baseline plus a
privileged full-state auxiliary critic during training. Distill its representation
or value into the observation critic. Use a formally unbiased asymmetric
actor-critic construction before allowing privileged state directly into the
advantage estimator.

### Credit assignment

`gamma=1` is correct for terminal-only zero-sum outcomes. Lambda does **not**
change the critic target or give that target a longer reward horizon: the critic
always trains on the complete-game Monte-Carlo outcome `z`. In the actor's GAE,
lambda interpolates between one-step critic bootstrapping and Monte-Carlo:

```text
lambda = 1  =>  A_t = z - V(s_t)
```

Thus higher lambda reduces bias from an inaccurate critic and increases
trajectory-level variance. The optimal value depends on critic quality and must
be re-tested after a critic architecture, calibration, or data-distribution
change. The code exposes and records lambda without changing the historical
default.

Evaluate all three already-supported endpoints:

| Arm | Lambda | Interpretation |
|---|---:|---|
| Control | 0.95 | historical, more critic bootstrapping |
| Intermediate | 0.99 | mostly Monte-Carlo with a small bootstrap contribution |
| MC endpoint | 1.0 | exactly `z - V(s_t)` for `gamma=1` |

First compute advantage variance, sign agreement, build-stratum statistics, and
policy-gradient norm for all three values on the **same saved rollout buffers**.
Then run matched policy arms on identical seeds and compute. Compare KL/rewinds,
early-death rate, castle diagnostics, and paired Elo.

`--warm-evar` interacts with this test. At lambda near 1, a poor critic increases
variance but no longer injects multi-step bootstrap bias, so a hard evar freeze
can stop the least critic-dependent arm. Report actor-update fraction and run a
short gate probe comparing the standing evar gate with the existing fixed-warmup
mode (`--warm-evar 0`). Do not attribute a frozen-arm difference to lambda. A
full run should use a gate selected before inspecting its Elo result.

### Symmetry

BC already has correct dihedral action relabeling, while PPO does not apply it;
deployment averages symmetric views. Add one random dihedral element per PPO
minibatch, transforming observations, legal masks, and action indices together.
Recompute old/reference log probabilities for the transformed sample under the
frozen behavior/reference policies. Never reuse the original action's scalar
log-probability after transformation.

This is the lowest-risk route to close the training/serving gap. If it helps,
consider a group-equivariant convolutional trunk later; do not introduce both in
one experiment.

## Training population: from mirror play to an adversarial curriculum

Mirror play should remain part of training, but it should no longer define the
entire game distribution. The current randomized sniper barely fires and the
heuristic anchors are often saturated, so their nominal presence is not style
coverage.

Maintain four roles inspired by league training:

1. **Main policy** — optimized for the robust mixture and eligible to ship.
2. **Main exploiter** — reset or warm-start periodically and trained only to beat
   the current main; never eligible to ship.
3. **League exploiter** — targets a prioritized mixture of historical main
   checkpoints to rediscover forgotten weaknesses.
4. **Behavior/style anchors** — top replay clones and only those scripted bots
   whose behavior actually fires often enough to define a distinct bucket.

Recommended rollout mixture for the main, subject to A/B:

- 40% current/recent main self-play;
- 30% current main exploiters;
- 20% prioritized historical mains and league exploiters;
- 10% externally sourced style clones or validated scripted anchors.

Prioritize opponents whose paired score against the main is near 0.5 and add a
floor for the worst matchup. Do not let a Nash point mass or a uniform floor hide
the fact that most members are inert. Archive a challenger only when it is a
statistically credible exploiter or when it adds behavior-space novelty.

Competition-state restarts can improve sample efficiency without shaping the
reward. Snapshot simulator state **and** observation memory from real full games,
then restart around turns 80–180 with oversampling of:

- build-legal competent states;
- early general pressure and defended rushes;
- first contact with alternative build timings;
- states where castle ownership changes.

Keep at least 75% full games initially and evaluate exclusively from normal game
starts. Reweight or report by the natural evaluation distribution so a policy
cannot win the benchmark by specializing to the restart sampler.

## Two-L4 execution plan

### Phase 0 — finish the work already in flight

Do not redirect either node mid-run. When each arm finishes:

1. verify the exact incumbent and candidate hashes;
2. run fresh paired `ship:candidate` versus `ship:incumbent` games;
3. run pinned rush tests and the unchanged promotion suite for historical
   comparability;
4. report the sniper's actual firing rate, not just its score;
5. accept neither arm unless the direct incumbent lower bound is non-negative
   and no style bucket regresses materially.

### Phase 1 — receptive field and critic repair

Run two separate matched cycles so the effects remain identifiable:

- Cycle A, Node 1: unchanged 8x32/context control;
- Cycle A, Node 2: function-preserving diagonal-to-matrix dense context;
- Cycle B, Node 1: current linear critic on the identical three-way data split;
- Cycle B, Node 2: critic linear-skip-plus-MLP on the same data/split;
- both: CPU rollouts with one JAX learner process per L4 and 60 workers;
- pause the 60-worker job for side evaluations, or cap side jobs at 16 workers;
- run the final D2H `epochs=0` timing once and abandon rollout rewrites unless
  the measurement contradicts the existing wall-clock decomposition.

Promote dense context only on paired full-game results, not BC top-1. Promote the
critic architecture only if it wins on untouched games and improves policy
training stability. A critic is an instrument, not a deployable candidate.

### Phase 2 — highest-value policy A/B

With the accepted critic and identical starting policy, use two more separately
identified cycles:

- Cycle A, Node 1: raw PPO control;
- Cycle A, Node 2: random dihedral PPO augmentation;
- Cycle B: compare lambda 0.95, 0.99, and 1.0. Two run concurrently and the third
  follows on the first free node, using identical saved-buffer diagnostics first;
- measure the interaction between lambda and actor eligibility; do not blindly
  apply the default evar freeze to the Monte-Carlo endpoint;
- regenerate competition-distance value data and refit rather than training
  through a stale critic.

Compare at fixed environment steps **and** fixed wall-clock. The winner still
must pass the full promotion contract.

### Phase 3 — league and adaptation

- Node 1 continuously trains the main against the robust population.
- Node 2 alternates main-exploiter and league-exploiter jobs, publishing only
  statistically valid or behaviorally novel opponents into the archive.
- Every 20 policy iterations, freeze both sides, refresh the cross-play matrix,
  refit/calibrate the critic if its held-out phase/style gates fail, and resume.
- After the league produces real diversity, run the GRU-64 versus feed-forward
  A/B. Before that point a recurrent model has little style variation to infer.

## Promotion contract

The primary endpoint is game outcome, never build count alone.

A candidate ships only when all of the following hold on fresh maps and paired
seats:

1. **Champion superiority:** 2,000 games against the incumbent; the 95% paired
   lower confidence bound is at least +25 Elo for a normal upgrade. A +0 Elo
   lower bound may preserve an experimental candidate for more testing but is not
   enough to replace a stable release. This is 1,000 paired boards, not 2,000
   independent samples. Historical interval simulations in the brief required an
   observed effect of roughly +34 to +66 Elo depending on within-board seat
   correlation; plan for a +50–60 Elo candidate, not a +30 Elo point estimate.
2. **Style floors:** use 2,000 games (1,000 paired boards) **per** validated rush,
   economy, castle-control, fog-hunter, and archive-exploiter bucket. Require a
   simultaneous lower score of at least 0.45. For five buckets, use predeclared
   Holm correction or conservative one-sided 99% per-bucket bounds. The current
   evaluator's separate 95% intervals are diagnostics, not yet a familywise 95%
   contract; extend it before calling this gate implemented.
3. **External generalization:** repeat on a seed/map range untouched by training,
   critic fitting, and checkpoint selection.
4. **Strategic diagnostics:** report first-castle turn, castle count by turns
   150/250, army and land differences at turns 100/150/250, general deaths before
   turn 100, build-site counterfactual value, and opponent-conditioned results.
   These diagnose a win-rate change; they do not override it.
5. **Operational safety:** zero protocol faults, zero illegal actions, deterministic
   artifact hashes, complete manifest, and p99 move time below 125 ms with the full
   shipped TTA/guard path.
6. **Regression:** engine differential, observation parity, memory parity, resume
   identity, and full unit/integration suites pass.

Use sequential stopping only with a predeclared test. Do not repeatedly inspect
ordinary win-rate intervals and stop on a favorable fluctuation.

## Experimental discipline and falsifiers

| Hypothesis | Clean test | Reject when |
|---|---|---|
| Less critic bootstrapping improves the actor | 0.95 vs 0.99 vs 1.0, same buffers/seeds and fixed compute | no paired Elo/stratum-A gain, or variance/KL/rewinds dominate |
| Nonlinear critic captures strategic interactions | residual MLP vs linear head, same split/data | final-test gain over scalar control is absent or does not predict safer PPO |
| Symmetry training closes train/serve gap | PPO augmentation vs raw PPO; measure raw and `ship:` | shipped Elo does not improve or raw-to-TTA gap remains unchanged |
| Learned exploiters improve robustness | mirror mix vs four-role population | champion gain is accompanied by a worse worst-style lower bound |
| Midgame restarts improve sample efficiency | 25% restarts vs 100% full games | fixed-wall-clock gain disappears on full-start evaluation |
| GRU learns opponent adaptation | GRU-64 vs feed-forward on same diverse league | no OOD style gain, state is not predictive, or serving latency/instability rises |
| Dense context matrices improve global-to-local coordination | diagonal-to-matrix context vs exact incumbent | no paired Elo/strategic-diagnostic gain after equal compute |

One variable per arm is the default. The only exception is a correctness fix that
has an exact equivalence or regression test. Archive all rejected arms with their
manifests; never silently recycle their critic or optimizer state.

## Changes made during this review

- Retracted the proposed neutral-castle serving change. Competition maps strip
  neutral castles, while a hypothetical fog capture cannot be assigned to a
  specific neutral castle from the structure token alone. Preserved the
  incumbent's feature semantics and added a competition-map invariant test.
- Exposed `--lam` in `learn.netoracle` and `learn.selfplay`, forwarded it through
  `learn.league`, stored it in resume identity, and preserved compatibility only
  for old default-0.95 resumes.
- Added an explicit opt-in neural-oracle reward mode: the default `terminal`
  objective remains bit-identical, while `--nn-reward-mode tempo` applies a
  bounded fast-win/slow-loss tie-break only to critic/GAE targets for a learned
  exploiter. Promotion and evaluation still use raw W/D/L, and mode/strength
  are part of the league resume identity and oracle manifest.
- Corrected the GAE documentation and added the lambda=1 identity test
  `A_t = z - V(s_t)`. Kept the default at 0.95; 0.99 and 1.0 remain experiments.
- Kept the original league checkpoint parameters in resume mismatch errors while
  still migrating an old implicit lambda only when it is the historical 0.95.
- Replaced the critic's reused holdout with source-stratified, complete-game
  train/selection/test partitions. Epoch and temperature selection see only the
  selection games; readiness is decided once on sealed test games against a
  game-balanced scalar control fitted only on training games.
- Added a linear-skip residual value head. Its final layer starts at exactly
  zero, so old critics and predictions migrate unchanged while the new branch
  is trainable immediately. Serving, warm-start, checkpoint growth, probing,
  and temperature calibration all understand the new schema.
- Added dense CxC global and regional context mixers with an exact
  diagonal-to-matrix migration. Both NumPy serving and JAX training accept the
  old vector checkpoints and the new matrices; the grow self-check verifies
  nonzero diagonal mixers preserve logits.
- Added `--augment` to PPO as a separately resume-identified arm. Observation,
  legal mask, and action relabeling share one tested dihedral map, and both old
  and reference log-probabilities are recomputed on each transformed minibatch.
- Upgraded the promotion suite to explicit per-bucket game counts and a
  Bonferroni-corrected family restricted to signal-bearing style floors; inert
  hunter/greedy/expander buckets remain smoke checks without consuming the
  sniper's alpha. The suite records the approximate score detectable at its n.
- Added an independent equivariant-function regression test coupling the PPO
  observation gather to its action relabel for all eight dihedral elements.
- Ran the complete local test suite successfully.

## Research basis

- Vinyals et al., [Grandmaster level in StarCraft II using multi-agent
  reinforcement learning](https://www.nature.com/articles/s41586-019-1724-z):
  adaptive league roles and counter-strategies rather than a single mirror
  opponent.
- Baisero and Amato, [Unbiased Asymmetric Reinforcement Learning under Partial
  Observability](https://arxiv.org/abs/2105.11674): privileged critics under
  partial observability require care to avoid biased policy gradients.
- Kapturowski et al., [Recurrent Experience Replay in Distributed Reinforcement
  Learning](https://iclr.cc/virtual/2019/poster/648): recurrent-state staleness,
  parameter lag, and burn-in are first-class distributed-training concerns.
- Parisotto et al., [Stabilizing Transformers for Reinforcement
  Learning](https://proceedings.mlr.press/v119/parisotto20a.html): gated
  Transformer-XL is a later memory option if a simpler recurrent model is
  insufficient.
- Cohen and Welling, [Group Equivariant Convolutional
  Networks](https://proceedings.mlr.press/v48/cohenc16.html): architectural
  rotation/reflection weight sharing is a justified successor to successful
  data augmentation.
- Espeholt et al., [IMPALA](https://proceedings.mlr.press/v80/espeholt18a.html):
  actor/learner decoupling and V-trace are appropriate if policy lag later
  becomes the throughput bottleneck; current measurements do not justify that
  rewrite yet.
