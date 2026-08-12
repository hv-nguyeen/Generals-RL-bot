# Search and Reanalysis Audit — 2026-08-12

## Decision

Do **not** spend the next two L4 runs on a longer PPO continuation, a larger
policy, or an AlphaZero loop.  First run a frozen-policy search falsification
probe.  The probe must answer two narrower questions on states the champion
actually reaches:

1. Does exact-engine top-k reanalysis find action improvements whose paired
   lower confidence bound is positive?
2. Does the matched critic preserve the terminal-rollout ordering of nearby
   actions well enough to truncate those rollouts?

`tools/searchprobe.py` implements this bounded experiment.  It deliberately
stores raw action-return samples rather than hard best-action labels.  Each row
uses the actual hidden simulator state that generated its observation and is
therefore stamped `single_actual_hidden_state` and `belief_safe: false`.  A
positive result is an upper bound on a possible observation-conditioned
teacher, not permission to deploy perfect-information search.

The cluster run has now answered the falsification question.  On 256 states
with 16 rollouts, the same-sample robust gain was `+0.0393`, while independent
8/8 selection/evaluation was `-0.00244` with 95% game-cluster interval
`[-0.01465, +0.01050]`.  Direct per-row search distillation therefore failed
its gate.  The bounded follow-up is an observation-conditioned Q ranker that
pools raw paired returns across games; see
`docs/Q-RANKER-PILOT-2026-08-12.md`.  It must pass a new sealed-game gain gate
before it can produce a policy.

No accepted checkpoint was present in this checkout or its local artifact
paths, so this audit does not report a new gameplay result or re-verify the
provided policy hash.  The supplied 2,000-game promotion result remains the
accepted empirical result.

## Present system

### Policy and observation

The accepted champion is reported as an 8-layer, 32-channel convolutional
policy.  The loader discovers architecture from checkpoint tensors rather than
assuming it.  In the present 43-channel schema, an 8x32 plain trunk with the
default diagonal global/regional context has about 79,850 policy parameters.
The optional 16x32 residual growth is function-preserving at initialization but
does not add a new source of targets.

The policy input is a padded 21x21 tensor:

- 24 current-frame channels: ownership, fog/terrain/structures/generals,
  log-armies, valid-board mask, and 12 broadcast clock/score/garrison/hidden
  army/max-stack scalars;
- 16 deterministic, observation-only memory channels: last seen owner/army,
  staleness, seen/enemy/castle history, local gains and local/global deltas;
- 3 strategic channels: enemy-general candidate prior, distance from home, and
  exact visible build cost.

`TemporalMemory` never reads simulator internals.  It is useful history, not a
posterior: it emits one handcrafted state and no probability distribution over
hidden ownership, armies, or opponent intentions.  Its general prior is also
deliberately conservative and less constrained than the richer heuristic
belief code.

The flat action space has `21 * 21 * 9 + 1 = 3,970` entries: for every source
cell, four directions times full/split movement plus one build slot, followed
by global pass.  A rules-derived legal mask removes padding and illegal moves;
split is distinct only where the source has at least three armies.  Deployment
selects the masked policy argmax (with optional dihedral averaging and safety
wrappers).

### Critic and objectives

The online critic is a separate network with the same trunk, not a shared
policy/value trunk.  Its scalar head pools global mean, global max, and nine
regional means (11 values per channel), with an optional 64-wide residual MLP,
then applies `tanh`.  Every state in a completed game receives the same direct
Monte Carlo target `z` in `{-1, 0, +1}`.  The actor uses clipped PPO,
entropy, a KL anchor to its initialization, and GAE (normally gamma 1 and lambda
0.95).  Mirror self-play trains both seats from the same current policy.

The standalone value trainer uses game-balanced train/select/test splits,
Huber loss, temperature selection, and explained-variance/ECE gates.  Those
metrics evaluate aggregate on-policy value prediction.  They do not establish
pairwise accuracy between nearby counterfactual actions.  The critic is a
`V^pi(history)` estimator: it has no action input, Q-learning objective, or
explicit off-policy successor coverage.

The matched self-play critic is saved under `phi__*`.  `ValueNet` previously
accepted only the standalone layout and treated an unmarked pooled head as a
legacy mean-only head.  The loader now accepts the matched namespace and infers
the unambiguous 11-block scalar schema, which makes the actual champion critic
auditable without rewriting its checkpoint.

## Why stable optimization did not yield a successor

### Mirror self-play

Mirror PPO estimates returns only for actions sampled from its own support.  It
contains no independent policy-improvement operator and no stronger teacher.
Once both seats enter a local equilibrium or cycle, a stable critic and small
PPO updates can faithfully optimize that distribution without producing new,
better decisions.  Increasing capacity preserves the same target bottleneck;
the failed 12x32, 8x64, and initially flat 16x32 experiments are consistent
with that diagnosis.

### Frozen-opponent `learn.rl`

This path is not a clean continuation of the champion objective:

- its JAX encoder is called without temporal memory, zeroing the 16 history
  planes and most memory-derived strategic information;
- dense land-lead increments reward expansion even when saving armies,
  building, hiding strength, or waiting is strategically correct;
- one fixed opponent rewards exploitable specialization;
- fixed rollout windows bootstrap their last row from `val[-1]` instead of a
  post-window next-state value.

The observed collapse in builds/castles and -129 Elo direct result are therefore
causal evidence against this route, not a checkpoint to rescue.

### Population neural oracle

The code can stably improve expected return against its sampled mixture while
failing the champion objective.  The population is mostly transitive, so the
raw Nash solution collapses to the champion; the uniform floor is added only to
the training distribution.  Weak opponents then provide cheap, abundant
improvements that dominate the gradient.  Healthy critic explained variance
shows that the optimizer predicts this mixture's returns; it does not show that
the candidate has learned decisions that beat the champion.  The fresh direct
gate correctly rejected it.

## Search support and gaps

`bot/policy/search.py` is not valid game search.  It edits a fogged observation
after only our action and evaluates that synthetic board.  It omits the
simultaneous opponent move, engine conflict order, global updates, terminal
rules, re-observation/fog, and a branch memory transition.  It is
observation-only (so it does not leak hidden truth), but its transition is
wrong.  It is sequential CPU code and is neither batched nor appropriate for
training reanalysis or deployment.

Other counterfactual tools do use `sim.engine` and copied memory.  They are
mostly specialized build/defence probes, often score leaves with handcrafted
safety/material functions, and use the simulator's one actual hidden state.
`learn.vecroll` supplies fast batched initial-state rollouts but does not expose
an arbitrary-state reset/fork interface.  The exact NumPy simulator is already
differentially checked against the training engine, so it is the correct first
probe even though it is not yet GPU-fast.

The new probe does the minimum correct experiment:

- reservoir-samples both-seat positions from frozen mirror self-play;
- expands the policy's masked top-k actions;
- samples one simultaneous opponent response per rollout and reuses it across
  all own candidates;
- applies both moves through the real engine and copies both temporal memories;
- uses common continuation seeds across candidates;
- plays to terminal (or the rules draw limit), optionally recording critic
  values at a fixed depth;
- accepts an action change only when its paired lower confidence bound over the
  policy argmax is positive;
- retains every sample, proposal log-probability, action, leaf value, and
  uncertainty statistic for later audit.

This is correct for a **single simulator determinization** and deliberately not
claimed to be partial-information safe.

## Ataraxos: paper facts versus transfer inference

Facts below are from the primary paper, [*Superhuman AI for Stratego Using
Self-Play Reinforcement Learning and Test-Time Search*](https://arxiv.org/abs/2511.07312):

- Ataraxos is for Stratego, not Generals.  Its move network has 14.7M
  parameters and its belief network 57.1M.
- The belief network is trained by maximum likelihood on hidden piece
  configurations from final self-play trajectories.  It uses a transformer
  encoder and autoregressive decoder with dropout.
- At test time it samples hidden configurations from the policy-dependent
  belief, then runs roughly 1,000 depth-40 policy rollouts, covering legal first
  actions/configurations.  Leaf move-network values are averaged.
- Search is formulated as one additional damped self-play update with a
  tabular magnetic-mirror-descent policy.  Reverse-KL regularization to both a
  magnet and the move network is load-bearing: the reported no-network-KL
  ablation falls from the base network's 2095 Elo to 1733, while full search is
  2218 Elo.
- The authors tried using search-generated self-play for network training, but
  its slowdown outweighed the data-quality gain; their networks are trained by
  direct policy self-play and search is test-time improvement.
- Their reported system uses vastly more compute/data than two L4s (208B
  environment steps, 163M games, H100-scale training).

Inference for this repository: Ataraxos supports belief-conditioned rollout
improvement and strong policy regularization, but it does not validate naive
AlphaZero reanalysis here.  We can copy the experimental principle—sample
plausible hidden states, average rollouts, and constrain the improved policy to
the network—without copying its model size or compute budget.

## Staged go/no-go plan for two L4s

### Stage 0: falsify search cheaply

Run on a CPU-capable node with the accepted policy and matched critic:

```bash
python -m tools.searchprobe \
  --weights "$HOME/top3-v2-shared/weights/selfplay-champion-gen1.npz" \
  --critic "$HOME/top3-v2-shared/weights/selfplay-champion-gen1.critic.npz" \
  --out /local/data/vng205/search-probe-gen1 \
  --games 128 --positions-per-game 2 \
  --topk 6 --rollouts 8 --critic-depth 32 --workers 32
```

Use held-out seed blocks for any tuning.  Inspect at least:

- top-k coverage and optimistic terminal oracle gain;
- `robust_change_frac` and `robust_lcb_gain` under the paired confidence rule;
- `oracle_best_at_topk_boundary_frac` (a high value means increase top-k before
  claiming the proposal set covers useful alternatives);
- critic pair accuracy, stratified by turn, gap size, and action type;
- sensitivity to rollout count, depth, opponent-response samples, and hidden
  determinization once alternatives exist.

Stop if robust gains disappear on held-out seeds or require impractical rollout
counts.  If terminal rankings exist but critic pair accuracy is near chance,
retain terminal rollouts and train an action-conditioned Q/ranker before using
critic leaves.  If both survive, shallow critic-truncated search is justified.

### Stage 1: conservative offline actor improvement

Do not train on per-row hard argmax labels.  Aggregate many hidden-state and
continuation samples for the same observation/history distribution, split by
complete game/seed, and learn `Q(h,a)` or pairwise advantages.  Cross-fit target
generation and validation.  A suitable policy target is a KL-regularized soft
improvement such as `pi_new(a|h) proportional to pi_old(a|h) * exp(A/beta)`,
restricted to actions with positive lower-confidence advantage.  Start by
updating only the action head; unfreeze the trunk only if held-out ranking and
fresh direct gates improve.  The only promotion result is a fresh 2,000-game
paired-board match against the accepted champion.

### Stage 2: minimum viable belief, only if sensitivity demands it

A full autoregressive world model is not the first implementation.  A practical
sub-100k addition can reuse the 8x32 observation trunk and predict:

- hidden opponent ownership probability per passable cell;
- a few log-army buckets or expected hidden mass;
- a categorical enemy-general location over legal candidates.

The new spatial heads themselves need only tens of thousands of weights; the
shared trunk dominates.  Training labels come free from exact self-play states.
Start with 0.5M strided, game-split frames and grow toward 2M only if held-out
calibration is data-limited.  A dense float16 43x21x21 input costs about 37.9 KB
per frame (19 GB at 0.5M; 76 GB at 2M), so store raw grids and memory fields in
compact integer/bit-packed form and encode in the loader instead.  A 4-8 KB raw
record keeps the same range near 2-16 GB.  A 0.5M-frame, 3-5 epoch pilot is only
a few thousand batches and should be capped at 12 L4-hours before expansion.

At sampling time enforce known mountains/castles, observed cells, exact
scoreboard land and army totals, and one legal general rather than sampling
independent heads blindly.  Generals' public aggregate totals make this
materially smaller than Stratego's belief problem.  Validate posterior
calibration, coverage of the true state, and search-action sensitivity across
samples; model likelihood alone is not a go criterion.

Multiple opponent-policy snapshots and small value/Q ensembles can expose
model/style uncertainty.  Score actions by a mean-minus-uncertainty or
quantile/lower-confidence criterion, not by one arbitrary determinization.

### Stage 3: only then consider a loop

An AlphaZero-style policy/value loop is not currently ready: the game is
simultaneous and imperfect-information, the existing tree surrogate is invalid,
there is no arbitrary-state batched fork API, and the critic has not been shown
to rank actions.  It becomes feasible only after Stage 0 demonstrates useful
action gaps, Stage 1 demonstrates offline improvement, and either a belief
sampler or a robust observation-conditioned Q makes the teacher information-set
valid.  At that point, add arbitrary-state batches to `vecroll` and compare the
full loop's wall-clock improvement per L4-hour against offline reanalysis.

## Verification performed

- `python -m tools.searchprobe --selfcheck`
- `python -m learn.netoracle --selfcheck`
- `python -m learn.selfplay --selfcheck`
- `python -m tests.test_all` — 53/53 passed
- one spawned-worker smoke probe wrote `rows.npz`, `summary.json`, and
  `manifest.json` from a disposable random checkpoint

The smoke probe validates plumbing only; it is not gameplay evidence.
