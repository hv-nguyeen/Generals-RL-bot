# Brief for Codex: architecture review and redesign of the training stack

> **Status 2026-08-11:** this is an architecture-review prompt and historical
> backlog, not a current runbook. The verified implementation is commit
> `5997415`; read `README.md`, `docs/STATE.md`, `docs/TOP3-V2-HANDOFF.md`, and
> `docs/CLUSTER.md` for the current two-node training arms. Re-check every
> “outstanding” item against the code before acting; several release/evaluation
> fixes described below have since landed, while the research proposals remain
> intentionally deferred. The Node 2 `ceiling-r2` pilot is now a completed,
> rejected experiment (`0.539` trained vs `0.504` init; `+0.071` required), and
> The 2026-08-09 Node 1 arm had no promotion result; the later accepted 8x32
> self-play checkpoint is recorded in the live-status document below.
>
> The live operational state is in
> `docs/CURRENT-STATUS-2026-08-11.md`. The accepted 8x32 self-play champion is
> the immutable baseline for an active function-preserving 8x64 capacity arm.

Paste this whole file as the prompt. It is self-contained: it carries the
measurements you need so you do not re-derive them, and the list of things this
project has already tried and rejected so you do not propose them again.

---

## Your role

You are doing an architecture review of a competition bot for generals.bot, and
then implementing the changes you can justify. The repository is at the path you
have been given. Read in this order before writing anything:

1. `CLAUDE.md` — the non-negotiables. They override anything in this brief.
2. `docs/CURRENT-STATUS-2026-08-11.md` — the live checkpoint, run, and
   monitoring state.
3. `docs/ml-log.md` — every ML attempt and local instrument that has failed, each
   with the measured reason. **This is the single most important file.** Six ML
   attempts and nine evaluation instruments have failed here. The log exists so
   none of them gets tried a seventh time.
4. `docs/STATE.md` — measured history and non-starters.
5. `docs/TOP3-V2-HANDOFF.md` — the current pipeline design.
6. `docs/NEXT-TRAINING-AUDIT-2026-08-09.md` — the most recent verification pass.
7. `docs/CLUSTER.md` — how to run anything expensive.

Then read the code: `bot/policy/net.py`, `bot/features.py`, `bot/memory.py`,
`learn/train.py`, `learn/selfplay.py`, `learn/netoracle.py`, `learn/valuetrain.py`,
`learn/rlenv.py`, `learn/vecroll.py`, `tools/evaluate.py`, `arena/rating.py`.

## The standing rules you inherit

- **Never claim a change is an improvement unless `tools.evaluate` passes the
  versioned promotion suite.** Measure `.best`, never `.live`.
- **`bot/` must stay self-contained and numpy-only.** It is the submission. It may
  not import `sim/`, `arena/`, or `analysis/`.
- **`sim/engine.py` mirrors the official engine exactly.** Touch it and run
  `make verify`.
- **Heuristic tunables live in `bot/config.py`; learned architecture is recorded
  inside every checkpoint.** Configs are strict and versioned.
- **A number at n<=36 is noise.** Ladder SE is ±58 there, ±22 at n=240.
- **Measure a candidate before building it.**
- **Hand-written overrides lose to this policy.** Five measured, all neutral or
  negative, the worst at −322 Elo.
- The competition sandbox is single-core CPU with a 150 ms move budget. Inference
  cost is not the binding constraint (20×128 costs 6.3 ms/move); **rollout
  throughput is**.

---

## Part 1 — What is already measured. Do not re-derive. Do not contradict.

### 1a. Ladder forensics snapshot, recorded 2026-08-09

Complete historical ladder record from that audit: 72 games, rank #22, Elo 1998,
24W-47L-1D. It is evidence for the analysis below, not a live leaderboard claim;
use the dated snapshot stored with the relevant artifact and re-fetch before
making a release decision.

| tier | opponents | Elo | our score |
|---|---|---|---|
| TOP | ResBot 3428, Kubic 3084, bca 2911, nanomena 2899, thor 2783, Mattz 2708 | #1–#6 | **0.014** (0W 1D 35L) |
| MID | ceryle 2094, hiems 1999, barsik 1941 | #18–#25 | 0.444 |
| WEAK | nappoleon 1845, vojtima4 1790, Hunter baseline | #32–#38 | 0.889 |

**The loss is a production loss, not a combat loss.** Army has exactly one source
and one sink under these rules (+1 per owned structure on even ticks, +1 per
owned tile on 50-ticks; combat destroys `2·min(attacker−1, defender)`; moves and
neutral captures conserve army). So cumulative production minus board army *is*
cumulative combat loss, with no estimation. On the 26 top-six games alive at turn
250:

```
                               us       them       diff
army produced               412.4      458.9      -46.5
army on the board           194.6      227.6      -33.0
destroyed in combat         218.8      232.3      -13.5
```

Combat is even to within 6% and what asymmetry exists favours us. The entire
deficit is production, and 80–86% of the production gap is castles.

**Fixed cohort timeline** (same 26 games at every row, no survivorship bias):

| turn | land diff | army diff | castles us/them | castle diff | army the castle gap predicts |
|---:|---:|---:|---|---:|---:|
| 100 | −3.5 | −3.9 | 0.00/0.00 | 0.00 | 0.0 |
| 125 | +0.3 | +4.5 | 0.04/0.35 | −0.31 | −1.9 |
| 150 | +0.2 | +6.3 | 0.12/0.62 | −0.50 | −7.0 |
| 200 | −2.1 | −13.6 | 0.46/1.27 | −0.81 | −20.0 |
| 250 | −4.5 | **−33.0** | 0.46/1.46 | −1.00 | **−39.9** |

Observed army swing t150→t250 is −39.3; the castle differential alone at
+0.5 army/turn/castle predicts −39.9. That is 101%. Land is flat at ~0 throughout
and explains nothing.

**Lead/lag within games:** castle deficit ≥1 first appears at median turn 128;
army deficit ≥20 at median turn 169; castles come first in 11 of the 14 games
where both occur.

**Where the castles go missing:** replaying our own fogged observation and
counting `features.legal_mask` build slots, we have a legal build on only **1.4%
of turns before turn 150** and 6.1% overall. They first build at median turn
**139 in 25/36 games**; we at turn **160 in 18/36**. They engineer the
35-army-on-a-safe-rear-tile situation; we wait for it to occur. This is
`castle_gather_min_army: 400` — v16 never walks army to a site.

**How games end:** 35 of 35 top-tier losses are general captures, mean turn 282
(vs 436 for MID). At the death tick: attacker 34.9, garrison 12.7, ratio 2.7x,
with 141 army on the board — 91% of our army elsewhere. Garrison share is 6–8% of
total army at *every* checkpoint in *every* tier, so it is a constant of the
policy, not something strong opponents provoke. Read it against the economy: the
kill is what a 33-army deficit cashing out looks like.

### 1b. Hypotheses measured and REFUTED on this same data. Do not propose them.

- **"We do not mass army."** Refuted. Max stack and top-1 army share are
  identical to the top six at every checkpoint (t150: 13.9/9.5% vs 13.7/9.3%;
  t250: 23.2/10.8% vs 26.5/11.4%). The WEAK tier is *more* concentrated than us
  and loses 89% of its games. Concentration does not predict strength here.
- **"We never threaten their general."** Refuted, and the first measurement of it
  was circular. Uncapped over the whole game it reads 25% vs 97% — but 97% *is
  the loss*. Capped at turn 250 on the alive-at-250 cohort: closest approach 4.7
  vs 4.0, reached 23% vs 12%, arriving stack 39.3 vs 45.6. Reach is symmetric.
- **"The opening is weak."** Refuted. Turn 50: land 23.6 vs 24.1, army 49.5 vs
  50.0. Turn 100: land −3.5. Nothing is wrong before turn 125.

### 1c. This contradicts a conclusion recorded in `ml-log.md`, and the log is the one to update

`ml-log.md` currently says *"castles are a bet on the game lasting, and against
strong opponents that bet loses"*, derived from a `head_b[8] += 9.094` experiment
that produced ~15 castles a game and dropped the ladder rank, reasoning that
"ladder opponents at 1800+ press hard and games end early".

The six bots at **2708–3428** Elo build ~1.0 by turn 250, first around turn 139,
and beat us 35–0–1. The log flagged this gap itself: *"Only the extremes were
tested"* (0 and 15) and *"an optimum plausibly sits between and remains
unmeasured."* The field's revealed answer is ~1.0–1.5, timed early — close to the
heuristic's own ~0.9. **Do not restore a constant build bias**; that exact change
is measured at −36 Elo head-to-head and a ladder rank drop. The finding is about
*timing and siting*, which a constant per-cell bias cannot express.

### 1d. The full rejected list from `ml-log.md`. Proposing any of these is a failure of this review.

Reward shaping of land lead (expectation exactly zero in a mirror); frozen-opponent
PPO (learns to farm one opponent); terminal-reward REINFORCE with no baseline
(collapsed 0.115→0.000); self-imitation from random init (plateaued 0.15);
positive-only advantage filtering; counterfactual auxiliary losses
(`--cf-*`, measured redundant against plain `buildbias`); a constant castle build
bias; hand-written overrides (five measured, worst −322 Elo); the wide garrison
lock at radius 6 (−265 Elo); top-1 accuracy as a model-selection metric (a net at
0.651 top-1 lost 198–2); evaluating against a single fixed opponent however
strong (v16 misled by +74 Elo while the ladder rank dropped); a greedy-baseline
submission bar (would have blocked a 1849-Elo bot); config-space search (league
exploitability ≈ 0.02, the 24 knobs are exhausted).

---

## Part 2 — Architecture findings to act on

These come from a code audit against the measured history. Each has a file and a
cost. **Treat them as the standing candidates, not as instructions: your job
includes challenging them and proposing better.** If you think one is wrong, say
so with the reason.

### A. Historical/remaining temporal-memory audit — verify before changing

Commit `5997415` removed the stale-owner condition from the structure-in-fog rule
in `bot/memory.py:97` and `learn/rlenv.py:84`, to fix castles built wholly in fog
being invisible. The fix over-corrected: `known_mountains` is cleared for any
visible non-mountain (`bot/memory.py:78`), so a **neutral castle that is scouted
and later re-fogs is permanently labelled `ENEMY_CASTLE`.** Reproduced on
engine-consistent frames (`visible = dilate8(own)`; castle at (4,4); we take
(3,3); opponent retakes it) — the plane flips to True at the re-fog tick. 0/60 in
v16-vs-v16 self-play because vision only shrinks when you lose land, but it fires
exactly when an opponent retakes ground near a scouted city.

Do not overload `known_mountains` — it feeds the MOUNTAIN plane
(`bot/features.py:161`) and a neutral castle is capturable, not impassable. Add a
separate `known_neutral_castles` set on visible `T_CASTLE & OWNER_NEUTRAL`,
cleared on visible `T_CASTLE & OWNER_OPP`, and exclude it from the SIF rule.
Mirror in `update_memory_jax`. Add the three-frame regression test; the existing
tests cover never-seen→SIF and first-frame→mountain but not this.

### B. Historical promotion-bar audit — implementation has moved on

`arena/rating.py` replaced the per-game SPRT with a fixed-sample empirical
Bernstein interval over board pairs (correct — the runner plays each seed twice
with seats swapped, and treating those as independent invalidates the interval).
Buckets marked `requires_paired_superiority` now need `accept H1` from that wider
fixed-sample interval. Measured
at 1000 boards with a 25-Elo margin, the observed Elo needed to pass is ~+34
(anti-correlated pairs), ~+57 (independent seats), ~+66 (correlated) — against
~+30 for the old SPRT. Current `CLAUDE.md` and handoff guidance describe the
pair-aware fixed-sample evidence; older diagnostic artifacts may still carry the
removed compatibility name `requires_sprt`. Do not compare old labels without checking
the tool version and reading the paired lower bound.

The current runner/evaluator paths now expose pair-aware fixed-sample results;
do not interpret new output as a sequential SPRT. If an older diagnostic still
prints the compatibility `sprt` key, read `paired_test`/the lower bound and
verify the exact tool version before using it as evidence.

### C. The trained object is not the shipped object

`evaluation/top3-v2.json` sets `candidate_spec: "ship:{candidate}"`, and `ship:` is
`GuardedPolicy(ClonePolicy(..., tta=cfg.tta, full=cfg.tta_full))`. Every config in
`configs/` has `tta=True, tta_full=True`. `learn/selfplay.py` and
`learn/netoracle.py` contain zero references to `tta` or `guard` — rollouts sample
from the raw net.

The guard is not the gap: `bot/policy/guard.py:114` records the default visible
trigger firing on 0.23% of turns for +5.8 Elo over 300 games. **TTA is the gap**,
and it changes the argmax every turn. PPO optimises `π_raw` while promotion and
the ladder score `avg_g(π_raw ∘ g)`.

Nobody has measured the size of it. One arena run, and it is an input to D:

```bash
python -m arena.runner --a "ship:$P" --b "clone:$P" --games 800 --workers 60 \
  --seed0 9300000 --out runs/tta-gap
```

### D. Dihedral augmentation is used for behaviour cloning and never for RL

`learn/train.py:242` `augment(x, y, g)` applies one group element to a whole
batch — pure gather, board dims from the `VALID` channel, action labels relabelled
through `_dihedral_maps`. The 2026-08-05 audit verified the permutation
exhaustively (8 elements × every cell × every direction × builds, 0 mismatches).
Grep for it in `learn/selfplay.py` or `learn/netoracle.py`: nothing.

Rollout is the binding cost. Advantage, return and value are all exactly
invariant under the group; the action index and legal mask permute through
`actmap`. The buffer at `learn/selfplay.py:2200` is already the shape `augment`
takes. `old_logp`/`ref_logp` are **not** invariant (the net is not equivariant),
so each augmented copy needs its own no-grad forward — which is the chunk loop
already at `learn/selfplay.py:2218`, run k times.

Honest cost: at 256 games/stage 0 the iteration is ~10 s with `roll` ~7 s, so k=4
takes ingest+update from ~3 s to ~12 s, ~19 s/iteration — 1.9x wall-clock for 4x
update data. This also shrinks C: a net trained on all orientations approaches
equivariance, at which point TTA stops being a function change.

**This must be A/B'd at matched wall-clock, not matched iterations.** Matched
iterations flatters it by construction.

Consider and cost the stronger version: making the trunk dihedrally equivariant
by weight-sharing, which removes the need for TTA entirely. Say whether the
action head's direction slots can be made to permute correctly with the group,
and what it costs in checkpoint compatibility.

### E. The context mixer is diagonal and fires once

`ml-log.md` names the receptive field "the most likely reason this pivot fails"
and prescribes "mean-pool-and-broadcast per trunk block". What exists is narrower
in two ways:

- **Diagonal.** `context_global`/`context_region` are shape `(C,)`
  (`learn/train.py:107`). Channel *c* receives `g_c × mean_c` — its own global mean
  and nothing else. Routing a global quantity into an unrelated local decision
  requires the trunk to have already colocated them in one channel index.
- **Once.** `learn/train.py:140` applies it after the last conv, so global context
  never passes through a convolution — only the 3×3 move head sees it. The trunk's
  receptive field is still exactly L.

Also: the critic pools mean **+ max** + 9 regional (`learn/valuetrain.py:49`); the
policy context has no max at all.

Both changes are function-preserving from the incumbent, which is what makes them
affordable — they grow through `tools.grow` exactly as C=24→C=40 did, with no
retraining from scratch:

| change | params | MACs | init that preserves the current function |
|---|---|---|---|
| `(C,)` → `(C,C)` matrix | +1,024 | +1,024 (0.007% of 14M) | diagonal = current value, off-diagonal 0 |
| apply per block | +2,048/block | same order | every block's scale 0 except the last |

### F. The critic head is a single linear layer on pooled features

`learn/valuetrain.py:42` `pooled` returns mean + max + 9 regional means = `11×C` =
352 features; `v_w` is a 1-D vector of length 352; then `tanh`. No hidden layer.
The critic cannot represent any interaction between pooled quantities — "army lead
× late clock" and "army lead × early clock" get the same weight.

`ml-log.md` puts the critic at the centre of everything: evar 0.20–0.35, castles
priced at −0.002 against a measured true stratum-A value of +0.117, and the
midgame attribution failure. One hidden layer of 64 is `352×64 + 64` = 22.6k
params, 22.6k MACs, 0.16% of a forward.

**This is the cheapest experiment available and needs no PPO compute.**
`learn.valuetrain` already owns the complete-game holdout, game-balanced metrics
and the fitted scalar-only control, and exits nonzero when the gate fails. Refit
both head shapes on identical shards and read the evar gain over control and the
ECE. Caveat that must be fixed first: `valuetrain` currently reuses one holdout
for both epoch selection and the final gate, so this needs the source-stratified
train/selection/test split before its number means anything. Treat F and that
split as one piece of work.

### G. The credit horizon is shorter than the failure it must attribute

`learn/netoracle.py:161` sets `LAM = 0.95`, with the comment "lam caps credit at
~20 steps". Each seat's episode is its own row sequence, so that is ~20 turns of
game time. The measured failure — the castle/army divergence in Part 1 — opens at
turn 128 and cashes out at turn 282, and `ml-log.md`'s midgame collapse has "its
consequence 50+ turns later".

Note the asymmetry already in the code: the actor gets a λ=0.95 advantage but the
critic's target is the full Monte-Carlo return (`learn/selfplay.py:2239`). Both
choices are individually argued in the `gae` docstring; the pairing is
unexamined. **λ is a module constant, not a flag** — expose it before anyone
sweeps it, or the sweep is a set of source edits with no resume identity.

### H. Two free measurements nobody has taken

- **`d2h` was declared the gate and never read.** `ml-log.md`: "read `roll 7.0/d2h
  0.3` and the whole hypothesis is dead; read `roll 7.0/d2h 3.5` and it is worth
  building. There is no third reading, and no number should be written here before
  that one is." Grep `docs/STATE.md` for `d2h`: no hits. The instrument was built;
  the one iteration that reads it was never run.
- **The PASS logit is the only head that ignores `VALID`.** `learn/train.py:168`
  and `bot/policy/net.py:240` both take an unmasked mean over the full 21×21 pad.
  Parity holds — both are wrong the same way. On an 18×18 board ~27% of that mean
  is padding activations, nonzero because the convs have biases, so PASS is a
  board-size-dependent function. `_context_mix` and `_pooled` both mask correctly.
  One-line fix, but it changes the function, so it needs a measurement.

---

## Part 3 — The open design question, which is the actual point of this review

The measurement in Part 1 says the deployed bot loses to the top six by failing to
build ~1 castle at around turn 140, and that this is worth ~40 army by turn 250
and the game by turn 282.

**The problem is that self-play cannot learn this.** The castle decision is
symmetric: in a mirror match both seats build, neither gains tempo, and the
gradient sees only the 35-army cost. `ml-log.md` states this mechanism twice and
it is why PPO deleted castles at every stage — `bld` 15 → 0.1 in 37 iterations. A
PSRO archive built from our own lineage cannot fix it either, because no member
builds.

`ml-log.md` also records the correct value of the decision, measured by forced
counterfactual deviation played to terminal (`tools/cfprobe.py`, 3000 games):

| stratum | n | mean delta | vs 0 |
|---|---|---|---|
| A — a build v16's castle block would take | 1794 | **+0.117** | +4.0σ |
| B — every other build-legal turn-seat | 2967 | **−0.129** | −5.6σ |

A−B = 0.246 ± 0.037, +6.6σ. Castles are good where a competent player builds them
and bad everywhere else, and ~91% of build-legal moments are the bad kind.

So the design question is: **what change to the training architecture lets a
policy learn a symmetric, timing-sensitive, opponent-dependent strategy that
mirror self-play is structurally blind to, without hand-writing the answer?**

Candidate directions, none endorsed — argue for or against each and propose your
own:

- Asymmetric archive composition, so a fraction of training games are against
  members that do build early and the decision stops being symmetric.
- A privileged full-state critic (actor sees only the deployable observation), so
  advantage estimates for a 70-turn-payback action are less noisy. Any actor
  dependence on privileged state is an immediate failure.
- Competition-state restarts: train terminal outcomes from reconstructed states
  near turns 100/140/250 with exact memory, mixed with full games, so the ~1.4%
  of turns that offer a build stop being a rounding error in the buffer.
- Longer λ (Part 2G) so credit reaches from turn 140 to turn 282.
- Observation channels for what the decision needs and cannot currently compute:
  `rules.build_cost_grid` is computable per cell, rear-safety is derivable, and
  `ml-log.md` notes that "information the net cannot compute has been worth more
  in this project than capacity or schedule."

Also worth your judgement: the 3970-way flat action space with a per-cell 9-slot
head, versus a factored head (source cell, then direction/split/build). And
whether the deterministic `TemporalMemory` should stay a verified feature layer or
become a learned recurrent belief — it is currently a hand-maintained 16-plane
state that both the trainer and the submission must keep byte-identical.

---

## Part 4 — What to deliver

Produce a design document plus patches. For **every** proposal, in this order:

1. **Mechanism.** What specifically is wrong now, with `file:line`, and what the
   change makes possible that is impossible today.
2. **Cost.** Params, MACs, ms/move against the 150 ms budget, effect on rollout
   throughput, GPU-hours. Say which numbers are measured and which are arithmetic.
3. **Migration.** Whether it is function-preserving from
   `runs/top3-v2/incumbent-refresh-onpolicy.npz`, and if so the exact
   zero/diagonal init that makes it so. Anything that forces a from-scratch
   retrain has to justify that separately.
4. **Falsifier, preregistered.** The specific reading that kills it. Not "if it
   does not help" — the field, the threshold, and the sample size.
5. **How it is measured.** Which existing gate. Sample-efficiency changes are
   compared at **matched wall-clock**. Anything scored against a single opponent
   is rejected on sight; use the archive and a champion clone so the reading lands
   near 0.5 where it can resolve.
6. **What it does NOT claim.**

Order the work by expected value divided by cost, and say why. My ordering, for
you to accept or overturn with a reason:

| # | item | cost |
|---|---|---|
| 1 | A: the neutral-castle regression | 6 lines + one test |
| 2 | H: read `d2h` | one iteration |
| 3 | C: measure the TTA gap | 800 games |
| 4 | F + the honest critic split | hours, no GPU |
| 5 | D: dihedral augmentation | one matched-wall-clock A/B |
| 6 | E: context mixer | grow + curriculum refresh A/B |
| 7 | B: document or fix the promotion bar | doc + two call sites |
| 8 | G: expose `--lam`, then sweep | after 4 |

## Part 5 — How this project fails, so you do not repeat it

Read these as hard constraints on your method, not as advice.

- **Its bugs are in its instruments, not its learning.** An adversarial
  line-by-line pass over the trainer found no correctness bug; every live risk was
  in a side tool, and a lying docstring was three of the four. Audit the thing
  that produces the number before you trust the number.
- **Nine local instruments have disagreed with the ladder in the same direction**
  — all reward committing army aggressively, because no opponent they could
  construct punished over-commitment.
- **Two throughput projections missed by 13x and by "slower than the thing it
  replaced"**, both because the number was derived from the term someone had
  already decided was the bottleneck. Measure first, derive second.
- **Three thresholds in this project turned out to be unreachable by
  construction** (stage-eval 0.60, greedy ≥ 0.90, last-decile evar ≥ 0.85). Before
  you set a bar, show it is achievable by something.
- **`.best` is close to a lottery** — the max of ~24 noisy draws is the luckiest,
  not the strongest.
- **Do not read `train-wr` as progress** (it plays the σ-floored mixture) and do
  not read `<- kept` as strength (winner's curse: 0.320 was a max over ~117 evals
  and settled at 0.255 on 400 fresh games).
- **Configs silently become each other.** `Config.from_dict` fills absent keys
  from today's defaults, so six ladder-measured "builds" are three distinct
  configs. Historical builds cannot be reconstructed from configs.
- **numpy and JAX must be checked, not assumed.** A wrong conv weight layout
  produced plausible logits, passed every smoke test, and lost 200–0.

Finally: **do not report an improvement.** Report what you changed, what it costs,
what would falsify it, and what still has to be measured. The promotion suite
decides whether anything improved, and it has not been run on any of this.

## Reproducing the Part 1 measurement

The ladder forensics scripts are not yet in the repo. Regenerate the corpus with:

```bash
python -m analysis.official fetch --player "H.V.Nguyen" --out runs/official7 --delay 1
```

`analysis/official.py::to_states` yields `(tick, engine.State)` so official
replays feed the existing tools, and `castle_builds` recovers builds from the +1
even-tick growth signature. Two internal checks say that reconstruction holds: it
reports 0.000 castles on the board at turn 50 (a build needs 35 army, impossible
that early), and the combat residual comes out symmetric to 6%, which it must be
physically and would not be if the structure count were badly wrong. If you
extend the analysis, put the scripts in `analysis/` with the rest.
