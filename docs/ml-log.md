# ML attempts: what was tried, what it measured, why it failed

Running log so nothing gets tried twice. Every entry needs a measured number, not
an impression. If an entry has no number it does not belong here.

Last updated 2026-08-04.

## The one-line summary

Six ML attempts failed. **The seventh is working.** A generals-distance
curriculum took competition-distance win rate against v16 from 0.152 to 0.420 in
300 iterations, where the same PPO without a curriculum managed 0.175 to 0.255
in 1500 and stopped. Reward density was the thing all six shared.

The heuristic has its own ceiling and we are now measuring it. v13 read 1737 over
102 ladder games; v18 — five unconditional mechanism fixes later — read **1587
over 90**, a 3.0σ regression. The "5-for-5 mechanism fixes" claim in earlier
versions of this file was measured on mechanism metrics (general-drain rate,
army brought home), NOT on Elo, and the aggregate is 150 Elo down. Only the
first two or three were ever confirmed by rating.

**The single most transferable finding in this document:** a network that
predicts the heuristic's move **65% of the time wins 1 game in 100 against it**
(198W-0D-2L, +798 Elo). Top-1 accuracy maps onto playing strength so steeply
that it is nearly worthless as a model-selection metric in a sequential game —
and every behaviour-cloning number here was measured with it.

**Two things were settled on 2026-08-04, and they point in opposite directions:**

* **Network capacity is NOT the ceiling** — *this claim is now in doubt, see
  "The receptive field" below.* 70k to 270k parameters, four sizes, all
  0.489-0.499 validation top-1. The study varied width and depth, but BC top-1
  saturates on LABEL NOISE at ~0.50, so it cannot distinguish "big enough" from
  "cannot see the board".
* **Reward density IS the problem, and a curriculum fixes it.** At generals
  distance 2-6 the critic reached explained variance 0.12 in **15 iterations**;
  at the competition's distance 17+ it never cleared 0.12 in **1400**.

---

## Attempts

### 1. Behaviour cloning — `learn/train.py`

Supervised, on ~2000 harvested ladder replays of many players.

**Result:** plays like the median field bot. Wins ~0.175 against `configs/v16.json`.

**Why it stops there:** it is trained to reproduce the average of everyone on the
ladder, so the average is its ceiling. Not a bug — a property. It is still the
right *initialisation* for anything else, and every later attempt that skipped it
never learned to play at all.

### 2. PPO self-play with land-lead shaping — `learn/rl.py`

**Result:** learned nothing measurable.

**Why:** the shaping term was the land lead, and in a mirror match the land lead
is zero-sum. Its expectation over the two seats is *exactly* zero, so the shaped
component of the gradient cancels by construction. This was arithmetic, not bad
luck.

### 3. Frozen-opponent PPO

The fix for (2) — train against a fixed opponent so shaping does not cancel.

**Result:** learned to farm land against that one opponent and got strategically
worse.

**Why:** reward misspecification. Land is easy to accumulate against a fixed
policy and is not what wins games.

### 4. Terminal-reward REINFORCE exploiter

**Result:** collapsed 0.115 → 0.000 in twelve iterations, games getting shorter
as it learned to die faster.

**Why:** no critic baseline. At a 10% win rate the update pushes *down* on 90% of
the policy's own moves every step, with nothing pulling back up.

### 5. Self-imitation exploiter — `learn/exploit.py`

Train only on winning games, so self-suppression is impossible.

**Result:** plateaued at 0.14–0.17 against the target and stayed there.

**Why:** started from random weights over a 3529-action space with a terminal-only
reward. It never learned to play the game at all — 0.15 is roughly what a
near-random policy scores.

### 6. PPO best-response oracle — `learn/netoracle.py` (2026-08-04)

The first attempt with all the pieces: BC init, critic baseline with GAE,
terminal-only reward, KL anchor to the initialisation, illegal actions masked
before the softmax, evaluation on held-out boards.

1500 iterations, 256 games each — 384k training games, ~8 h on 60 workers.

**Result:** collapsed, then recovered, then plateaued below the target.

| iters | eval vs v16 | |
|---|---|---|
| 0 | 0.175 | the clone |
| 160–310 | **0.000** | total collapse, 310 iterations at a 0% win rate |
| 570 | 0.25 | recovered |
| 570–1499 | 0.15–0.32, mean ~0.22 | plateau, 900 iterations of no progress |

Gate needed +0.177 on 400 fresh games. Rejected.

**Settled on 400 fresh games (2026-08-04):**

```
clone:runs/nn/br16.best.npz  vs  ours:configs/v16.json
  102W 0D 298L  score 0.255   elo -186.2 [-227.8, -149.1]
  sprt llr -9.22 -> accept H0 (no improvement)
```

So the best checkpoint is worth **0.255**, not the 0.320 the `<- kept` line
showed — winner's curse, exactly as the plateau mean predicted. Attempt 6 is a
failure with a number on it.

**Why it collapsed:** critic warmup was a fixed **3 iterations**, and `evar`
(explained variance) sat at +0.00…+0.04 through the whole early phase. The policy
was being updated from advantages that were pure critic noise. The likely
consequence is worse than the lost time: the collapse destroyed the BC warm
start, so everything after iteration 320 is PPO relearning from a wrecked policy
— i.e. attempt (5) again. The 0.25 plateau is plausibly *what PPO-from-scratch
reaches here*, not what PPO-from-a-clone reaches.

**What was genuinely new:** the first attempt that recovered from collapse
instead of staying dead. The machinery is right; the schedule is wrong.

**Do not read `train-wr` as progress.** It plays the σ-floored mixture (87.5% v16
plus 12.5% weak opponents the net beats easily) while `eval` plays raw σ = 100%
v16. `train-wr` reached 0.43 while `eval` was 0.22. Only `eval` counts.

**Do not read `<- kept` as a policy's strength.** 0.320 was a max over ~117 evals
at ±0.05 each; the expected max from noise around a 0.22 mean is ~0.35. Winner's
curse.

**Planned fixes, untested:**

1. Freeze the policy until `evar > 0.10`, however many iterations that takes.
2. Lower LR for the first ~200 iterations.
3. Tighter KL anchor early.
4. Tripwire: stop if eval falls 2 SE below the `eval 0` baseline.
5. Rollback to `.best.npz` and halve LR on collapse, instead of continuing
   downhill.

---

## The evaluator problem

Nine local instruments have been built. Every one disagreed with the ladder in
the same direction: they reward committing army aggressively, because no opponent
they could construct punished over-commitment.

| instrument | verdict |
|---|---|
| arena vs hand-written bots | ranked v12 above v13; ladder has them 241 Elo the other way |
| self-play SPRT, 1200 games | approved a −241 Elo change |
| `hunter:3` gauntlet | agreed with the arena, also wrong |
| behaviour clone as opponent | imitates typical play, cannot punish counterfactual commitment |
| field-trained win-probability model | 1.6 pp spread across builds 270 Elo apart, and ranked the worst build first |
| greedy / expander baselines | lose 93–99% to our bot; inert |

**The rule that came out of it:** ship only mechanism-level fixes found by reading
replays. Every weight-level change shipped on local evidence has regressed.

### Why over-commitment was unpunishable — and partly why

`_can_attack` priced an unseen enemy general at a fraction of the opponent's
**global** army total. That total does not move when their army marches away, so
an opponent who empties their base was valued as if fully garrisoned. "Strike
because their army is committed elsewhere" was not expressible by any setting of
any knob. Fixed in v17 (`attack_discount_committed`).

---

## The config league — `learn/league.py`

PSRO over the ~24 commitment knobs. Archive grows by best response, so the signal
is "beats everything found so far" and needs no external yardstick.

**Probe (best response to v12 alone):**

| | vs v12 | vs v16 | gap |
|---|---|---|---|
| evolved punisher | 0.792 ±0.102 | 0.396 ±0.102 | **+0.396 ±0.144** |

2.75σ. A config-space punisher of over-commitment **exists** — and it is *worse*
overall (0.396), i.e. a specialist, which is what a real punisher looks like.
Config at `runs/probe/probe.json`. Worth keeping as an arena opponent.

**League (best response to the archive mixture):** **0.522 ±0.013.**

Exploitability ≈ 0.02. **The config space is exhausted** — there is no Elo left in
the 24 knobs. This retroactively explains the weight-level regressions: they were
not bad luck, there was nothing there to find, and v12 lost 241 Elo by moving
*off* a near-optimal plateau.

**Why the league degenerates:** nothing we can construct beats v16, so the payoff
matrix is transitive and fictitious play returns a point mass. `best response to
the mixture` and `beats one bot` become the same sentence. Watch the
`effective opponents` line — below 2.0 it is frozen-opponent PPO wearing a
league's clothes. Seeding the probe punisher does **not** fix this: v16 still
beats it, so the ordering stays transitive.

---

## Traps found the hard way

**Configs silently become each other.** `Config.from_dict` fills absent keys from
today's defaults, so a config saved before a key existed loads as a different bot.

```
IDENTICAL v9 == v13 == v14 == v15 == v16
IDENTICAL v6 == v7  == default
```

Six ladder-measured "builds" are three distinct configs, and the v14/v15/v16
changes live in code. **Historical builds cannot be reconstructed from configs** —
use `git worktree` at the commit and `stdio:` to play an old build.

**numpy and JAX must be checked, not assumed.** A numpy conv used the wrong weight
layout (im2col lays columns `(kh,kw,cin)`, weights stored `(cout,cin,kh,kw)`),
produced plausible logits, passed every smoke test, and lost 200–0. Max abs diff
was 22.9. There is a regression test now; keep it.

**Equivalence tolerances are hardware-dependent.** That same test then failed on
GPU because JAX uses TF32 for f32 convs on CUDA. Use `Precision.HIGHEST` and a
magnitude-scaled tolerance.

**Elo noise.** SE ≈ 694.9 × 0.5 / √n → ±58 at 36 games, ±41 at 72, ±26 at 180.
Most build comparisons made in this project were never resolvable.

**The arena's SPRT verdict named the wrong side until 2026-08-04.** `llr` is
computed from A's results, so H1 means "A beats B", but the string said
"B is better". Every accepted-H1 verdict this tool printed named the loser.
Nothing was decided on it — the Elo sign is on the line above and that is what
got read — but check the sign, not the words, in any older output.

**`ours` with no config argument is not the current build.** It is `Config()`,
whose dataclass defaults still carry `lock_enabled=True` — the setting measured
at −265 Elo. Benchmarks against bare `ours` are against one of the weakest
configurations we have shipped, not against v16/v18.

---

## Inference budget is not the constraint

Single-core numpy, 21×21 board, 12 input channels, measured:

| arch | params | MACs | ms/move | of 150 ms |
|---|---|---|---|---|
| 4×32 (current) | 33k | 14M | 0.4 | 0% |
| 8×64 | 380k | 117M | 1.2 | 1% |
| 10×128 | 1.9M | 591M | 3.0 | 2% |
| 20×128 | 3.9M | 1242M | 6.3 | 4% |

Even at 10× slower competition hardware, 20×128 fits. **The real constraints are
rollout throughput** (workers run numpy inference every turn; 7.5× slower forward
= 7.5× fewer training games per night) **and BC training data** (~2000 replays
will overfit a multi-million-parameter net).

---

### 7. Curriculum self-play — `learn/selfplay.py` (2026-08-04, running)

The first attempt aimed at the cause the other six share: at the competition's
minimum generals distance of 17 an episode is ~500 turns for ONE bit of reward.

Taken from AverageJoe (same author as our starter kit; 81.5% and #1 on the
generals.io ladder) after reading it — no code copied, it has no licence. What
transferred: the distance curriculum, `adv_top_frac`, `num_epochs 1`, the
terminal-only reward. What did not: its neutral-city magnet branch (we have no
neutral castles), its 17-42 top stage (BFS distance 42 cannot occur on an 18-21
grid at 0.24-0.26 mountain density), and every architecture-coupled
hyperparameter.

**Probe result, 15 iterations at stage 0 (distance 2-6):**

```
iter   0  turns 156  dist 3.9  evar +0.00   [critic warmup 1/20]
iter  10  turns 153  dist 3.7  evar +0.05   [critic warmup 11/20]
iter  15  turns 162  dist 3.8  evar +0.12   <- policy unfroze
comp-eval  0  0.050 +-0.035 (400 games, dist 17+)
stage-eval 0  0.500 +-0.050 (200 games, vs stage-entry self)
```

The hypothesis holds. `evar` crossed 0.10 in **15 iterations** where attempt 6
never crossed it in 1400. Reward density was the binding constraint, not the
critic architecture and not the network size.

**One design assumption was wrong:** stage-0 games run **156 turns, not the ~80**
the design assumed. Still 3x denser than the ~480 at competition distance, but
the throughput projection was 2x optimistic and iteration times followed.

**The promotion gate took three attempts.** Advancing on win rate against v16 is
worthless below stage 4: `bot/belief.py` builds the enemy-general prior as
`passable & (dist >= 17)`, so at distance 4 the true general is not in v16's
candidate set at all and 0.60 measures its miscalibration rather than our
learning. Each stage now evaluates against the policy's OWN stage-entry weights,
both seats played. An unchanged policy reads exactly **0.500** — a fixed point,
invariant to any opponent's weakness at any distance. Confirmed in the probe.

**Scale, stated honestly:** 1200 iterations x 256 games is ~307k games and ~180M
transitions against AverageJoe's ~52M games and 26 billion. **135x short**, with
a 70k-parameter CNN against a 22M-parameter transformer.

### IT TRANSFERS. First working run, from `clone-8x32`

`comp-eval` plays 400 games against `ours:configs/v16.json` at competition
distance 17+ — boards the policy has never trained on.

| iteration | stage | comp-eval vs v16 |
|---|---|---|
| 0 | 0 (2-6) | 0.152 (the clone) |
| 200 | 1 (4-9) | 0.356 ±0.035 |
| 250 | 1 | 0.384 |
| 300 | 1 | **0.420** |
| 330 | → 2 (7-13) | still climbing |

**+0.27 in 300 iterations.** Against attempt 6 — same PPO, same reward, no
curriculum — which went 0.175 → 0.255 in 1500 iterations and then sat flat for
900 more. Three times the gain in a fifth of the iterations, and this one has not
stopped.

I predicted "expect it to move off 0.05, do not expect 0.50". That was wrong by a
wide margin and the prediction is left above deliberately.

**The promotion gate is too strict.** `stage-eval` sat at 0.48-0.58 and never hit
0.60 twice, so stage 2 was entered by `--stage-cap` (forced), not earned. The
policy improves modestly over its own stage-entry snapshot each time and the
compounding shows up in `comp-eval` instead. A self-referential target gets
harder exactly as fast as the policy improves, so 0.60 may be unreachable by
construction. Loosen it — 0.55, or promote on comp-eval trend — but not mid-run.

**What this changes strategically.** The heuristic's parameter space is exhausted
(0.522) and its best measured build is 1737. A trained policy that reaches v16
parity in 50 iterations is the first thing in this project with a path past that.
Throughput is now the binding constraint, not method: ~10k games/hour on CPU
rollouts against a derived ~230k/hour GPU-vectorised.

### The peak was iteration 50, and the run then froze

Reading 200→300 as a rising trend was wrong. The full series:

```
iter   0   0.152  <- kept        the clone
iter  50   0.515  <- kept        parity with v16, in FIFTY iterations
iter 100   0.323                 collapse
iter 150   0.399
iter 200   0.356
iter 250   0.384
iter 300   0.420                 recovering, never regained the peak
```

Then `[critic warmup 211/20]` at iteration 346: the policy had **re-frozen** and
stayed frozen. Stage 2 games run 333 turns against stage 0's 156, `evar` fell
below the re-freeze line, and the kill rule only arms at stage 0
(`not warmed and stage == 0 and ...`), so at stage 2 it can freeze indefinitely.
The run burned ~250 iterations on critic-only updates. **Arm the freeze timeout
at every stage, not just stage 0.**

### The checkpoint is real, and it is NOT a 1737-equivalent bot

`sp.best.npz` confirmed on 400 fresh games: **207W 0D 193L, 0.517, +12 elo**
against v16. Move times 1.3 ms mean, 15.1 ms worst, zero faults.

But head-to-head parity is not a rating. Against a common yardstick:

| opponent | net | heuristic |
|---|---|---|
| v16 | 0.517 (+12) | — |
| greedy | 0.772 (+212) | **0.932 (+455)** |
| hunter:3 | 0.925 (+436) | 0.958 (+520) |

**Measured against greedy the heuristic rates 243 Elo above the net; measured
head-to-head they are level.** Both cannot be a rating. This is the
non-transitivity the league was built to detect: the net matches v16
specifically while dropping 23% of its games to greedy where the heuristic drops
7%. On a ladder you play the whole field, so 0.517 vs v16 does not translate to
~1717.

**The bar for submitting a network is therefore not "beats v16".** It is
**greedy ≥ 0.90 AND v16 ≥ 0.55** — uniform strength across styles, checked before
spending a submission. Head-to-head against one opponent is exactly the evidence
that made v12 look good.

---

## The receptive field — the most likely reason this pivot fails

`bot/policy/net.py` is an L-layer stack of 3x3 convolutions with **no dilation
and no pooling**, so its receptive-field radius is exactly L.

| arch | radius | window |
|---|---|---|
| 4x32 (shipped) | 4 | 9x9 |
| 8x32 (deepest ever trained) | 8 | 17x17 |
| **a 21x21 board, corner to corner** | **20** | — |

Now list what the heuristic actually decides on: `dist_home`, `dist_enemy_gen`,
`dist_enemy_terr`, `dist_unowned`, the distance-ordered cumulative defence sums,
the threat scan, `hidden_enemy_army`, the turn number. **Every one is a
whole-board BFS or a global scalar. None is computable inside a 9x9 window**, and
widening the net at fixed depth adds no reach at all — `4x64` has 122k parameters
and still sees 9x9.

So the scaling study above may have measured the wrong thing. Both readings fit
the data and only one was recorded:

* the model is big enough and the labels are noisy — what the table says;
* the model **cannot see the board**, and top-1 saturates on label noise long
  before that becomes visible.

This is also the most plausible explanation of the non-transitivity: a policy
with radius 8 plays locally-good moves and gets outmanoeuvred on a board it
cannot perceive as a whole.

### RESULT (2026-08-04): top-1 accuracy is a nearly worthless proxy for strength

Distilled 2000 v16-vs-v16 games — 1,750,514 labels, 44 shards, zero label noise
by construction, passes dropped, whole games kept inside a shard so the
validation split holds out whole games. `tools/distil.py`.

**A 4-layer, 32-channel net (radius 4) converged at val top-1 0.651** — flat over
the last five epochs with the learning rate decayed to nothing. Compare 0.499 for
the same architecture on ladder replays, so **~0.15 of the old BC ceiling was
label noise** and the rest was not.

It is UNDERFITTING, not overfitting: train loss 1.0366 against val nll 1.0997,
converging together and both still creeping down at LR≈0. A data-limited model
shows train loss falling while validation stalls. This is the opposite — it
cannot fit even the training set, so the shortfall is representational.

Then the number that matters:

```
ours  vs  clone:/tmp/d4.npz
  198W 0D 2L   score 0.990   elo +798.3
```

**A policy that predicts the heuristic's move 65% of the time wins 1 game in
100 against it** — and `ours` bare is the WEAK default config with
`lock_enabled=True`. Generals is sequential; one wrong move in three compounds
into a lost position.

So every BC top-1 number in this document, including the whole scaling study,
was measuring something with an extremely steep and unknown mapping onto
playing strength. **Do not use top-1 to choose an architecture.**

### And imitation is the wrong task anyway

| policy | trained by | vs the heuristic |
|---|---|---|
| d4 clone, 0.651 top-1 | imitation | **0.010** |
| `sp.best.npz`, 8 layers | self-play RL, 50 iterations | **0.517** |

Both are shallow conv nets with limited receptive fields. The imitator is
unplayable; the RL policy reached parity in fifty iterations.

**RL does not have to reproduce the heuristic's decision function.** It never
needs to compute `dist_enemy_gen`; it only has to win, and it can find a policy
that works within whatever it can see. Imitation is strictly harder because it
must match a function built from whole-board BFS.

So the receptive field is a hard ceiling on COPYING the heuristic and is not
demonstrated to be a ceiling on BEATING it — an 8-layer net already reached
0.517 by self-play. The experiment below answered a real question and, in the
same run, showed that question was less load-bearing than the argument for it
claimed.

**Conclusion: stop imitating, put the compute into RL.**

### The experiment that settles it in one hour

**Distil the heuristic, not the ladder.** v16 is deterministic given its config,
so labelling its own games gives **exactly zero label noise** — which removes the
confound entirely.

1. Play ~300 v16-vs-v16 games at competition distance, dump (observation, v16's
   action) for both seats. ~290k examples, minutes on the arena.
2. Hold out whole games, using the shard split.
3. Train 15 minutes each at `--layers 4 --channels 32` (radius 4) and
   `--layers 12 --channels 128` (radius 12, ~1.9M params — the inference table
   says 20x128 costs 6.3 ms/move, 4% of budget, so depth is affordable).

| result | conclusion |
|---|---|
| L4 stalls ~0.5-0.7, L12 clears ~0.90 | **receptive field is the ceiling.** Go deeper, or add a mean-pool-and-broadcast global path per block, which is far cheaper than depth. Re-cost throughput for a 1-2M-param net. |
| both stall below ~0.7 | **the observation is the ceiling.** No policy over these 12 channels can imitate the heuristic; it needs the belief-derived channels bot/belief.py maintains. |
| both clear ~0.90 | architecture and observation are both fine; the ceiling is training, and the pivot proceeds as planned. |

## Network capacity is not the ceiling — as measured by BC top-1

`tools/scaling.py`, one variable at a time, 1.94M examples from 2787 replays,
3 held-out shards (49,152 positions, ~120 games) instead of the 8192 contiguous
rows (~7 games) the old split used.

| arch | params | best val top-1 | best val nll |
|---|---|---|---|
| 8x32 | 70,569 | 0.499 +-0.039 | **2.1021** |
| 4x64 | 122,441 | 0.489 +-0.041 | 2.2204 |
| 8x64 | 270,153 | 0.494 +-0.044 | 2.1802 |

**Four times the parameters, no gain on either metric.** Depth alone and width
alone both flat. The limit is label noise: several moves in a position are
equally reasonable, different demonstrators pick differently, and no model
resolves that.

Two cautions recorded so the table is not over-read. `8x32` was still improving
at epoch 19 of 20, so its row is a floor rather than a converged measurement —
`--patience 3` never fired because nothing plateaued. And the SE is an
across-shard estimate from only 3 shards, so it is itself noisy.

**The old 0.614 figure is not comparable.** It came from the 7-game validation
set and was inflated by correlation between positions in the same game.

A behaviour clone is also much weaker than the heuristic: `ours` (bare defaults,
which still carry `lock_enabled=True`, the setting that cost 265 Elo) beat
`clone-4x64` **172-28, score 0.860, +315 Elo**. So even against one of our
weakest configurations the clone scores 0.140.

That reframes attempt 6 rather than excusing it: PPO took a clone from ~0.15 to
0.255 against the heuristic — it roughly doubled the win rate — and then stalled
900 iterations short of parity. Directional only; different clones, different
opponents.

---

## The training foundation (2026-08-04) — what it changes and what it does not

No Elo here on purpose: this entry is plumbing, and plumbing is judged by what it
makes possible and by what it can no longer get silently wrong. **Everything
below either has a command that prints it or is labelled an estimate.**

### Three things changed

**1. The action space models castle builds: 3529 -> 3970.** `441*9 + 1`, i.e.
slot 8 of every cell is "build here". It used to be `441*8 + 1` with builds
folded into PASS, which meant two separate failures: the behaviour clone was
trained to PASS on exactly the positions a strong player built a castle on, and
no RL policy could ever emit a build or learn to defend against one, because its
self-play opponent could not build either. Under our ruleset a castle is +0.5
army/turn forever for ONE turn of tempo — `bot/policy/castle.py` calls it the
single biggest economic lever in the ruleset, and the heuristic uses it.

**2. The observation carries the five scalars it used to throw away: 12 -> 20
channels.** `bot/obs.Obs` always carried `turn`, `my_land`, `my_army`,
`opp_land`, `opp_army`; `features.encode` read only the three grids. Now eight
broadcast planes: `turn/1200`, `turn%2` (structures grow on even ticks),
`(turn%50)/50` (all-tile growth), `turn>=800` (deathtouch), and `log1p/6` of both
army totals and both land totals. `features.scalar_features` is the single
definition, called by the numpy encoder with `np.log1p` and by `encode_jax` with
`jnp.log1p`, because two hand-written copies of an eight-element tuple in channel
order is precisely the drift the seam check exists to catch.

Two concrete things the network could not previously see:

* **the clock.** Deathtouch at 800 changes the win condition discontinuously and
  1200 is a DRAW. A critic with no clock cannot tell turn 100 from turn 790 and
  therefore cannot predict "this ends as a draw" — a mechanical candidate reason
  `evar` stalls at 0.12 at competition distance and crosses it in 15 iterations
  at distance 2-6, where games end decisively at ~156 turns.
* **hidden army.** `opp_army` counts the opponent's FOGGED tiles, so
  `opp_army - visible enemy army` is `belief.hidden_enemy_army` — the quantity
  v17's `attack_discount_committed` fix was built on, and by construction not
  computable from the board the network sees.

**3. The rollout is vectorised** (`learn/vecroll.py`) over prebuilt map pools
(`tools/pools.py`), behind `selfplay.py`'s existing produce/train boundary.
`--backend gpu` switches the TRAINING rollout only; comp-eval, stage-eval and the
paired gate stay on the CPU process pool, because they play `arena.agents`
opponents that cannot exist in a JAX kernel and because they are the ground truth
the new path has to be measured against. **`--backend cpu` is still the default
and is unchanged.**

### Measured

| what | number | how |
|---|---|---|
| numpy/JAX seam | 3840 steps, 1374 frames, 9 board shapes, encoder max abs diff **1.2e-07** | `make verify` |
| flat action map | all 3970 indices, 0 mismatches | same |
| build coverage | 661 frames with a legal build, 192 executed, 193 cells, costs 35..109 | same |
| scalar coverage | all 8 broadcast channels varied over the run | same |
| competition mapgen | byte-identical to the previous commit over 2000 maps, 5 seed blocks + 5 curriculum bands | sha256 of int32 bytes + shape, this tree vs a worktree of HEAD |
| `make test` without jax | 19/19, with two explicit SKIPPED lines | import blocker on `jax` |

### Estimated, NOT measured

**~40k env-steps/s -> ~65 games/s -> ~230k games/hour on the L4, against ~2.8
games/s on CPU rollouts today.** This is arithmetic from board size and step
cost, not a benchmark. Nobody has run it. `third_party/generals-bots/tests/
test_performance.py` is the first thing to run on the cluster, and the number
that matters afterwards is the `util` field in the iteration line: the batch runs
until the LONGEST game in it finishes, so short games sit idle and the real
figure is `util` times the peak.

Device memory is arithmetic on measured per-object sizes: ~160 MB resident pool,
~320 MB peak across a stage promotion, tens of MB of per-step buffers. Not close
to 24 GB, so batch width is not memory-bound.

### Four correlated-map and recompilation traps, confirmed in the starter kit source

Not opinions about the library — line numbers, all still true of
`third_party/generals-bots` as vendored:

* `pool_idx=jnp.int32(0)` for every state (`core/game.py:139`) and it increments
  identically (`core/env.py:339`), so **512 parallel envs replay ONE env's worth
  of boards** after the first episode boundary.
* `init_state` hardcodes `self.max_grid_size` for BOTH dimensions
  (`core/env.py:275,289`), so every vectorised env is 21x21 and **the 18-21
  variable-size competition distribution never appears in training at all.** This
  one is invisible in a diversity count.
* `mode="competition"` sets `min_generals_distance=17` authoritatively
  (`core/env.py:155`) and its preset never sets `max` (`core/env.py:59-84`), so
  `GeneralsEnv(mode="competition", max_generals_distance=6)` is min=17/max=6,
  which empties the candidate set and drops into a `farthest` fallback
  (`core/grid.py:356-358`) **with no error.** A curriculum built the obvious way
  gets silently wrong maps.
* both distance args are `static_argnames` (`core/grid.py:142-143`), so K
  curriculum bands are K sets of compiled kernels.

`tools/pools.py` + `learn/vecroll.py` avoid all four by construction rather than
by fixing them: we index the pool ourselves so `pool_idx` is never read, the
pool carries the true per-board `h`/`w`, `_MODE_PRESETS` is never constructed,
and every stage is the same `(N,21,21)` array so promotion is an argument change
and not a retrace.

### KNOWN LIMITATIONS — not fixed, and none of them is cheap

Recorded here rather than left unstated. The first is the one to attack next.

**1. The receptive field is a hard ceiling, and it is arithmetic.** `bot/policy/
net.py` is a plain 3x3 stack — no dilation, no pooling, no downsampling — and
only the PASS logit gets a global mean. An L-layer 3x3 stack has receptive-field
radius exactly L: the default 4 layers see a 9x9 window on a 21x21 board, and the
deepest net ever trained here (8x64) sees 17x17. Corner to corner needs 20.

Consequences that are not opinions: the build surcharge has radius 6
(`rules.SURCHARGE_RADIUS`), so a 4-layer net cannot compute a castle's price — it
is saved only by the legal mask already encoding affordability, and it still
cannot prefer a flat-35 site over a 47 one. Every global field the heuristic runs
on (`dist_home`, `dist_enemy_gen`, `dist_unowned`, the threat scan, the
cumulative defense arithmetic) is a whole-board BFS. **The policy can see a fight
but not where its general is if the fight is 10 tiles away.**

"Capacity is not the ceiling" above does NOT contradict this and does not cover
it. That table is BC val top-1, which the same section says is label-noise
limited at 0.489-0.499 — an instrument saturated by label noise cannot
distinguish a net with enough capacity from a net that cannot see the board. The
cheap next step is mean-pool-and-broadcast per trunk block so the move head gets
the global context the pass head already gets; it changes checkpoint topology, so
it is a measured experiment and not a foundation change.

**2. There is no history and no memory.** One frame, no recurrence. `bot/belief.py`
maintains `mem_owner`/`mem_army`/`mem_turn` (what a tile looked like when last
seen and how stale that is), `ever_seen`, `ever_enemy`, `enemy_castles`,
`candidates`. None of it is in the observation and none is recoverable from a
single frame. Two specifics:

* **enemy castles through fog are literally invisible.** The wire protocol reports
  a fogged mountain and a fogged castle with the same code 5, and the encoder maps
  `T_STRUCTURE_IN_FOG` into both FOG and MOUNTAIN. The belief separates them only
  because it snapshotted the terrain on turn 1 (competition strips neutral
  castles, so every turn-1 type-5 is a mountain). A single-frame policy cannot do
  this at all, and the heuristic scores capturing them (`w.cap_castle`).
* **the enemy-general prior.** `belief.candidates` inverts the generator's own
  seating rule (BFS distance >= 17, room within tolerance) and pins the general to
  a few dozen cells before first contact. Not in the observation and not local.

**3. Stage 0 does not merely fail to teach the castle economy — it teaches its
negation.** At generals distance 2-6 every siting rule in `castle.py` is
unsatisfiable by construction: `castle_safe_dist` wants the rear, and at distance
2-6 there is no rear. Builds are AFFORDABLE there (a general reaches 35 army at
turn 68 and stage-0 games measure 156 turns), so a self-play policy will
correctly learn that spending 35 army leaves a snipeable 0-army tile next to the
opponent — the exact inverse of the stage-5 truth. The KL anchor and
`MIN_RESIDENCY` then carry that prior forward. Stages 1-3 are also too tight for a
rear. **Watch the new `bld` field** (builds per game, both seats) in the iteration
line: it is the only thing that reports whether the 3529 -> 3970 change bought
anything, and a run that sits at `bld 0.00` at stage 4-5 has an action it never
takes.

**4. The build credit path is thin, and that is quantitative.** Reward is
terminal-only (attempts 2 and 3 killed shaping). A castle pays ~+200 army over the
remaining ~400 turns of a stage-5 game, maybe a 3-5pp win-probability shift, ~0.1
in z units against a critic residual std of ~0.89 at evar 0.12. That is ~0.11
signal-to-noise per build decision: ~80 build samples for a 1-sigma read and
~2000 for a confident one. Affordable at 256 games/iteration only if builds
actually occur — see `bld` again.

**5. Multi-turn plans are not expressible as one action, and never were for
anyone.** `castle.plan` returns a `site=` with no action when it wants to walk
army to a build site over 20-40 turns before paying 35; the thrust and the
corridor-hold are likewise sequences. These are policies over the action space,
not missing actions.

**6. Value shards are not meta-checked.** `learn/train.shard_list` refuses a
policy dataset whose `meta.json` disagrees on `n_actions` or `channels` — those
are wrong LABELS, which train perfectly cleanly. `learn/valuetrain.py` globs its
shards directly, but a stale-width value shard dies on a conv shape error, which
is loud, so it was left alone.

### What still has to be true for any of this to matter

The seam check is the deliverable, not a checkbox. `make verify` runs it and
`make test` runs it. If it ever prints a COVERAGE line with a zero in it, the
equality it asserts has gone vacuous and the run that follows proves nothing.

## Dead ends closed by measurement

**Territory churn is not a scoring bug.** A field analysis put our tiles changing
hands ~78 times in wins and ~288 in losses and concluded we take ground we cannot
hold. `analysis/churn.py` over 200 games says otherwise — in BOTH wins and losses
the tiles that flipped were born with MORE army than the tiles that held:

| | flips/100t | young% | fronts | birth army flipped | held | ratio |
|---|---|---|---|---|---|---|
| win (34) | 29.2 | 31 | 8.4 | 7.0 | 3.0 | 2.33 |
| loss (165) | 48.6 | 14 | 6.7 | 3.0 | 2.0 | 1.50 |

Caveat worth stating: birth army is partly a proxy for *contestedness*, not for
garrison adequacy — we spend more army taking frontier tiles, and frontier tiles
are the ones that flip. So the metric cannot fully separate the hypotheses. But
the direction is unambiguous and it is the opposite of the claim, and `young%` is
LOWER in losses (14% vs 31%), i.e. the tiles we lose are ones we held longer.
That is a front being fought over and lost, not tiles rented for a turn.

**"Avoid multiple weak fronts" is dead too.** We hold MORE separate contested
borders in wins (8.4) than in losses (6.7).

What survives is churn per turn — 48.6 vs 29.2 — which is a symptom of losing a
sustained fight. That is the army-economy problem, and it agrees with deathwatch:
we die to stacks two to three times our garrison.

## What has actually worked

Mechanism fixes found by reading replays. Five, all held:

1. castle-gather stall
2. `defense_within` counted army it could never bring home in time
3. the thrust launched from the general
4. the move scorer drained the general too
5. the general was counted as our own strike stack, so gathering triggered ATTACK
   and the base marched off in one move

The loop: harvest replays → `analysis/deathwatch.py` and `analysis/churn.py` →
name the mechanism → fix it → check the mechanism moved. Mechanism numbers are
readable at 12 games. Elo is not.
