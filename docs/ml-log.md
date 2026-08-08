# ML attempts: what was tried, what it measured, why it failed

Running log so nothing gets tried twice. Every entry needs a measured number, not
an impression. If an entry has no number it does not belong here.

Last updated 2026-08-08.

**For the current state and what to do next, read `docs/STATE.md`.** This file is
the full measured history and the reasoning; that one is the short version.

## SOLVED 2026-08-08: self-play does NOT unlearn castles. The critic did.

This file has said since 2026-08-06 that "self-play unlearns castles at every
stage". That was measured with a critic that prices a castle at **-0.002**, and
it was the critic's verdict, not the game's. Given a critic that prices them
correctly, self-play LEARNS to build:

```
bld 0.60 -> 2.29 over ~23 training iterations, bldA positive throughout
```

Four things were needed and removing any one refreezes the run. They are listed
in the order they were discovered, because each only became visible once the
previous was fixed.

### 1. What a castle is actually worth, by stratum

`tools/cfprobe.py` forks a live game at a legal build, forces the build in one
branch, re-rolls a control from the same snapshot under common random numbers,
and plays both to terminal. 3000 games on `sp44.best`, stage 4, 322 s on 60
cores:

| stratum | n | mean delta | SE | vs 0 |
|---|---|---|---|---|
| A -- a build v16's castle block would take | 1794 | **+0.117** | 0.029 | +4.0 sigma |
| B -- every other build-legal turn-seat | 2967 | **-0.129** | 0.023 | -5.6 sigma |

A - B = 0.246 +- 0.037, **+6.6 sigma**. Castles are not good or bad; they are
good where a competent player builds them and bad everywhere else, and ~91% of
build-legal moments are the bad kind. That is why `tools/buildbias` alone failed
on 2026-08-07 and why the correction "castles are a symptom" was half right.

Nothing in this repo had measured this. Every previous castle number routes
through the critic -- `vprobe` reads its belief, `bldA` is its GAE advantage,
the buildbias verdict is an inference from what PPO did next. A forced
unilateral deviation played to terminal is the one estimate that does not.

### 2. The critic was starved, not broken

`tools/vprobe` on the sp43/sp44 pair, against the rule it pre-registered at
`tools/vprobe.py:22`:

```
castle, free          -0.0023 +-0.0014   the asset is not priced
castle, minus 35 army -0.0051 +-0.0018   a build reads as pure cost
garrison 25 vs 2      +0.2610 +-0.0388   the critic works fine on other things
```

True value +0.117, critic's value -0.005. GAE computes a build's advantage as
V(after) - V(before), so **every build gets a negative advantage automatically**,
whatever it did. That is `bldA -1.2` explained, and it is the complete mechanism
behind castles being deleted at every stage.

`STATE.md:1082` records sp9's critic at **+0.28** and concluded "the blind-critic
diagnosis is dead". That was an 8x32 trained where its own policy built 1.5
castles a game. **Critic readings do not transfer across lineages** (see also
547da6e) -- probe the pair you are about to use.

### 3. `--init-critic` was loading every pretrained critic wrong, twice

`learn/valuetrain.py` fits `forward()` under `logaddexp(0, l) - y*l`, so its
output is a SIGMOID logit. `learn/selfplay.py` reads the same function through
`tanh(l)` against targets in {-1, 0, +1}. The conversion is

    2*P - 1 = 2*sigma(l) - 1 = tanh(l / 2)

so the head must be **halved** -- a units bug, fixed in `load_critic`. And that
is only half of it: halving assumes temperature 1, and a BCE fit is
overconfident. `tools/calibtemp.py` sweeps the head scale and scores each with
`evar`, the metric the trainer is actually judged by:

| T | evar |
|---|---|
| 1.0 (raw) | **-0.054** |
| 2.0 (the units fix alone) | +0.068 |
| **4.0** | **+0.117** |
| 8.0 | +0.104 |

Note the failure is NOT bias: `Var(z - c) == Var(z)`, so a constant offset cannot
move `evar`. Expanding, `evar = (2*Cov(z,V) - Var(V)) / Var(z)`, so it goes
negative exactly when `Var(V) > 2*Cov(z,V)` -- a critic swinging harder than its
correlation earns.

**This plausibly invalidates 547da6e, "value24 does not transfer"** -- 0.759 BCE
on field positions buying negative evar on self-play returns. The fit may have
been sound and the scaling threw it away, which would mean the whole
`learn/valuedata.py` pipeline was retired on the strength of a units bug.

### 4. `--critic-lr 1e-3` memorises the buffer

The default destroys a critic that arrives already fitted. Within one iteration:

| `--critic-lr` | `v` at iter 3 | `evar` at iter 3 | `diff` |
|---|---|---|---|
| 1e-3 (default) | 0.154 | **-0.35** | -0.48 |
| 1e-4 | 0.797 | **+0.06** | -0.10 |

An honest MSE at `evar ~0.1` is ~0.88. Reaching 0.15 means it fitted the buffer,
and 226k samples are only ~512 independent trajectories -- an 8x64 net can learn
which board it is on and recall the outcome. Read `v` against `1 - evar`: if `v`
is far below it, the critic is memorising.

### What did NOT work, so nobody builds it again

**The counterfactual aux losses added nothing over plain `buildbias`.** A
competitor ("bca") reported that a counterfactual replay buffer plus a
build-preference loss and a successor-value loss broke their model out of a
no-build mode. Implemented as `--cf-frac/--cf-pref-weight/--cf-value-weight` in
`learn/selfplay.py` and measured as a paired run:

* sp72 (both weights 0.05) and sp73 (both 0) recovered `bldA` from -1.15 to ~0
  **identically**, and sp73 -- without the terms -- reached +0.60/+0.56/+0.58
  where sp72 reached +0.24/+0.15/+0.27.
* On sp70, at `bld 0.00`, the terms had nothing to act on at all: sp44 samples
  `p(build|legal) ~ 0.003`, so `L_pref` can only reweight actions never taken.
  `tools/buildprior.py` records the same shape on sp46, "50 iterations with
  `bldA +0.00` on every single one".

Once builds occur, ordinary play teaches the critic what they are worth. The
machinery is redundant; `tools/buildbias --bias 6.0` plus a correctly loaded
critic is the whole recipe. The flags remain, defaulted to 0, and
`docs/design-counterfactual-build.md` records the design and its red team.

Two further corrections from that work, both worth keeping:

* The gate in that design conditioned on stratum A, but the trainer never sees A
  in isolation -- its coin only fires when both reservoirs are populated, so the
  realised buffer is ~31% A / 69% B and its mean is **-0.018 +- 0.015**. Gate on
  the quantity the loss actually multiplies, not on a sub-population.
* `cfprobe`'s `capt` column is a LOSS rate, not a capture rate: it reads the
  terminal state, so on a decided game it is true iff the fork seat lost.

### The limiter, and what is still open

An offline critic goes stale as the policy moves. Fitted at `bld 0.6`, it bought
16 productive iterations; the policy reached `bld 2.0`, the critic went
off-distribution and refroze. It recovered once after 19 iterations and then did
not recover in 40. So the loop is **policy iteration**: fit, train ~20, refit.

Also true and unresolved:

* **`comp-eval` at 0.887 cannot see this.** One sigma is +-60 Elo there. It read
  0.876 -> 0.887 over 50 iterations, 0.3 sigma, while `bld` more than tripled.
  Use `--opp` with a champion clone so a reading lands near 0.5.
* **`.best` is therefore close to a lottery** -- the max of ~24 noisy draws is
  the luckiest, not the strongest. Snapshot on a timer and arena several.
* **It is all learned in a mirror.** Both seats build, so neither is punished for
  the tempo. The castle rate may be calibrated against itself and wrong against
  an aggressor. `STATE.md`'s "opponent in a TRAINING seat" is still never tried
  and is now the highest-value item.

## The one-line summary

**The neural bot reached 1849 Elo, rank 21/86 — the best result in the project,
and 112 Elo above the best heuristic build.** Seven ML attempts failed first; the
difference was a generals-distance curriculum, which made the reward dense enough
for the critic to learn.

```
neural (sp3, iter ~250)   1849   rank 21/86   27W-8L-1D   n=36
  opponent-controlled     1878 +-70  (mean opponent 1755)
v13 heuristic             1737   n=102   ->  +112, 1.7 sigma
v18 heuristic             1587   n=90    ->  +262, 3.8 sigma
```

Local numbers at that checkpoint: **0.705 against v16** at competition distance
(+151 Elo) and 0.843 against greedy.

### And a SINGLE STRONG opponent lies too — castles, 2026-08-05

`bot/policy/net.py`'s head is one 3x3 conv emitting `PER_CELL` channels, so the
build slot's bias is a single scalar shared by every cell. Adding **+9.094** to
`head_b[8]` took the best net from ~0 castles a game to **~15**, with no
retraining. Measured against v16 on 400 identical boards it looked like a large
free win:

```
sp3.best              293W-107L   0.733   (+175 elo)
sp3.best + build bias 323W- 77L   0.807   (+249 elo)
```

**It was a matchup artifact.** Three other measurements disagreed and the ladder
confirmed them — the rank DROPPED and the build bias was removed.

| test | verdict |
|---|---|
| **ladder** | **rank dropped** |
| builder vs non-builder, head-to-head, identical weights bar one scalar | −36 elo |
| PPO self-play at stage 3 | deleted them: `bld` 15 → 0.1 in 37 iterations |
| vs v16 | +74 elo ← the outlier |

**The rule this document gave after the greedy failure was too weak.** It said
compare head-to-head against the STRONGEST available opponent. v16 *was* the
strongest fixed opponent and it still misled — it builds ~0.9 castles a game
itself, so over-building beats its particular style without being generally
good. The correct rule: **no single fixed opponent measures general strength,
however strong it is.** Only a diverse field does, and locally that means an
archive, not one bot.

Two further notes worth keeping:

* **Self-play is structurally blind to symmetric strategies.** If both sides
  build, neither gains, so the gradient sees the immediate 35-army cost and no
  benefit. Same shape as attempt 2, where land shaping had expectation exactly
  zero in a mirror. PPO's rejection of castles is therefore weak evidence on its
  own — it was right here, but it would reject a genuinely good symmetric
  strategy too.
* **Only the extremes were tested.** 0 castles and 15. The heuristic builds
  ~0.9 and gains +111 Elo from it (`castle_enabled=False`: 262W-138L, 400
  games), so an optimum plausibly sits between and remains unmeasured.

**WHY v16 MISLED, precisely.** A castle repays over ~70 turns, so its value is a
function of how long the game lasts — and that is a property of the OPPONENT.
v16 farms and plays slowly, so games against it run 480+ turns and 15 castles
have time to compound. Ladder opponents at 1800+ press hard and games end early;
army spent on a castle at turn 150 never returns, and you are 35 down in the
fight that decides the game.

So the finding is not "castles are bad". It is **castles are a bet on the game
lasting**, and against strong opponents that bet loses. This also explains why
the heuristic's 0.9 per game beats both 0 and 15: `castle_gather_min_army: 400`
means it never walks army to a site, it only builds when a stack is ALREADY
standing on one. That is an opportunistic bet with almost no downside, not a
planned investment.

The implication for the network is that a constant bias is the wrong shape for
this decision entirely. `head_b[8]` shifts every cell on every turn identically;
what the decision needs is context — how long is this game likely to last, am I
under pressure, is this site safe. That information exists (`rules.build_cost_grid`
is computable per cell, threat is derivable) and none of it is in the 20
observation channels. See the ceiling discussion above: information the net
cannot compute has been worth more in this project than capacity or schedule.

### The greedy yardstick was the tenth instrument to lie

Before submitting, the two local measures disagreed by 315 Elo:

| measured by | verdict |
|---|---|
| head-to-head vs v16 | net is **+151 above** the heuristic |
| the greedy yardstick | net is **164 below** it (0.843 vs the heuristic's 0.932) |

The ladder says head-to-head was right. **Compare against the STRONGEST available
opponent head-to-head; do not weight scores against weak baselines.** greedy and
hunter are inert — you beat them either way and the margin carries no
information. This document previously recommended a "greedy >= 0.90 AND
v16 >= 0.55" submission bar; the greedy half of that bar would have blocked a
1849-Elo bot.

---

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
transferred: the distance curriculum, `num_epochs 1`, the terminal-only reward.
**Correction (2026-08-05): `adv_top_frac` was NOT taken** — `learn/selfplay.py:52`
lists it under NOT, because positive-only learning is attempt 5's design. An
earlier version of this line claimed it transferred and was wrong. What did not: its neutral-city magnet branch (we have no
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

### The 23x was wrong by 13x, and the reason is the whole story (2026-08-04)

The paragraph above derived ~230k games/hour against ~10k on CPU. **Measured on
the cluster, on an L4, `learn/selfplay.py --backend gpu`:**

| games | time/iter | s/game | GPU-Util | vs CPU |
|---|---|---|---|---|
| 256 | 10 s | 0.039 | 0.21 | 1.7x |
| 2048 | 128 s | 0.0625 | 0.11 | ~1.0x |
| 256, `--backend cpu` | 17 s | 0.066 | — | 1.0x |

**A derived 23x measured 1.7x, and it got WORSE as the batch grew.** Utilisation
falls with batch size, which is the signature: the device is idle ~80% of the
time and bigger batches buy marginally better kernels while adding proportionally
more serial host bookkeeping.

**Diagnosis: host-bound, not compute-bound.** Python sat inside the inner loop —
per turn it gathered states, called JAX, decoded actions, stepped the env,
checked terminations. A stage-0 game is ~156 turns and a competition game ~480,
so an iteration was hundreds of sequential host round trips each handing the GPU
~1 ms of work and then making it wait. Secondary: the net is 73k parameters
(a 3x3x32x32 conv on 21x21 is a ~32x288x441 matmul), far too small to occupy an
L4 even while a kernel IS resident — and `nvidia-smi`'s GPU-Util is "time with
any kernel resident", not capacity, so 0.21 overstates the true FLOP fraction.

**What changed: `--backend scan`.** The rollout now runs inside `jax.lax.scan`,
`--scan-chunk` turns per device call. At chunk 40, stage 0: kernel launches
160 -> 4, blocking host calls 645 -> 20. `--backend cpu` is still the default and
`--backend gpu` is unchanged and still supported; the scan is a third choice.

**Not bit-identical, and that is measured, not conceded.** A scan body is a
different XLA compilation from a standalone `@jax.jit`, so the conv trunk fuses
differently: logits move 3.6e-07 and `logp` by 1 ulp (4.8e-07 over 8 games x 480
turns, XLA:CPU). Observations, legal masks, sampled actions, lengths and outcomes
are exact. `logp` is the PPO ratio denominator and 5e-07 of relative error in it
is orders below any gradient. The gate
(`tests/test_all.py::test_the_scanned_rollout_matches_the_python_loop`) is exact
on everything that decides a game and bounded on `logp` —
**and its horizon is 480 turns because at 160 the two paths ARE bitwise equal**,
i.e. the first version of that test passed for a reason that had nothing to do
with the property it claimed.

**PROJECTION, not a measurement: 2.5-4x at 256 games, i.e. 0.010-0.016 s/game.**
**REFUTED — it measured 0.7x. See the next section; the paragraph is kept as
written so the projection can be read against what happened.**
Nobody has run it. It rests on two load-bearing guesses: that the ~2.1 s of
kernel-resident time in today's 10 s iteration is unchanged by the scan, and that
the ~1.8 s of host memcpy per iteration (`np.concatenate`, `_pack_columns`'s
strided per-column gather, the ingest copy, the minibatch fancy-index gathers)
is untouched by it — because it is. **The derived ~40k env-steps/s ceiling
remains unreachable by this path at all**, because every sample still makes a
full device -> host -> device round trip; AverageJoe, where that number came
from, never leaves the device. Treat 23x as a lesson, not a target.

**What falsifies it:** if `--backend scan` at 256 games reads worse than
~7 s/iteration, the bottleneck was never the round trips and the remaining cost
is the host memcpy path and the D2H copy, neither of which this touches. If
GPU-Util still reads ~0.2 at 256 games while s/game improves, the gaps closed but
the kernels are the next problem and the answer is a bigger net, not a faster
loop.

### Throughput: every projection this repo has made, and what it measured (2026-08-04)

One table, kept in one place, because the pattern is the finding.

| change | projected | measured, 256 games stage 0 | s/game | error |
|---|---|---|---|---|
| `--backend cpu` (baseline) | — | 17 s/iter | 0.066 | — |
| `--backend gpu` | **23x** | 10 s/iter | 0.039 | **13x optimistic** |
| `--backend scan` | **2.5-4x** | ~19 s/iter | ~0.075 | **SLOWER than gpu** |
| this commit (instrumentation) | **~1.0-1.1x, and it is not a speedup** | not yet run | — | — |

The scan figure is not quite like-for-like: it was taken on a 12x32 net, ~1.5x
the arithmetic of the 8x32 the other two rows used, so scan is about **1.4x worse
than `gpu`** corrected rather than 1.9x. Either way it is not faster, and
`--backend gpu` remains the fastest of the three. GPU-Util reads 0.14-0.39 in
**all three** backends: the device is idle most of the time in every one of them.

**What the scan did fix, and what it exposed.** It removed the per-turn host
round trips exactly as designed — blocking host calls 645 -> 20, launches
160 -> 4. That was the diagnosed problem and the fix was correct. It did not
touch the next constraint: the **device-to-host copy**. Per chunk `vecroll` calls
`np.asarray` on the stacked observations and blocks; at 256 games x ~130 turns
that is ~1.5 GB moved off the device with the GPU idle for the whole transfer,
then moved back for the update.

**Two projections, two misses, and the same mechanism behind both: the number was
derived from the term someone had already decided was the bottleneck.** So this
round derives nothing and measures first. This commit adds no backend. It adds
the instrument that tells you whether a backend is worth writing:

* `roll/d2h pack ing trn` on every iteration line. Four phases plus, inside
  `roll`, the copy split off from compute by `vecroll._to_host`
  (`block_until_ready` then time the `np.asarray` — it waits for a computation
  `np.asarray` waited for anyway, so it costs nothing and moves no boundary).
* `--epochs 0` as the cross-check that needs no trust in a timer placement.
  Read its `--help`: the obvious recipe measures the wrong thing twice over.

**PROJECTION, labelled, for the device-resident buffer this is measuring for —
and it is UNDER 1.5x by this repo's own arithmetic.** Ceiling is Amdahl on the
transfer share: this file already puts host memcpy at ~1.8 s and the PCIe legs at
~0.5 s of a 10 s iteration, so removing the round trip **completely** projects
`10 / (10 - 2.3)` = **1.3x, range 1.2-1.4x**. That is worth roughly one third of
what `--backend gpu` already delivered, for a rewrite of the buffer contract.
**Do not build it unless `d2h` comes back much larger than that arithmetic
predicts.** The `d2h` field is the falsifier and it is now printed: read
`roll 7.0/d2h 0.3` and the copy is 4% of the rollout and the whole hypothesis is
dead; read `roll 7.0/d2h 3.5` and the arithmetic above was wrong and it is worth
building. There is no third reading, and no number should be written here before
that one is.

**If it does get built, form B (store STATES, recompute the 20-channel encoding
in the training pass) over form A (keep the stacked observations on device), and
the reason is memory, not speed.** Both have the same transfer ceiling above.
A `GameState` is ~4.9 kB a game = ~2.4 kB a sample against the observation's
17.6 kB, i.e. **7.2x smaller** — note that is the achievable reduction; the
"observations are 97% of the payload" figure is the payload *share* and is not
the same number. It matters at stage 5: form A must size the padded buffer for
`max_turns` 1200 because a preallocated device buffer cannot break early, giving
11.2 GB of buffer plus a gathered training copy against an **18.4 GB** effective
pool (`XLA_PYTHON_CLIENT_MEM_FRACTION` defaults to 0.75 of the L4's 24 GB) —
it OOMs at 256 games, and the cliff sits between stage 2 and stage 5, i.e.
exactly where the current run is heading. Form B is ~4.5 GB there. Whichever is
built needs a startup check against `memory_stats()["bytes_limit"]` that raises,
not warns: the equivalent host-RAM failure was found as a `MemoryError` minutes
into an iteration.

**bf16 is still unused and still untried.** The L4 has bf16 tensor cores; the
rollout forwards run f32 convs at 32 channels and never touch them. It is
independent of everything above. Any dtype change has to be read against `dnp`
(kill at `1e-4 x scale`, healthy 1e-4 to 1e-3, seam measures 1.2e-07) — and
unlike f16 -> f32, **bf16 is lossy**, so that check is load-bearing rather than a
formality.

**AND THE PRIOR QUESTION, which no speedup answers.** comp-eval has been flat at
~0.63 for 240 iterations at stage 2. If the ceiling is the method, a 3x speedup
buys three times as many iterations of the same plateau. The receptive-field
limit under KNOWN LIMITATIONS is the standing candidate for that ceiling and it
is not a throughput problem. Spend the `d2h` measurement — it is one iteration —
before spending anything else here.

**Host RAM, not device RAM, is what OOMs this.** The rollout buffer is
`steps x 2*games x 18.2 kB`: 10.8 GB at 256 games / 1200 turns, 34.7 GB at 2048
games / 480 turns. Both paths used to hold the per-chunk list AND the joined
array at once, doubling that to 21.7 GB on a box also hosting 60 workers, found
as a numpy `MemoryError` minutes into an iteration. `vecroll._join` frees each
part as it copies; peak is now the result plus one part. Device memory is not the
constraint at any batch anyone would type — chunk 40 fits ~5000 games on the L4.

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

---

## The critic as the bottleneck: the diagnosis, the instrument, and a PROJECTION

**Nothing in this section is measured yet.** It is written before the run so the
prediction can be read against the outcome, the way the 23x and the 2.5-4x
throughput projections above were not.

### The diagnosis

With terminal-only reward and gamma=1, a move's advantage is essentially
`V(s_next) - V(s)`. That difference is the ONLY mechanism connecting a move at
turn 100 to a win at turn 450. Our critic reports explained variance 0.20-0.35,
so most of that difference is noise, and credit assignment degrades toward
"every move in a won game gets upvoted" — which cannot separate the good moves
from the bad ones inside the same game.

Two observed failures this predicts, both already in this document:

1. **Castles.** A castle costs 35 army now and returns +0.5 army/turn for the
   rest of the game, roughly a 70-turn payback. Given a build prior, PPO deleted
   them within 37 iterations (`bld` 15 -> 0.1) and held there. A weak critic sees
   the immediate cost and not the diffuse benefit.
2. **Midgame collapse.** Over 336 ladder games, mean land at turn 100 is **48.8
   for us against 45.4** for the opponent; by turn 200 it is **69.0 against
   75.2**. We win the opening and lose the midgame, and whatever goes wrong there
   has its consequence 50+ turns later — the attribution a weak critic cannot
   make.

### What the instrument measures, and what it deliberately does not

`learn/selfplay.py` now prints `evar` in bands with a control:

```
evar +0.31 (e+0.04 m+0.22 l+0.88 sc m+0.18 l+0.80)
```

`e` is the first 10% of each episode's plies, `l` the last 10%, `m` frac
0.20-0.45 — the midgame window the collapse lives in, which the first version of
this instrument did not print at all.

**No absolute number here is a claim, and that is the point.** V* is a
martingale under terminal-only reward, so every band carries a ceiling set by how
fast games resolve and how often a losing seat is sniped through fog. Simulated
over a bounded win-probability martingale with z drawn consistently with V* — i.e.
a PERFECT critic — the last-decile evar reads anywhere from **0.29 to 1.00**
across plausible operating points, and 0.80 at a plausible one. An earlier draft
of this work put a `l >= 0.85` bar on that field. **That is the stage-eval 0.60
mistake for the third time in this project** (see also the greedy >= 0.90
submission bar, which would have blocked a 1849-Elo bot): a threshold that may be
unreachable by construction.

`sc` is the fix and it is the only readable number. It is the same evar for the
same subset, from an ordinary least squares on the **eight broadcast scalar
planes the observation already carries** (`turn/1200`, `turn%2`, `(turn%50)/50`,
`turn>=800`, `log1p` of both army and both land totals) — a predictor that cannot
see the board at all. Both predictors face the identical ceiling, so **the
ceiling cancels and only the gap means anything**:

| reading | conclusion |
|---|---|
| `m`/`l` well above `sc` | the critic reads the board; a low absolute number is information the state does not contain, and no value loss recovers it |
| `m`/`l` at or below `sc` | the critic extracts nothing the clock does not already give — the only reading that justifies touching the value loss |
| `nan` | Var(z) = 0, an all-draw batch. Not a critic result |

Cost: two `lstsq` calls on an (n, 9) design, **7 ms at n = 71k**, no games.

### The fix under test

An HL-Gauss distributional value head, `--value-head hlgauss`, **default off**.
128 bins over [-1, 1], sigma 0.04, cross entropy — AverageJoe's numbers, and our
reward is already terminal +-1 with gamma=1 so the range matches exactly.

Two things about it are worth recording because they cut against the framing that
motivated it:

* **It is not distributional here.** Our returns have support exactly
  {-1, 0, +1}, so the three label rows occupy 13/26/13 of the 128 bins with
  **exactly zero overlap mass in float32**. It is a 3-way classifier with fixed
  label smoothing. It buys no distributional information.
* **The one real mechanism is narrow.** What remains is killing the `(1 - tanh^2)`
  factor in the MSE gradient, which is a function of |V| alone: 1.00 at |V|=0,
  0.19 at 0.90, 0.0199 at 0.99. It is severe only where |V| is large — the last
  decile — and ~1.0 in the midgame, where the diagnosis says the disease is. **The
  mechanism and the diagnosis do not overlap.** The castle decision in particular
  sits at turn 100-200 with |V| ~ 0, so this change does nothing to the ~0.11
  signal-to-noise per build decision computed under KNOWN LIMITATIONS 4.

One arithmetic trap found and fixed before the run: at z = +-1 the Gaussian is
centred on the range edge, so half its mass is truncated and renormalised away
and the CE optimum is **kappa * V\*** with kappa = 0.9677, not V\*. GAE is NOT
invariant to that — the terminal delta is `z - V_{T-1}` with z unscaled, so
`adv_t(kappa V) = kappa adv_t(V) + (1-kappa) lam^(T-1-t) z`, a z-signed residue
that std-normalisation cannot remove (measured: the terminal advantage runs 1.33x
the scalar head's). `values_of_hl` divides kappa out, which restores
`adv(V_hl) == adv(V_scalar)` to f32 round-off and makes evar comparable between
the two heads — without which **the change would have been unmeasurable, which is
worse than not making it.**

### PROJECTION — labelled, and expected to be a rejection

| field | projection |
|---|---|
| `l` | +0.05 to +0.20 |
| `m`, `e`, and the `sc` gaps | ~0 |
| comp-eval / Elo | **-20 to +30, centred on ZERO** |
| castles (`bld`) | no effect |

**What would falsify it:** `l` rises >= +0.10 while `m` and comp-eval stay flat
over 300 iterations. **That is the outcome expected, and it must be written up
here as a rejection, not a win** — otherwise this becomes the third derived
improvement in this project reported from the metric it optimises.

### The bar, stated so it can be falsified

Keep the head only if it clears **+50 Elo on a 400-game SPRT** against a scalar
run trained to the same stage from the same clone and the same seed, AND the
midgame band `m` improves relative to `sc`. `l` is a screen, never a merge
criterion. At 400 games SE is ~+-35 Elo, so anything under ~+70 needs a second
batch.

**And the honest A/B needs three arms, not two.** At fixed `--critic-lr 1e-3`,
switching to CE changes the critic's effective learning rate by a factor that
varies from 0.4x to 12.6x across the state space (tanh-MSE |dL/du| is 2.25 at
|V|=0.5 and 0.079 at |V|=0.99, against CE's O(1)). A scalar-vs-hlgauss test at
one `--critic-lr` is confounded with a critic-LR sweep. The control is a third
arm — **scalar head at 5x `--critic-lr`** — and it is cheaper than the thing it
controls for. If that arm matches hlgauss, the loss function was never the
variable.

**The price nobody had stated:** compute and memory are free (the head is 4k
parameters, 0.013% of trunk MACs; 4.2 MB against ~231 MB for one trunk
activation). The cost is **two GPU-nights minimum** — three matched runs to a
comparable stage plus the gate — plus the diagnostic run before them.

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


---

## The net reinvented the general-drain bug (2026-08-05)

`deathwatch` over 50 ladder losses of the NEURAL bot:

```
16/50 losses: army left the general in the last 15 ticks with an enemy stack
              within 3 steps. One trace: general 85 -> 2 with a 64-stack adjacent.
mean garrison the tick before death: 27.7, dying to stacks of 30-90
mean reinforcement brought in:       16.4  -- it reacts, and arrives short
```

The heuristic needed three separate commits to stop doing exactly this
(`324daa0` the thrust, `fa1a5ca` the move scorer, `3b8871f` the strike stack).
The network arrived at the same mistake independently.

**It cannot unlearn it from self-play.** At the curriculum's short distances
emptying the general is correct tempo, so the habit is learned in the regime
where it is right; and in a mirror match both copies do it, so neither gains and
the gradient sees nothing wrong. The same symmetric blindness that deleted the
castles.

So the fix belongs outside training. `bot/policy/guard.py` wraps the shipped net
with the heuristic's hard-override tier, which `ClonePolicy` never had:

1. **win-in-one** — rules-exact, no judgement
2. **deathtouch** — from turn 800 any touch on a general wins, so this is checked first
3. **garrison veto** — refuse a move that empties the general when a visible
   enemy stack within `general_block_radius` is at least `general_block_ratio` of
   the garrison. **The NARROW version**: v16's radius 3, ratio 0.5. The wide
   version — an unconditional lock at radius 6 — measured **-265 Elo** and must
   not come back. Same idea, different settings, opposite sign.

The veto hands the net **its own second-choice move**, not a hand-written
alternative, so the policy still decides everywhere the guard has no opinion.

**Status: written, selfchecked, UNTESTED against a baseline.** Needs wrapped-vs-raw
on identical boards before it ships.

## The audit: the trainer was clean, the instruments were not (2026-08-05)

An adversarial line-by-line pass over `learn/selfplay.py` and `learn/vecroll.py`
— the two files producing current results — found **no correctness bug**. Seat and
reward bookkeeping, GAE at episode boundaries, the KL anchor's rollout-time
logprobs, masking before `log_softmax`, the scan identity at 480 turns, and the
absence of any surviving shaping all hold. An exhaustive check of the dihedral
permutation over all 8 group elements x every cell x every direction x builds
found 0 mismatches, against an in-repo selfcheck that samples 3 cells.

Every live risk was in a side tool, and four were fixed:

| tool | what it did |
|---|---|
| `tools/mixbuilds.py` | spliced build frames into ALL shards including the 3 `train.py` holds out, so validation could not detect the splice failing — exactly how the previous castle attempt hid its own failure. At reps=16 the same frame was on both sides. |
| `tools/pools.py` | claimed its boards were the ones `--backend cpu` plays. Disjoint seed blocks; any CPU/GPU quality A/B was cross-distribution. |
| `learn/train.py` | printed `arena.runner --a ours` as the suggested benchmark. Bare `ours` is the -265 Elo config, so every clone would have been flattered by its own tool. |
| `tools/buildprior.py` | docstring said the policy drives both seats; it plays seat 0 against `--opponent`, so the +9.094 was calibrated against v16's style specifically. |

**The pattern holds: this project's bugs are in its instruments, not its
learning.** A lying docstring is a bug, and three of these four were.
