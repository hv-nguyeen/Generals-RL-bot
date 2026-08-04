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

**Two things were settled on 2026-08-04, and they point in opposite directions:**

* **Network capacity is NOT the ceiling.** 70k to 270k parameters, four sizes,
  all 0.489-0.499 validation top-1. Stop scaling the net.
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
(0.522) and its best measured build is 1737. A trained policy that is at 0.42
against v16 after 300 iterations, still rising, is the first thing in this
project with a path past that. Throughput is now the binding constraint, not
method: ~10k games/hour on CPU rollouts against a derived ~230k/hour
GPU-vectorised.

---

## Network capacity is not the ceiling

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
