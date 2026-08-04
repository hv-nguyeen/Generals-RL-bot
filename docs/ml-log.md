# ML attempts: what was tried, what it measured, why it failed

Running log so nothing gets tried twice. Every entry needs a measured number, not
an impression. If an entry has no number it does not belong here.

Last updated 2026-08-04.

## The one-line summary

Six ML attempts, none has produced a bot worth submitting. Every Elo gain in this
project has come from reading replays and fixing a mechanism. Weight-level and
learned changes are 0-for-9; mechanism fixes are 5-for-5.

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
