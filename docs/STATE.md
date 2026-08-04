# Where the project is — 2026-08-05

`docs/ml-log.md` is the full measured history and it is long. This file is the
short version: what is true right now, what is running, and what to do next.
Read this first, then the log for the reasoning behind any line.

## Standing

**1808 Elo, rank 24/89.** A neural policy, not the heuristic.

```
neural, best reading       1849   rank 21/86   n=36
neural, current            1808   rank 24/89   n=30
v13 heuristic, best        1737   n=102
v18 heuristic              1587   n=90
```

The heuristic is retired as a submission. It survives as an opponent, a
calibration point, and the source of `configs/v16.json`.

## Checkpoints that matter

| file | what | vs v16 |
|---|---|---|
| `runs/nn/sp3-i500-0733.npz` | best confirmed policy | **0.733** (+175 Elo) |
| `runs/nn/sp3-bp-0807.npz` | same weights + castle bias | 0.807 vs v16, but the ladder REJECTED it |
| `/local/data/vng205/clone20.npz` | the BC clone everything starts from | 0.360 |

**Never delete a checkpoint.** They are the only diverse opponent set that has
ever agreed with the ladder, and a self-referential yardstick needs them.

## Architecture

8 layers, 32 channels, ~73k parameters. 20 input channels (12 spatial + 8
broadcast scalars). 3970 actions = 441 cells x 9 slots + pass, where slot 8 is
BUILD. Runs 1.2 ms a move against a 150 ms budget.

`bot/main.py` ships the net when `bot/weights.npz` is present, wrapped in
`bot/policy/guard.py`. Delete that file before packaging a heuristic build or it
silently ships as a net.

## Training method

Curriculum self-play PPO. Generals start 2-6 tiles apart at stage 0 (games ~104
turns) and walk out to the competition's 17+ (~480 turns). Reward is terminal
only: +1 / -1 / 0. There is no shaping anywhere and two attempts died of it.

The curriculum is why RL works here after six failures. At short distance the
critic reached explained variance 0.12 in 15 iterations; at competition distance
it never crossed it in 1400.

## What is running

| node | run | init | status |
|---|---|---|---|
| c1 | `sp8` | `sp3-bp-0807` | stage 3, PPO deleted the castles, comp-eval ~0.740 |
| c2 | `sp5` | `clone20-d12-bp` | 12x32 depth test, 0.677 at iteration 250 |

Both write `.best.npz` whenever comp-eval improves, so killing either at any
point keeps its best.

## Next, in order

**1. A/B the guard.** `bot/policy/guard.py` is written, selfchecked and
UNTESTED against a baseline. It targets the 16-of-50 losses where the net
emptied its own general with an enemy stack within 3 steps. No training needed.

```
python -m arena.runner --a clone:runs/nn/sp3-i500-0733.npz --b ours:configs/v16.json --games 400 --workers 32
```

then package the same weights (the guard wraps automatically when
`bot/weights.npz` exists) and replay the same 400 games through `stdio:`.
Wrapped must beat raw or the veto radius/ratio needs loosening.

**2. Read the overnight runs.** sp8 above 0.807 means removing castles helped.
sp5 above 0.739 means depth has a higher ceiling and 8x32 gets retired.

**3. Information channels.** Threat (nearest enemy stack, size, distance to our
general), dist-home BFS, fog memory, per-cell build cost. All computable, none
in the 20 channels. Zero-init the new stem columns and `sp3.best`'s function is
preserved exactly, so training continues from the current best with no
curriculum restart.

Precedent: adding the 8 broadcast scalars moved the clone from 0.152 to 0.360 —
the largest single measured gain in the project. Information has beaten capacity
and schedule every time here.

## Measured non-starters — do not propose these again

| | evidence |
|---|---|
| bigger networks | four sizes 70k-270k, all 0.489-0.499 BC top-1; depth 12 matched depth 8 per iteration at 2x wall clock |
| rollout throughput | projected 23x measured 1.7x; projected 2.5-4x measured 0.7x. The rollout is 11-24% of an iteration's arithmetic, so even a free one caps at ~1.2x |
| behaviour cloning | ceiling is median field play; saturates on label noise |
| config tuning | PSRO best-response to archive 0.522 +-0.013 — the ~100-knob space is exhausted |
| HL-Gauss critic | our returns are exactly {-1,0,+1}, so the bins are disjoint and it is a 3-way classifier. Expected -20 to +30 Elo for two GPU-nights |

## Rules that survived contact

**No single fixed opponent measures general strength, however strong it is.**
Eleven local instruments have disagreed with the ladder, always in the same
direction. The most recent: a castle bias measured +74 Elo against v16 on 400
identical boards and dropped the ladder rank.

**Top-1 accuracy is worthless as a model-selection metric.** A network
predicting the heuristic's move 65% of the time wins 1 game in 100 against it.

**Self-play is blind to symmetric strategies.** If both sides do a thing,
neither gains, so the gradient sees only its cost. This is why PPO deleted the
castles and why it cannot learn not to empty its general — at short curriculum
distances that is correct tempo, and in a mirror both sides do it.

**Elo needs 100+ games.** SE is +-58 at n=36. 1849 and 1808 are the same number.

**Check the Elo sign, not the verdict string**, in any arena output from before
2026-08-04 — the SPRT verdict named the losing side.

## Traps in the tooling

* `Config.from_dict` fills absent keys from today's defaults, so old configs
  silently become new bots. Six ladder-measured "builds" were three configs.
  Historical builds need `git worktree` plus `stdio:`, not a config file.
* Bare `ours` is `Config()` with `lock_enabled=True` — the setting measured at
  -265 Elo. Always benchmark against `ours:configs/v16.json`.
* `tools/mixbuilds.py` used to splice into the held-out shards, which is how the
  previous castle attempt hid its own failure. Fixed; it now names the shards it
  leaves alone.
* Two runs on one L4 OOM at ~22 GB. Always `export XLA_PYTHON_CLIENT_PREALLOCATE=false`
  and check nothing else holds the card.
