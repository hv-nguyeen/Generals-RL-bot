# Where the project is — 2026-08-05

`docs/ml-log.md` is the full measured history and it is long. This file is the
short version: what is true right now, what is running, and what to do next.
Read this first, then the log for the reasoning behind any line, and
`docs/CLUSTER.md` for how to install and run anything on the VU box.

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

**On the ladder now: `generals-bot-nn6`** — `sp5.best.npz` packaged with
`configs/v13.json`, guard active. Early reading is bad. Record `n` before
treating that as a result: SE is +-58 at n=36, and this file's own rule is that
Elo needs 100+ games.

If it holds past n=100 it is a genuine contradiction, not a mis-ship. The arena
put sp5 **+32.2 Elo over sp3** on 2000 games, and sp3 is the checkpoint that
read 1849/1808. A ladder result below sp3 cannot be explained by "we shipped the
weaker overnight run" — sp5 losing to sp8 by 25.2 says nothing about sp5 vs the
thing currently ranked. That would make comp-eval-plus-arena instrument twelve,
and the first one to fail *after* passing an internal consistency check.

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

## Where the checkpoints physically are

`/local/data` is per-node and nothing syncs it, so each run can only see its own
files. This has cost time twice; the map:

| cluster | holds | reach it |
|---|---|---|
| 1 | `sp3-i500-0733`, `sp3-bp-0807`, `sp8.*`, `configs/` | `ssh -J vng205@ssh.data.vu.nl vng205@1.compute.vu.nl` |
| 2 | `clone20*`, `sp5.*` | JupyterHub lands here |

Any head-to-head needs both sides on ONE node. Copy through `~` and verify the
file arrived before running anything.

## Overnight runs, both finished

| run | init | base | best | promotions | ended by |
|---|---|---|---|---|---|
| `sp8` | `sp3-bp-0807` | 0.759 | **0.820** @200 | both FORCED | iterations |
| `sp5` | `clone20-d12-bp` | 0.335 | **0.800** @350 | three EARNED | KILL 2 |

Read each against its own `base`, not against numbers from another instrument —
sp3-bp scores 0.807 in the arena and 0.759 in comp-eval, and they are not
comparable. sp8 moved 1.2 SE and both its stage promotions were cap-forced,
which is the trainer reporting the policy was not ready. sp5 moved +0.465,
cleared every gate, and hit 0.800 on the dist-17+ eval while training only at
7-13.

sp5's critic reached **evar 0.405, the highest ever recorded here**, then fell
under 0.05 and stayed there for 20 iterations until KILL 2 fired. The run did
not plateau; its value function died. `.best.npz` is written before the kill, so
both bests are safe.

Both `.best` scores are a max over ~24 draws and are ~1 SE optimistic.

## The head-to-heads that decided it

2000 games each, `faults 0` on all three:

```
sp8 vs sp3   +59.5  [+44.5, +74.6]   accept H1
sp5 vs sp3   +32.2  [+17.2, +47.4]   accept H1
sp5 vs sp8   -25.2  [-40.3, -10.3]   accept H0
```

**sp8 > sp5 > sp3, and the triangle closes.** Common-opponent predicts sp8 over
sp5 by 59.5 - 32.2 = +27.3; measured directly it is +25.2. Two Elo apart. That
is the first internal validation any instrument here has passed — every earlier
one was a single number nothing could contradict.

`runs/nn/sp8.best.npz` is the new champion, +59.5 over the checkpoint that
scored 1849 on the ladder.

Caveat that still stands: sp3, sp8 and sp5 share an ancestor, so a blind spot
common to all three is invisible to all three comparisons. v16 is the only
strong opponent from another lineage.

## 2000 games costs 85 seconds

Every Elo number in this project before 2026-08-05 was 400 games at +-29,
because 400 games used to be slow. It is not: 400 in 14s, 2000 in ~85s at 32
workers, SE +-8. **Default to 2000.** sp8's comp-eval gain read as 1.2 SE of
noise and it is +59.5 in the arena; the instrument was too blunt to see it.

## Next, in order

**1. The guard is neutral, measured three ways. Stop crediting it.**

```
guard vs clone (mirror)   0.507   +4.7  [-10.3, +19.7]
guard vs hunter           0.979   1956W 3D 41L
clone vs hunter           0.979   1957W 3D 40L
```

One game in 2000 between wrapped and raw. It is not what sank nn6, and it is
not an improvement either. Keep it — win-in-one and deathtouch are rules-exact
and cost 0.1 ms — but it is not an Elo source and no further tuning of the veto
radius is worth a run until something can measure it.

`arena/runner.py` now prints A's mode histogram, so **the next run of the above
answers whether the veto ever fires at all.** Near-zero `guard-veto` means the
trigger is too tight; a large count with 0.507 means the veto is real and
worthless. Those want opposite fixes and nothing before today could tell them
apart.

**HUNTER IS SATURATED AT 0.979 — do not put it in a gauntlet.** It loses to
everything we have, so it scores every checkpoint the same and spends games to
say nothing. This is the trap the multi-opponent comp-eval was built to expose,
and it caught its first opponent before the first run.

The bitter part: the opponents that discriminate are `v16` (0.733) and our own
checkpoints (sp8 vs sp3 = 0.585). **Diversity is capped by strength.** A
stylistically different opponent that cannot survive contact measures nothing,
so the useful gauntlet is v16 plus the lineage, which is exactly the correlated
instrument the first rule warns about. There is no local escape from this; the
ladder is the only opponent set that is both strong and different.

**2. Decide what the ladder is measuring.** nn6 (sp5 + guard) is underperforming
against an arena that predicted +32.2 over sp3. Get `n` first. If it is real,
the arena triangle passing its own consistency check did not make it valid, and
`sp3-i500-0733.npz` on cluster 1 is the rollback to a known 1808.

**3. Information channels.** Threat (nearest enemy stack, size, distance to our
general), dist-home BFS, fog memory, per-cell build cost. All computable, none
in the 20 channels. Zero-init the new stem columns and `sp3.best`'s function is
preserved exactly, so training continues from the current best with no
curriculum restart.

Precedent: adding the 8 broadcast scalars moved the clone from 0.152 to 0.360 —
the largest single measured gain in the project. Information has beaten capacity
and schedule every time here.

## The castle sweep, and why its numbers cannot set the rate

`sp8.best` builds 0.23/game at p(build|legal) = 0.0034. `tools/buildprior.py`
shifts one scalar — slot 8's bias in the 3x3 head, shared by every cell. Each arm
is 2000 games against unmodified `sp8.best`, `faults 0`:

```
target 0.008   +22.8  [+7.8, +37.9]
target 0.015   +47.0  [+31.9, +62.3]   <- peak
target 0.030   +31.4  [+16.4, +46.4]   <- the rate the ladder rejected
```

**The 0.030 arm was a pre-registered control and it failed.** That is the setting
that gave sp8 `bld 2.48` and cost ladder rank, and the arena scores it +31.4. The
arena over-rates castles, for a reason that is not mysterious: every opponent
here is our own lineage and none of them punish over-building. Nobody rushes you
while you are 35 army down. **These numbers cannot pick the rate.**

What survives: all three arms positive over 6000 games, so the direction — more
than self-play's 0.23/game equilibrium — is consistent; and the curve has an
INTERIOR peak, so even a castle-biased instrument finds a point where more stops
helping. The peak is ~2.2 se above the 0.008 arm and indistinguishable from
0.030, so "0.015 is optimal" is NOT supported. "Somewhere in 0.008-0.030 beats
0.0034" is.

Fairness both ways: the ladder verdict was on `sp3-bp-0807`, a different base
policy, at an unrecorded `n`. Two weak instruments disagreeing is not one
refuting the other.

Settle it on the ladder, one variable at a time: **nn7 = sp8.best plain**,
**nn8 = sp8-bp015**, identical but for the prior.

## Why self-play will never answer this

PPO is not failing to learn about castles. At `bld 2.48` with 256 games an
iteration it saw ~635 builds per iteration and took 50 iterations to crush the
rate — ~30,000 samples against the ~2000 that `ml-log.md:939` computes are needed
for a confident read at 0.11 signal-to-noise per build decision.

It had fifteen times the evidence required and concluded castles were not worth
it. **In a mirror that is correct**: both sides build, both gain the same income,
the win-probability differential is zero. No reward change, no critic change and
no extra sampling alters that. Either the opponent builds at a different rate
than you do, or the rate is set after training.

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
* The arena needs the four BLAS exports from `docs/CLUSTER.md:90` as much as
  training does, and there it fails silently instead of loudly: 32 workers
  oversubscribe to a ~350 ms mean move, every move over 150 ms becomes a fault
  and a forced pass, and the match scores 0.500 — reproducible and wrong. Read
  the `faults` line before the Elo line, every time. Caught once this session
  mid-run; the giveaway was 2000 games crawling where 400 took 62s.
* `make` targets hardcode `PY := .venv/bin/python`, which does not exist on the
  cluster. Every cluster invocation needs `PY=` (`docs/CLUSTER.md:63`).
* The submit recipe's last line writes a score-stamped snapshot so the live
  `.best.npz` cannot overwrite the keep. Write a NEW filename. A copy of the
  form `cp runs/nn/spN.npz runs/nn/sp3-bp-0807.npz` lands on an existing
  checkpoint and destroys it, which is the one thing this file says never to do
  — and it leaves a file whose name lies about its contents.
