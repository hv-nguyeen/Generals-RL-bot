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

**1865 Elo, rank 23/90, 18W-17L-1D over 36 games** (2026-08-05). The best
reading the project has had. The *early* reading of that same run looked bad and
was noise — the ±58-at-n=36 rule caught exactly the mistake it exists to catch,
so do not react to a ladder number before `n` is real.

**WHICH BUILD THESE 36 GAMES BELONG TO IS NOT RECORDED, and the profile does not
say.** nn6 (`sp5.best` + guard), nn7 (`sp8.best` + guard) and nn8
(`sp8-bp015` + guard, the castle prior) were all built the same day. Match ids
run 109463-109498 in time order, so a build that went up mid-session splits the
record and the pooled 1865 belongs to no single bot. **Write down the id of the
first match after every upload** — without it a ladder number cannot be
attributed, and attribution is the entire reason for shipping one variable at a
time.

### Per-opponent, and one of them is a wall

```
thor                   0W 6L 0D     0%     <-
blakeboss              2W 3L 1D    33%
Oleksandr Tymkovych    3W 3L 0D    50%
john9801               4W 2L 0D    67%
vojtechpour            4W 2L 0D    67%
Hunter (baseline)      5W 1L 0D    83%
```

0-for-6 against `thor` is p = 1.6% under even odds. One opponent has something we
have no answer to, while everything else sits at 33-83%. That is the single
largest identified loss of Elo and it is an exploitable weakness, not variance.

Also note the loss to the **Hunter baseline**, which the arena scores 0.979 over
2000 games. One loss in six is only ~12% unlikely, so not yet evidence — but the
arena says that should happen once in 50 games, not once in 6.

### Game lengths, measured on the ladder rather than assumed

36 games: median **320** turns, and **4 of 36 (11%) cross turn 800**.

```
1200, 948, 914, 808, 773, 681, 580, 557, 515, 498, 476, 451, ...
```

Deathtouch (`rules.DEATHTOUCH_TURN = 800`) is therefore live in about one game in
nine. Training reaches it — `--max-turns` defaults to `TURN_LIMIT` 1200 — but the
stage-5 mean is ~480, so it is the tail there too.

### The 1200-turn draw is a specific, named failure

Game `#109482` vs blakeboss is the `1D`: it hit the turn limit. From turn 800 any
move onto a general wins outright regardless of army, so the bot had **400 turns
in which the whole game reduces to "walk onto their general"** and did not
convert.

`guard.winning_move` handles deathtouch only when the enemy general is already
visible and adjacent. **There is no general-hunting behaviour anywhere in the net
or the guard.** Under deathtouch that is the only thing that matters. A known
regime, 11% of games, where the correct policy is nearly trivial to state and we
have none of it — a sharper target than castles.

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

## THE FINDING: it dies to army it cannot see

Ladder forensics on the ten shortest losses, measured at the tick the general is
EMPTIED rather than at the tick it falls:

```
id       garrison   enemy total   visible   HIDDEN   visible threat <=3
109479      27          237          33       204            2
109498      45          150          31       119            2
109484      50          134          18       116            0
109470      19          159          12       147            2
109478      26          121          24        97            7
109476      13          162          76        86            0
109486      31          172         105        67           51
109471      22          102          60        42            2
109475      21          112          87        25            0
109466      15           41          19        22            0
```

**Hidden enemy army exceeds the garrison in 10/10, by 1.2x to 7.7x. Visible
threat within 3 is ~0 in 9/10.** The killer arrives 7-36 turns later.

Measure at the DEATH tick and the killer is adjacent in all ten — that reading is
worthless and was reported as decisive earlier in the day. The decision that loses
the game happens 10-36 turns before, when the board looks safe.

So this is not garrison valuation and not receptive field. **The bot has no idea
how much army it cannot see.** It also explains thor (6-0 against us): thor ends
with more army on equal land because it stockpiles, and a stockpile in fog is
invisible until it lands.

**`bot/policy/guard.py` is dead as designed.** `drains_general` requires a visible
enemy within radius 3 at decision time; that condition holds in 1 of 10. It
measured neutral in the arena because it almost never fires.

### The fix, and both halves are already in the observation

`opp_army` (scoreboard total) is ALREADY one of the eight broadcast scalars.
Visible enemy army is a sum over the observation. The bot has both and never
computes the difference.

```
GARRISON      log1p(my general's army) / 6
HIDDEN_OPP    log1p(max(0, opp_army_total - visible_opp_army)) / 6
```

The ratio comes free: `log1p(hidden) - log1p(garrison)` is the log ratio and is a
linear combination of two scalars, which the first stem layer computes itself.

Visible-threat scalars are the WRONG fix for the same reason the guard is: visible
is precisely what is not dangerous at decision time.

### Every candidate scalar, measured before building

At the drain tick, 30 long losses vs 13 long wins on the ladder. Medians, and
the loss/win ratio is the whole decision rule — build it if the ratio is large,
drop it if it is not:

```
                                win    loss   ratio   verdict
hidden enemy army                28     191   6.8x    SHIPPED  (HIDDEN_OPP)
opponent's biggest VISIBLE stack 16      49   3.1x    SHIPPED  (MAX_STACK_OPP)
our biggest stack                13       6   2.2x    SHIPPED  (MAX_STACK_MINE)
hidden-army density             2.1     3.2   1.5x    rejected
BFS distance to nearest reserve   9      13   1.4x    rejected
our garrison                     34      38   1.1x    kept only as the ratio's
                                                       other half
army concentration              .030    .033  1.0x    rejected
```

A global MAXIMUM is unreachable for this architecture and that is why the max
scalars are worth channels: the army planes carry it per cell, but the critic
mean-pools (average, not max) and the policy head is a 3x3 conv (local, not
global).

**The 2-minute check killed four ideas that reasoning liked**, including two of
mine that were already written up as obvious. Ratios under ~2x are noise here.

Incidental finding worth chasing later: in **26 of 30 losses and 9 of 13 wins
there was no stack of 20+ army anywhere except the general.** The bot almost
never holds a reserve — the exact inverse of thor's "equal land, more army".

### `snipe:` — the opponent that makes this measurable

Nothing in our lineage decapitates, so an anti-decapitation fix reads ~0.500 in
the arena however well it works. `hunter` does the right thing and is saturated at
0.979, discriminating nothing. `snipe:weights.npz` is the missing combination: the
net plays the game, and a stack over `STACK_MIN` marches at the enemy general and
stays on it.

400 games per arm, same weights on both sides, plain net as A:

```
@30    +124.0                 a castle costs 35, so every tile that could build
                              marched instead. built/game 0.00.
@60     +15.6 [-18.2, +49.8]
@90      +5.2 [-28.5, +39.0]
@120     -6.1 [-39.8, +27.5]  <- default
@160     +5.2 [-28.4, +38.9]
```

**IT WAS ONLY A PEER BECAUSE IT WAS INERT.** Measured 2026-08-05 with the mode
histogram, sniper on the A side so its firing rate prints, against the real net
on identical weights:

```
threshold   fires    Elo (sniper vs plain net)
@40        13.8%      -20.9
@50         9.1%      -17.4
@60         6.8%      -20.9
@80         3.1%       +6.9
@60-200     1.7%       +8.7     <- the shipped default
```

The more it snipes, the WORSE it does, and at the default it fires on 1.7% of
turns — it is a 98%-identical copy of the net. The earlier sweep that made it
look like a peer was measuring inertness.

Against `Greedy` as the inner policy the opposite holds (0.500 inert at @120,
0.592 firing at @60, and every win a thin general against a 68-74 stack), so the
strike logic is correct. A strong policy simply punishes the commitment: marching
a 40-60 stack across the map donates army.

**The design flaw: it does not BANK.** thor's signature is equal land with MORE
army — refusing to spend, then committing once. This marches whatever stack is
biggest, continuously, from turn one. That is a leak, not a sniper.

Consequence: any comp-eval `snipe` column taken before this is fixed measures
nothing. Do not read it.

**Banking was tried and is WORSE.** Feed the general's surplus into a fixed
staging tile with half-moves, keeping a garrison, then commit when it is big —
thor's shape, and it measured **-137 / -111 / -100 Elo** at thresholds 60/100/140
with the bank firing on ~10% of turns. Reverted. Spending a tenth of your moves
shuffling army and thinning your own general is worse than not doing it, against
a policy that punishes both.

### The pattern: hand-written overrides lose to this net

```
guard  (win-in-one + garrison veto)      +4.7 Elo, fires on 1 of 10 real cases
snipe  (march the biggest stack)        -20.9 when it actually fires
snipe  (bank, then commit)         -100 to -137
```

Three attempts, three results in the same direction. **The net is now strong
enough that a human-specified rule interrupting it costs more than it gains** —
even one aimed at a defect with ladder evidence behind it. That retires the guard
as well, and it means the decapitating opponent probably cannot be hand-built.

What remains for measuring decapitation resistance: the ladder itself, or a net
TRAINED to snipe. Not an afternoon's work either way.

## Four things measured on 2026-08-05 that change how runs are set up

**Stage 3 is exhausted for `sp8.best`. Start at stage 4.** sp9 ran 400 iterations
at stage 3 and its stage-eval read **0.495** — 0.500 is the exact fixed point for
a policy against its own stage-entry weights, so it could not beat itself from
400 iterations earlier. comp-eval agreed, oscillating around 0.635 against a base
of 0.641. Two independent instruments, one negative result. Do not spend another
night there.

**The build "collapse" was correct behaviour, not a pathology.** `bld` sat at
0.01 for all of stage 3 and jumped to **0.43-0.68 within four iterations of
entering stage 4**. Stage 0-3 boards are 11-17 apart with no safe rear;
`tools/buildprior.py` says exactly that. The policy declines to build where
building is wrong and builds where it pays. This retires the whole
"self-play deletes castles" thread AND the `--stage-replay`-suppresses-builds
theory — sp11 with `--stage-replay 0` behaved identically.

**A 12-layer PLAIN trunk kills the critic. Two for two.** sp5 (12x32): evar
0.405 -> under 0.05 for 20 iterations -> KILL 2. sp11 (12x64): evar 0.14 ->
negative by iteration 43, policy frozen. Every 8-layer run has a stable critic.
The mechanism is in `net.py:145` — the residual wiring exists precisely so "the
skip path stays linear all the way through, which is the whole reason deep stacks
train", and neither run used it. **Depth beyond 8 requires `--residual`** (odd
layer count only). Until then, widen instead of deepening.

**`tools/vprobe` measures what the critic believes.** sp9's critic values a
castle at **+0.28 in z units** — the +111 Elo castle effect implies ~0.34 per
castle at v16's ~0.9/game, so it is within 20% of an independently measured
value. The castle asset is NOT unpriced and the blind-critic diagnosis is dead.
But `garrison 25 vs 2` reads **+0.0040 +-0.0017**: 23 extra army on the general
is worth 0.4% of a win, so the critic cannot tell a defended general from an
empty one. That is the decapitation problem with a mechanism attached, and it is
the one blind spot that survived the day.

## Measured non-starters — do not propose these again

| | evidence |
|---|---|
| ~~bigger networks~~ **RETRACTED 2026-08-05, see below** | the evidence was BC top-1, which this same file calls worthless |
| rollout throughput | projected 23x measured 1.7x; projected 2.5-4x measured 0.7x. The rollout is 11-24% of an iteration's arithmetic, so even a free one caps at ~1.2x |
| behaviour cloning | ceiling is median field play; saturates on label noise |
| config tuning | PSRO best-response to archive 0.522 +-0.013 — the ~100-knob space is exhausted |
| HL-Gauss critic | our returns are exactly {-1,0,+1}, so the bins are disjoint and it is a 3-way classifier. Expected -20 to +30 Elo for two GPU-nights |

### Capacity: RE-CLOSED 2026-08-05, on evidence that supports it

sp13 ran 8x64 (275k params, 3.8x) overnight from `grow(sp9-i500)`, warm critic,
same recipe. Best was comp-eval 0.703 at iteration 100 against a base of 0.682
(+0.021, inside noise) and it declined after. Head-to-head against `sp9-i600` at
8x32: **255W 13D 332L, -44.8 Elo [-72.9, -17.3], accept H0, faults 0.**

Read that as ZERO GAIN, not as "wide is worse". sp13 started from `sp9-i500`,
and i600 beats i500 by +43.3 on 2000 games — so -44.8 against i600 is almost
exactly the head start it never closed. A full night at 3.8x the parameters
moved the policy nowhere.

So the entry goes back — but for the right reason this time. The old
justification was BC top-1, a metric this same file calls worthless. The new one
is a 600-game head-to-head against the reigning champion.

Two earlier attempts failed on confounds and are not evidence either way: sp11
(12x64) died of the 12-layer plain-trunk critic collapse, sp12 (8x64) stalled at
`[critic warmup 20/20]` starting cold at stage 4.

Also observed: the wider net builds 0.16 castles/game against i600's 0.69. Every
strong checkpoint this project has produced builds MORE.

### Capacity: the retraction (superseded by the above)

"Bigger networks" sat in that table on two claims and neither holds.

**Four sizes 70k-270k all scored 0.489-0.499 BC top-1.** Top-1 is the metric this
file separately calls *worthless as a model-selection metric* — a net that
predicts the heuristic's move 65% of the time wins 1 game in 100 against it. The
capacity question was closed with an instrument we rejected.

**"Depth 12 matched depth 8 at 2x wall clock."** sp5 IS 12x32. It reached
comp-eval 0.800 from a much weaker init and EARNED all three of its stage
promotions where sp8 had both of its forced. That is not evidence against depth.

Compute is not the constraint either: 8x32 is ~32M MACs a move at 1.1 ms, and
12x128 (~4M params) is ~780M MACs, roughly 26 ms against a 150 ms budget. We use
under 1% of what the rules allow.

What capacity still cannot fix is the three measured bottlenecks — self-play's
blindness to symmetric strategies, a critic already near its martingale ceiling,
and information absent from the input. A larger net trained by a blind signal is
blind at 2x the wall clock.

**The exception, and the reason to test it: receptive field is exactly the layer
count.** 8 layers, radius 8, on a 21x21 board — no cell can see the far corner.
If the field's larger bots are DEEPER rather than merely wider, they may be
winning on global context, which is a real gap. Our largest measured gain ever
(0.152 -> 0.360) came from 8 broadcast scalars, a crude substitute for the same
thing.

**The test:** 12x64 (~290k, 4x current) on the same recipe, head-to-head against
sp10 at 2000 games on the gauntlet. One GPU-night. Until it runs, capacity is
OPEN, not a non-starter.

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
