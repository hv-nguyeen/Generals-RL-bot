# Where the project is — 2026-08-07

`docs/ml-log.md` is the full measured history and it is long. This file is the
short version: what is true right now, what is running, and what to do next.
Read this first, then the log for the reasoning behind any line, and
`docs/CLUSTER.md` for how to install and run anything on the VU box.

## FINAL STANDING (competition close, 2026-08-07)

**Deployed: `generals-bot-nn12`** — `sp16-c24` weights + test-time augmentation +
the guard. Rank **27 of 107** over 200 ladder games, peak Elo 1912.

Started the cycle at rank 27, 1840, 45% over 240 games. The bot is far stronger
now (sp16 measured +35.7 in the arena and 45% -> 66% on the ladder, 6 sigma; TTA
added +31.2 on the shipped configuration). The rank did not move because the
field improved too.

**What is deployed is the best thing measured.** Everything else was tried:

| attempt | result vs `sp16-c24`, 2000-game arena |
|---|---|
| sp40/sp41/sp42, re-draws from `sp9-i600` | all ~-30 (sp41 confirmed -31.7) |
| BC 8x64 clone | -165 |
| 8x64 grown from sp16 (sp29, sp31) | critic never warmed, KILL 2 |
| from-scratch 8x64 (sp32) | never learned to play, KILL 2 |

**sp16 was a tail event, not a typical draw.** Three independent re-draws of its
exact recipe all landed ~30 Elo below it. Do not assume the recipe reproduces.

### CASTLES ARE A SYMPTOM, NOT A CAUSE — corrected 2026-08-07

This file has repeatedly treated castle rate as causal: "every checkpoint that
actually gained ground built MORE". The correlation is real (0.73 built/game ->
+35.7 Elo, 0.59 -> +5.7, 0.00 -> -7 to -20) but the direction was assumed, never
tested. It is now tested, and it runs the other way.

`tools/buildbias.py` raises `head_b` at the build slot on a checkpoint's
INITIALISATION and lets PPO decide. On the BC-lineage 8x64, whose build slot sat
at -2.03 against a head spread of 0.517 -- four sigma below every other action:

* `--bias 1.5` moved it to -0.53, still the lowest slot. `bld` stayed 0.00-0.02.
* `--bias 3.0` moved it to +0.97, the HIGHEST bias in the head. `bld` reached
  only 0.06 and decayed to 0.01 within eight iterations, with `bldA` steady
  around -1.2.

Two conclusions. **The head bias was never the binding constraint**: a build is
only legal with 35 army on an owned tile, and a policy that plays a spread
expansion game rarely has it, so the opportunity mostly does not exist however
the logit is biased. And **where builds did happen, PPO measured them and drove
the rate back down** -- castles cost that policy games.

So a strong policy builds castles because it can afford them and knows when.
Building does not make a policy strong, and forcing builds on one that cannot
support them just loses. Do not chase the build rate as a target; it is an
indicator of a policy that is already working.

### A run killed by KILL 2 can be revived with an EARLIER critic

sp44 (BC 8x64, `--start-stage 2`) climbed the champion column 0.28 -> 0.36 ->
0.40 -> 0.43 over 150 iterations -- about +115 Elo, the only sustained
improvement anything showed this cycle -- then its critic collapsed after the
promotion to stage 3 and KILL 2 fired at iteration 185.

Restarting from `sp44.best` **with `sp44.resume.npz`** inherited the collapsed
critic and the counter climbed monotonically 1 -> 9 with no reset, on course to
die again. Restarting the same policy with **`sp43.resume.npz`** -- the critic
from the earlier run, which had warmed at stage 0 and carried sp44's whole climb
-- warmed within 3 iterations.

So the resume file's critic is not always the one to resume with. When KILL 2
fires, the critic in that run's own resume is by definition the one that failed;
pair the improved policy with the last critic known to have been healthy.

### The four things worth carrying forward

1. **Inference compute pays and is barely touched.** TTA over the dihedral group
   is +31.2 for zero training, at 5 ms of a 150 ms budget. That axis had never
   been spent and is still 96% unspent.
2. **Measure the ARTIFACT, not the component.** TTA measured +32.8 on a raw
   `ClonePolicy` and shipped as exactly zero, because the guard re-scores and
   dropped the flag. The `ship:` arena spec now constructs the agent exactly as
   `bot/main.py` does. Every deploy failure this cycle was a packaging bug the
   in-process arena could not see.
3. **comp-eval only works near a 0.5 score.** At 0.886 one SE is +-60 Elo and it
   misread in every direction. Moved to a champion clone at ~0.47 it predicted
   -30 against an arena result of -31.7 -- the first accurate call it made.
4. **A BC clone from the ten strongest players is far better than the log says.**
   `ml-log` records BC at 0.175 vs `v16`; 8x64 on 3,664 top-player games with
   dihedral augmentation reads **0.628**. It also produced the first 8x64 critic
   that ever fitted (`sp43.resume.npz`, warmed at stage 0). Both are assets for a
   next cycle, not for a deadline -- the clone is still 165 Elo behind sp16.

### First three things to try next cycle

1. **Temporal channels.** The encoder sees one frame, so it cannot distinguish an
   advancing stack from a parked one -- and the ladder losses are decapitations
   by a stack that arrives 7-36 turns after the general empties. A competitor
   independently reports using 3 look-backs plus EMAs. This needs a BC restart,
   which is now cheap and produces a strong clone.
2. **More of the move budget.** Search needs a `bot/`-side transition function and
   a trustworthy evaluator, and both are logged traps -- but TTA proves the axis
   pays, and 145 ms a move is still unspent.
3. **An opponent in a TRAINING seat.** Still never tried. Symmetric self-play is
   the most-replicated failure here: castles get deleted at every stage because a
   mirror cancels the signal. `vecroll` stacks both seats through one forward, so
   this is real surgery, not a flag.

## THE RECIPE IS EXHAUSTED — six runs from `sp9-i600`, none beat it

| run | change | result vs i600 |
|---|---|---|
| sp13 | 8x64, 3.8x params | -44.8 Elo = zero gain over its own init |
| sp14 | stage 5 | -21.0 Elo |
| sp16 | 22ch encoder | abandoned, superseded |
| sp17 | 24ch encoder, full 1200 iters | **+4.3 [-10.7, +19.4]** |
| sp18 | 20ch control, died at 563 | **-3.6 [-18.6, +11.3]** |

**Both arms are flat and ~8 Elo apart with overlapping intervals, so the encoder
change did nothing measurable in either direction.** An earlier reading of pooled
comp-eval columns (sp17 0.51, sp18 0.463) suggested a ~30 Elo gap and a story
about the channels preventing degradation; the arena refuted it. Do not revive
that story without a 2000-game number.

**comp-eval at 150 games/opponent is a shortlist tool, never a verdict.** It has
now erred in every available direction: under-read sp8 (+59.5 real), under-read
i500 and i600, over-read sp14 (+0.059 while -21.0 in the arena), and over-read
the sp17/sp18 gap by ~22 Elo. Its per-opponent SE is ±0.041.

Oddity worth keeping: sp18 builds ZERO castles and is still level with i600,
which builds 0.55.

sp17 is the decisive one because it ran to completion with a matched control. Its
champion column across all 13 comp-evals — ~1950 games — averages **0.51**, and a
separate 2000-game arena run reads **+4.3 Elo**. Two independent measurements
agreeing on nothing. Pooled score oscillated 0.545-0.655 around a 0.573 base for
1200 iterations with no trend.

So the four hidden-army channels did not rescue it, and neither did width, stage,
or more iterations. **Stop launching self-play runs from the champion.** The two
things never tried are below: a critic trained on real games, and an opponent in
a TRAINING seat rather than the eval set.

One detail worth carrying: both sp17 and sp18 ended at `bld 0.01` with `bldA`
around -3, having stopped building castles entirely. Every checkpoint that
actually gained ground in this project built MORE.

### SOLVED 2026-08-06: castles only pay past ~17 tiles, and the curriculum unlearns them

Measured across three runs the same night, same checkpoint, only `--start-stage`
different:

| stage | distance | `bld` | `bldA` |
|---|---|---|---|
| 3 | 11-17 | 0.00-0.02 | -0.9 to -3.0 |
| 4 | 17-24 | 1.16-1.59 | -0.05 to +0.35 |
| 5 | 17+ | 1.82-2.44 | -0.12 to +0.25 |

Close generals mean army spent on a castle is army not spent attacking, so at
stage 3 the policy correctly learns castles are bad. **The ladder is 17+, where
they are good.** sp19 reproduced the sp17/sp18 endpoint -- `bld 0.01`, `bldA -3`
-- by iteration 77 instead of 1199, and `comp-eval` (measured at dist 17+) fell
0.886 -> 0.858 while it did.

This explains the unexplained sp17/sp18 signature above.

**Careful with the comp-eval half of it.** SE is +-0.035 at 400 games. Stage 0's
0.886 -> 0.807 is 2.3 sigma and real. Stage 3's readings over 250 iterations are
0.858, 0.877, 0.910, 0.881, 0.896 -- every one inside 1 sigma of base, i.e. flat.
The 0.858 was called a decline here on first sight and it was 0.8 sigma of
nothing.

So: **stage 0 actively damages competition play; stage 3 merely fails to improve
it while unlearning castles.** The castle result rests on `bld`/`bldA`, which are
per-iteration means over 256 games and far tighter than comp-eval.

**Refined the same night by sp22, and the first framing was too narrow.** At
stage 4, where castles do pay, sp22 still went `bld 1.59` at iteration 0 to
`0.03-0.11` by iteration 74, recovering only to `0.18-0.56` by 125. So it is not
the curriculum that unlearns castles -- **self-play unlearns them at every
stage**, and short distances merely finish the job (stage 3 settles at 0.00,
stage 4 at ~0.3). The champion arrives building 1.59 a game and training cuts it
3-5x wherever it runs.

That points at the opponent, not the board: in a mirror match both sides skip
castles together and neither is punished for it, which is the same blind spot as
the evaluator problem in `ml-log.md`.

**The impasse, stated exactly:** stage 3 trains but teaches the wrong game;
stage 4+ is the right game but the critic will not fit there.

## START HERE

### Ladder games are SHORTER than they feel: median 349 turns, mean 400

Measured over 600 harvested games from the ten strongest players:

```
p10 193   p25 251   p50 349   p75 484   p90 674   p95 819
40% reach 400 turns, 24% reach 500
```

Against our self-play at each stage:

| stage | distance | self-play turns | vs the ladder |
|---|---|---|---|
| 4 | 17-24 | ~420 | close to the mean of 400 |
| 5 | 17+ | ~500 | the ladder's 75th percentile |

**`--start-stage 5` is a measured non-starter**, but NOT for the reason first
recorded here, and the first comparison offered for it was not like-for-like.
Both arms, arena, 2000 games, `.best` against `.best`:

```
sp24.best (stage 4)   +5.7  [ -9.3, +20.8]
sp25.best (stage 5)  -14.9  [-30.1,  +0.1]   accept H0
```

~20 Elo for stage 4, and the two differ only in `--start-stage` (both C=24,
`--stage-replay 0`, `--warm-evar 0.05`, same init and critic). The version of
this claim first written here compared sp24's ARENA number on `.best` against
sp25's comp-eval AVERAGE over a whole run -- different instrument, different
checkpoint. Caught in review, not by the author.

Note sp25.best builds 0.45 castles a game and still loses, so castles are
correlated with strength but are not the whole mechanism.

**Why is unknown.** The first explanation written here -- that stage 5 trains on
the wrong part of the distribution -- is contradicted by the distances:

```
real ladder generals-distance (600 games):  mean 22.7  median 22
  17-24: 69.0%    <- all stage 4 ever sees
  25-30: 23.7%
  31-40:  7.3%
```

Stage 5 self-play runs at `dist 22.4-23.7`, matching the real 22.7 almost
exactly, while stage 4 runs at 20.0-20.6 and never sees 31% of real games. The
arm with the CORRECT distance distribution is the one that lost by 24 Elo.

Game length is confounded and cannot carry the argument either: mirror self-play
drags games out regardless of distance, so stage 5's ~500 turns against a ladder
mean of 400 says more about self-play than about the boards.

**The untested middle is a 17-30 band, which would cover 92.7%.** `STAGES` is a
module constant with no CLI override, so testing it needs a flag.

What is certain: stage 4 produced the only checkpoint that ever beat the
champion, measured in an arena playing the FULL competition distribution
(`dmin` defaults to 17, unbounded). Truncated training transfers to untruncated
evaluation, whatever the mechanism.

### Overnight 2026-08-06 into 08-07: four arms, and what to do with them

Deadline is roughly 2026-08-08. Currently rank 29/100 at 1864, climbing.

| run | node | init | the one variable | why |
|---|---|---|---|---|
| sp24 | A | `sp16-c24` | none, the control | does iterating from the new champion gain again? |
| sp26 | B | `sp16-c24` | `--dist-tail 0.15` | the untested middle of the distance dose curve |
| sp27 | sp16's, C=22 | `sp9-i600-c22`, `--seed 1` | seed only | is +35.7 reproducible, or was it seed luck? |
| sp29 | old sp24's | `sp16` widened to **8x64** | capacity | the big swing, see below |

**Morning, per node, in this order.**

1. `grep "^iter" runs/nn/spNN.log | tail -n 3` -- is it at `stage 4 (17-24)` still,
   is `bld` above ~1.0, is there no `[critic warmup]` counter climbing?
2. `cp runs/nn/spNN.best.npz runs/nn/spNN-peak.npz` BEFORE anything else.
3. `$PY -m arena.runner --a clone:runs/nn/spNN-peak.npz --b clone:runs/nn/sp16-c24.npz --games 2000 --workers 16`
   (use `sp9-i600-c22` for sp27, which is C=22.)
4. Ship anything clearing **+25 Elo over sp16**, then start the next round from it.

**`--stage-cap 400` promotes every arm to stage 5 at iteration 400, and stage 5
measured -14.9.** sp16 avoided this by accident, running `--stage-cap 1200` with
`--iters 1200` so it never force-promoted. Treat iterations past 400 as a
different experiment, and prefer `--stage-cap >= --iters` on future runs.

**Runs peak at iteration 250-350 (8x32) or ~100 (8x64, from sp13).** Everything
after is decline. `--iters 1200` is mostly waste; 400-600 is the useful window.

### sp29, the capacity swing -- and why the closure it reopens was confounded

`docs/STATE.md` closed capacity on sp13: 8x64, -44.8 Elo, read as zero gain. That
verdict does not survive:

* `--stage-replay` default 0.25 landed 10:10 on 2026-08-05, `tools/grow.py` at
  13:08, and the `--init-critic` fix sp13 needed at 20:59 -- so sp13 launched
  that evening **under the 0.25 default**, and the discovery that 0.25 destroys
  castle-building came the following afternoon.
* `build_jobs` draws `rng.integers(0, stage)` over ALL lower stages, so
  `--start-stage 4` does not escape it.
* **sp13's arena castle rate was 0.16/game against the champion's 0.69** -- the
  suppression signature exactly, and its net movement over its own init was
  -1.5 Elo, indistinguishable from what 8x32 runs under the same bug produced.

So the data cannot separate "capacity is useless" from "the replay bug flattened
this run like it flattened the others". sp29 tests it properly: 8x64 grown from
sp16 with `--stage-replay 0`. The widening is play-verified function-preserving
(720 positions, identical argmax), so sp29 starts AT champion strength and any
divergence is capacity alone.

Odds are honest, not hopeful: ~12% of clearing +25. Downside is one node-night,
because the grown init IS the champion.

Only widen, never deepen -- 12-layer plain trunks killed the critic twice (sp5,
sp11).

### THE LADDER CONFIRMED IT: 45% -> 65% win rate (2026-08-06)

```
nn10  sp9-i600     1840,  rank 27/94,   45%  over 240 games
nn11  sp16.best    1864,  rank 29/100,  65%  over 228 games
```

**+20 points of win rate at n=228 is about 6 sigma** (SE 3.2pp). The first
unambiguous ladder improvement in this project, and the first time a local
instrument and the ladder have agreed.

Read the WIN RATE, not the rank. The rating is still converging -- it restarted
from 1756 after the swap and has climbed 108 points so far; sustaining 65%
implies roughly 107 Elo above the opponents currently being matched, so it is
under-rated and should keep rising. Rank moved 27/94 to 29/100 because the field
grew and rank is dense around 1850, where 24 Elo is worth a couple of places.

Note this is a much bigger effect than the arena's +35.7 measured against
`sp9-i600`. Different opponent pools, so the two are not directly comparable, but
for once the local number was the CONSERVATIVE one.

### FIRST CHECKPOINT TO BEAT THE CHAMPION: sp16.best, +35.7 Elo (2026-08-06)

```
sp16.best vs sp9-i600-c22, 2000 games
  1077W 51D 872L  score 0.551
  elo  +35.7  [+20.7, +50.9]   accept H1
  built/game  A 0.73  B 0.71   first castle turn 162 in 1042/2000
```

Nine runs from this champion had produced nothing. This one clears the +25 bar
with the interval entirely above zero. Saved as `runs/nn/sp16-i350-elo36.npz`,
sha256 `5a834c8b0308953a65d5981ee605328be022408fb84d9b2cfe7a70bbc153e78b`. C=22,
so it packages on the C=22 node and compares against the C=22 champion.

**MEASURE `.best`, NOT `.live`.** The same run, 450 iterations later:

```
sp16.live   built/game A 0.00   elo -9.2  [-24.3, +5.8]   accept H0
```

The peak is transient and the policy loses its castles again after it. Every
arena verdict recorded earlier on 2026-08-06 -- sp19 -7.1, sp22 -20.0 -- was
measured on `.live`, so those runs may have had a good checkpoint nobody looked
at. Re-measure them on `.best` before believing they failed.

**`.best` is also at risk while a run continues.** comp-eval overwrites it
whenever a later eval scores higher, and at +-0.029 a noise spike can replace a
+35.7 checkpoint with a worse one. Copy it out under a new name the moment a run
measures well.

The castle correlation now holds across every measured checkpoint:

| checkpoint | `built/game` | Elo |
|---|---|---|
| sp16.best | 0.73 | **+35.7** |
| sp24-peak | 0.59 | +5.7 |
| sp16.live | 0.00 | -9.2 |
| sp19.live | 0.00 | -7.1 |
| sp22.live | 0.00 | -20.0 |

### THE CAUSE: `--stage-replay` defaults to 0.25, so "stage 4" was never stage 4

`--stage-replay` draws that share of every training batch from **already-cleared
stages**. Default 0.25. So a run launched with `--start-stage 4` still takes a
quarter of its boards from stages 0-3, including stage 0's 2-6 -- boards where
castles are correctly worthless. It learns that, and it is right to.

Measured 2026-08-06, five runs, arena at 2000 games against the champion:

| run | `--stage-replay` | `built/game` A vs B | Elo vs `sp9-i600` |
|---|---|---|---|
| sp16 | **0** | **0.53 vs 0.62** | **-4.2 [-19.2, +10.9]** |
| sp19 | 0.25 (default) | 0.00 vs 0.54 | -7.1 [-22.2, +7.9] |
| sp22 | 0.25 (default) | 0.00 vs 0.57 | -20.0 [-35.1, -5.0] |
| sp18 | 0.25 (default) | 0.00 vs 0.55 | -3.6 |
| sp17 | 0.25 (default) | `bld 0.01` at end | +4.3 |

**Every run that kept castles is level with the champion. Every run that lost
them is at or below it.** sp16 is the only one launched with `--stage-replay 0`,
and it is the only one that still builds.

This also resolves a discrepancy that looked like a contradiction: sp22's
in-training `bld` read ~0.3 while sp16's read 0.66, because sp22's figure is an
average over a board mix that was 25% short games.

**Use `--stage-replay 0` on any run that starts at stage 4 or above.** The flag
exists for a real reason -- sp8 lost the earlier distances after each forced
promotion -- but that reason applies to a run climbing the curriculum, not to one
pinned at competition distance.

### comp-eval against v16 is measured at 0.886, where it is nearly blind

comp-eval never enters the gradient -- it only chooses which checkpoint gets
`<- kept`. But it chooses badly at the operating point every run has used.

Elo resolution per unit score is `173.7 / (s(1-s))`, so where the score sits
decides what the instrument can see:

| measured at | Elo per 0.01 score | +-1 SE in Elo |
|---|---|---|
| 0.886, vs `ours:configs/v16.json` alone | ~17 | **+-60** |
| 0.757, vs two opponents (sp16) | ~9 | +-26 |
| 0.500, vs `clone:sp9-i600` | ~7 | +-17 |

sp19's series over 500 iterations reads 0.848 to 0.910 -- about 105 Elo of pure
noise around a flat mean. **It cannot see the +25 Elo bar this file sets.** That
is the mechanical reason comp-eval has misread in every direction, and it is not
fixed by more games: 400 -> 2000 games only halves the SE, while moving the
operating point from 0.886 to 0.500 improves resolution 3.5x for free.

**Every future run should include `clone:` the reigning champion in `--opp`.** A
score near 0.5 is where the instrument is sharpest, and it is also the only
opponent whose strength is the thing we care about beating.

**THERE ARE FOUR GPU NODES, not two.** The MOTD maps them: `1.compute.vu.nl` =
pcoms008a, `2.` = pcoms008b, `3.` = pcoms009a, `4.` = pcoms009b. Only `/home` is
shared; every `/local/data` is separate. A run was lost for a day because its log
lived on a node nobody had logged into since. `find /local/data/vng205 -maxdepth
4 -name "sp*.log"` on each node is the way to find them.

**sp16 is on the third node and is the most promising run in the project.**
C=22, `--start-stage 4`, critic warm-started from sp9's self-play resume,
`--warm-evar 0.03`, and comp-eval over TWO opponents at 600 games (SE +-0.028),
which is the only kind of instrument this file trusts.

```
  0  0.757   [v16 0.87  sp3-c22 0.64]
 50  0.768   [v16 0.87  sp3-c22 0.67]
100  0.777   [v16 0.91  sp3-c22 0.65]
150  0.807   [v16 0.92  sp3-c22 0.69]   <- kept
200  0.762   [v16 0.86  sp3-c22 0.66]
```

Three consecutive rises, +1.7 sigma at the peak, then back to base. No other run
has risen three evals in a row. It crashed at iteration 202 on a server restart
and sat idle for hours before being resumed on 2026-08-06 with the SAME command
plus `--resume` (and `>>` so the log keeps its history).

**Do not extract a C=24 tarball on that node.** Its repo is C=22 and every sp16
checkpoint is 22-channel; `net.py:181` would refuse them all.

sp16 also confirms the castle result on a second encoder: `bld 0.57-0.89` with
`bldA` steadily positive at stage 4, against sp19's `bld 0.00` at stage 3.

**Overnight 2026-08-06: sp19 and sp22.** Both start from `sp9-i600-c24b` (the
ladder champion, grown to C=24, verified `0`/`0` by `tools.grow --show` on both
clusters). They differ in BOTH stage and critic, so they are not a matched pair
-- each answers its own question.

| | cluster | stage | critic | what it asks |
|---|---|---|---|---|
| sp19 | 1 | 3 (11-17) | `value24.npz`, ladder-trained | does it keep unlearning castles all night? |
| sp22 | 2 | 4 (17-24) | `sp9-phi-c24.npz`, self-play-trained | does a same-distribution critic hold where a ladder one did not? |

Three arms died first and each result stands on its own:

* **stage 0** (sp19's first launch): critic fits instantly, but `comp-eval` fell
  0.886 -> 0.807 in 100 iterations. Training an 1840-Elo policy on 3-tile boards
  makes it worse at the ladder's.
* **stage 5** with `value24`: warmup counter reached 12/20 by iteration 40 with
  the policy frozen throughout.
* **stage 4** with `value24`: KILL 2 fired at iteration 31, never once clearing
  the gate. The diagnostic line was
  `evar -0.10 (m-0.18 l+0.09 sc m+0.03 l+0.34)`.

**Read that last line, it is the important one.** The scalar control's mid-game
explained variance collapses from +0.27 at stage 0 to +0.03 at stage 4, so long
mirror-match returns really are much less predictable. But `sc_l` is still +0.34
while our critic manages +0.09 late and **-0.18 mid, worse than predicting the
mean**. The gate is passable; `value24` just does not transfer.

Why it does not: it was fitted on ladder games between DIFFERENT players, where
outcomes often turn on a skill gap visible as an army or land asymmetry. In
mirror self-play those cues are symmetric and carry nothing. **0.759 BCE accuracy
on ladder positions bought negative explained variance on self-play returns** --
so the gate in `learn/valuetrain.py` is too weak a predictor. A future critic gate
should score explained variance on self-play returns, not accuracy on field
positions.

sp17 (+4.3) and sp18 (-3.6) are the controls and need no re-running: same
champion, same C=24 encoder, cold critic, both flat on 2000 games.

```bash
grep -E "evar|sc " runs/nn/sp19.log | tail -n 20
```

**`evar_l - sc_l > 0` has never been true in this project.** sp19 read -0.68 at
iteration 0 and -0.11 at iteration 1 — closing faster than any previous run, but
still under the control. What matters is whether it crosses zero and STAYS, not
whether it touches it once.

The bar for either arm, stated so it cannot be rationalised afterwards: **>25 Elo
over `sp9-i600` on 2000 arena games** (SE ~8, so ~3 sigma). Anything less is not a
result whatever comp-eval says.

**If both are flat, that closes the critic hypothesis** — one arm with a
pretrained critic, one with the train/eval distribution mismatch removed as well,
both flat, is strong evidence to stop pursuing the critic and go back to replay
forensics.

Why sp21 is worth a GPU-night: `--start-stage 5` was never viable before, and
`learn/selfplay.py` says why — *"the reason a run cannot begin at the distance its
policy actually plays is the cold critic, not the policy"*. Every run to date has
therefore trained mostly on boards the ladder never shows. sp21 is the first that
trains only on the competition distribution. It is also the higher-variance arm:
the curriculum exists because terminal-only reward over a ~20-ply credit window is
sparse on a 400-turn game, and if that crutch is load-bearing sp21 stalls inside a
hundred iterations.

## Standing — the first RELIABLE ladder number

**1840 Elo, rank 27 of 94. 107W 128L 5D over 240 games, 45%.** (2026-08-06)

At n=240 the standard error is ±22. **Every earlier ladder number in this project
was n<=36, i.e. ±58, and should be treated as noise** — including the 1849 that
sp3 read and the 1865 from the morning of the 5th. We therefore do NOT know
whether the +137 Elo of arena-measured gains (sp3 -> sp8 -> i500 -> i600) reached
the ladder at all. Not disproven; never tested, because there was no trustworthy
number on either side.

Consequence for planning: **a ladder A/B needs ~200 games per build**, most of a
day. The ladder cannot be the iteration loop. It is the only instrument that has
never been wrong, so it is the court of final appeal, not the working bench.

The field is bimodal — roughly 19 opponents beat us at under 35%, 20 we beat at
over 65%, and little in between. Nine have shut us out 6-0 (Mattz, bca,
tvojtatko, ResBot, candide, Nicholas, Kubic, bitterlessonpilled, nanomena), while
both baselines go 6-0 our way. That is what rank 27 of 94 looks like from inside:
a tier above and a tier below.

### Corrected on 2026-08-06: "loses early, wins late" was a small-sample artifact

On 43 games it read median win 436 turns against median loss 245, and a lot of
reasoning was built on it. On 337 games:

```
median win 384   median loss 348

turns     wins  losses
<200        12      27     <- 2.25:1 against us, and 12% of games
200-400     78      82
400-600     41      40
600-800     18      12
800+        12      15
```

Only the sub-200 band is lopsided. Everywhere else it is a coin flip.

## THE CRITIC IS WORSE THAN A LINEAR MODEL, and the fix is now connected

From sp17's own log, the trained critic against the `sc` control (least squares
on the 12 broadcast scalars, no board at all):

```
             critic                      sc control
midgame      -0.09 -0.05 -0.09 -0.00     +0.02 +0.05 +0.01 +0.06
last decile  +0.15 +0.22 +0.20 +0.36     +0.28 +0.29 +0.25 +0.39
```

`selfplay.py` states the reading itself: *"evar_l - sc_l <= 0 says the critic
extracts nothing the clock does not already give."* It is negative in both bands,
on every reading.

This matters more than any policy change, because **the critic is how a terminal
reward becomes per-move credit.** The advantage PPO applies to a move is the
critic's estimate of what that move did to the win probability. A critic at this
quality grades every strategic question — castles, garrison, reserves — close to
randomly. There is no separate "reward function" to design; there is a critic
that does this job or does not.

### The pipeline that fixes it was 90% built and disconnected

`learn/valuedata.py` turns replays into (position -> did this player win), and its
docstring names our exact problem: *"an evaluation function grounded in games
where REAL opponents did the punishing. Our local opponents never punish
over-commitment."* No action inference, so none of the label noise that capped
behaviour cloning. `/local/data/vng205/val` was built at some point and went
unused — because until `--init-critic` landed on 2026-08-05 there was nowhere for
the output to go. `--init-critic` now accepts both a run's `.resume.npz` and a
standalone `valuetrain` model.

**BUILT AND GATED 2026-08-06.** `field24/` is 3,453 ladder games from the ten
strongest players, harvested to `/local/data/vng205/val24` as 660,560 positions
in 11 shards (stride 4, both seats, ~310 MB).

```bash
$PY -m learn.valuedata ~/field24 --out /local/data/vng205/val24 --workers 32
$PY -m learn.valuetrain --data /local/data/vng205/val24 --out /local/data/vng205/value24.npz --layers 8 --channels 32
```

`--layers 8 --channels 32` is not optional: `valuetrain` defaults to
`DEFAULT_LAYERS = 4`, and a 4-layer critic cannot be loaded by `--init-critic`.

**Result: control 0.721, net 0.759, gain +0.039 — the first ML instrument in this
project to beat its own control.** Read it with one discount: 0.759 is the max
over six epochs, chosen on the rows that score the gate, and the last-third
plateau sits near 0.740. The unselected gain is nearer **+0.02**, which is a pass
at the threshold rather than a comfortable one.

The first attempt read net 0.753 against control 0.786 and the gate correctly
refused it. That was the split, not the critic: `valuedata` writes shards in
source order, so holding out `shards[-1]` held out a *player*, and training loss
fell to 0.3556 while val accuracy wandered. `--val-frac` now samples within every
shard. Cost of finding this: ten minutes of GPU, because the control ran before
training rather than after.

The lesson generalises past this file — **a held-out split that follows the write
order of the data is a held-out subpopulation.** Every shard writer here groups
by source.

Sizing: ~380k samples from the existing `field/` at stride 4, both seats, against
a ~71k-parameter critic. `--stride 2` doubles it for free. Dihedral augmentation
is EASIER for a critic than for a policy — the label is invariant under flips and
rotations, so no action remapping — but `valuetrain` has no `--augment` yet, and
the honest first step is to train once and compare train against held-out loss.
The real risk is distribution shift, not size: these are other bots' positions.
It is a warm start and PPO continues on-policy, so it only has to beat random.

**Downloading (2026-08-06, on the laptop):** `~/Downloads/field24/` — full game
histories for the ten opponents that beat us, via
`python -m analysis.official fetch --player NAME --delay 1.0`. ~40 KB per game
gzipped, ~1900 games, ~75 MB. `/api/leaderboard?matches=1&player=NAME` serves any
player's match list, so the strongest players' complete games are available.
The fetcher now retries a 403 with 30/60/120s backoff and then gives up. **Never
hammer it** — that is how the cluster lost access once.

### Per-opponent at n=36 — SUPERSEDED by the n=240 numbers above

Kept because the thor forensics below were run on it. The 0-6 has since become
1-11 over a much wider field, and eight other opponents now shut us out too, so
"thor is uniquely a wall" did not survive the larger sample.

```
thor                   0W 6L 0D     0%
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

### Game lengths at n=36 — SUPERSEDED, see the n=337 table above

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

8 layers, 32 channels, ~73k parameters. 24 input channels (12 spatial + 12
broadcast scalars). 3970 actions = 441 cells x 9 slots + pass, where slot 8 is
BUILD. Runs 1.2 ms a move against a 150 ms budget.

The 8x32 trunk is fixed for everything that has to interoperate, on either
cluster: `learn.valuetrain` defaults to 4 layers, so a critic built without
`--layers 8 --channels 32` cannot be loaded by `--init-critic`. Capacity is a
closed question, not a per-node choice — see the re-closure below.

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

## RUNNING overnight 2026-08-05: the first controlled experiment

Same init (`sp9-i600`), same warm critic, same stage 4, same `--stage-cap 1200`,
same gauntlet including the champion itself. **One variable: the encoder.**

| | tree | encoder |
|---|---|---|
| `sp17` | `/local/data/vng205/generals-bot` | 24ch — GARRISON, HIDDEN_OPP, MAX_STACK x2 |
| `sp18` | `/local/data/vng205/c20/generals-bot` | 20ch — control, built from tag `c20-nn10` |

At iteration 100, the `sp9-i600` column:

```
sp17   0.50 -> 0.56
sp18   0.50 -> 0.46
```

Both start at exactly 0.50, which is the migration verifying itself — the policy
IS i600 there, playing a copy of itself. Then they diverge in opposite
directions. 150 games each, so the 0.10 gap is ~1.7 se: suggestive, not settled.

**The verdict is the arena, not comp-eval:**

```bash
$PY -m arena.runner --a clone:runs/nn/sp17.best.npz --b clone:SP18BEST --games 2000 --workers 32
```

Both are 24- and 20-channel respectively, so `tools.grow` the control up first.

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
