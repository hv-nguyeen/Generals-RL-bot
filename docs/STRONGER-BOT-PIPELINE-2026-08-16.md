# A pipeline that will produce a stronger TRAINED net — 2026-08-16

## Mistakes this session (so we don't repeat them)

1. **Grinding the 8×32 champion** (topology, league/self-play continuation). It
   is at its architecture ceiling — both training arms gated an exact tie
   (984-984). More training of that arch cannot beat it. *Lesson: change the
   architecture, not the iteration count.*
2. **Killing the one gate that mattered** (trained 16×96 vs champion) to chase a
   training run. *Lesson: cheap measurements first.*
3. **Training a big net with a COLD critic at competition distance.** Every
   collapse traced to this: a fresh critic can't value 17+ boards, so either the
   policy stays frozen (`warm-evar` gate) or you force it and PPO follows garbage
   advantages and degrades (`stage-eval 0.50 -> 0.40`). *Lesson: don't run PPO
   without a warm critic; and the failure mode of RL here is instability.*

## The design: DISTILL the measured-stronger teacher into one net

RL here is fragile (critic collapse). **Distillation is not.** We already have a
teacher that is *measured* stronger than the champion — the guarded ensemble
`shipens:champion+topo-night-best@logit`, +17.4 Elo [+2.2,+32.7], 0 faults. A
single net trained by **behaviour cloning** to imitate that teacher's moves:

* is **supervised** — cross-entropy to the teacher's chosen action. No critic, no
  PPO, so the cold-critic collapse that killed every run *cannot occur*.
* yields a **single trained net** (not an inference-time ensemble trick) that
  plays like the +17 teacher at one net's cost.
* goes into a **moderately bigger arch (8×64)** — 4× the champion's params to
  have capacity to fit a two-net teacher, still 8 layers so it stays far inside
  the 150 ms budget (unlike the 16-layer nets that caused ladder faults).

Why this is high-certainty: the teacher is measured-stronger (not hoped), BC is
stable, and a wider net has the capacity to reproduce the ensemble's policy.
Imitation loses a little, so expect roughly +10 to +17 over the champion in one
net — confirmed by the same paired gate, not asserted.

## The pipeline (all on the node — nothing trained locally)

New tool `tools/distill_gen.py`: plays the teacher in self-play (+ vs
heuristics), records `(features.encode(obs), teacher_move)` in `learn/train.py`'s
exact shard format (`x` f16 (N,C,21,21), `y` i32, `meta.json`). Validated
locally: shards load through `learn.train.load_shard`.

1. **Generate** the distillation set (teacher = the +17 ensemble):

       python -m tools.distill_gen \
         --teacher 'shipens:/local/data/vng205/champion.npz+/local/data/vng205/topo-night-best.npz@logit' \
         --opponents 'self,greedy,hunter,expander,ours:configs/v16.json' \
         --games 4000 --out /local/data/vng205/distill-bc --workers 32

   `self` labels both seats (2× data, on-teacher-distribution); the heuristics
   add off-distribution states for robustness (DAgger-lite).

2. **Distil** into an 8×64 student (supervised, GPU, stable, minutes-to-hours):

       python -m learn.train --data /local/data/vng205/distill-bc \
         --out /local/data/vng205/distill-8x64.npz \
         --layers 8 --channels 64 --epochs 20 --augment

3. **Gate** the student vs the champion (the verdict):

       python -m arena.runner --a ship:/local/data/vng205/distill-8x64.npz \
         --b ship:/local/data/vng205/champion.npz \
         --games 2000 --workers 32 --time-limit-ms 150 --quiet | grep -iE 'elo|fault|time'

   `elo_lo > 0`, 0 faults, time < 150 ms → a genuinely stronger **trained single
   net**. Ship it as `bot/weights.npz`.

## If we want to go past the teacher (iterated amplification)

BC can only match the teacher, not exceed it. To climb further, iterate:
distil -> new net; ensemble {new net + champion + topo-night} is a *stronger*
teacher -> distil again. Each round the single net gets stronger (this is the
AlphaZero/expert-iteration loop, minus the fragile critic). And once a strong BC
net exists, RL *does* become safe — its critic can be pre-warmed with
`learn.valuetrain` (or a low-stage curriculum) instead of cold at competition
distance, removing the collapse.

## Budget note

8×64 ≈ 2× the champion's move time (~24 ms) — far inside 150 ms. If a later,
bigger student wins on strength but busts the budget, **distil it down** to a
fast net (same tool, `--layers/--channels` smaller). Capacity for strength,
distillation for speed.
