# Observation-Conditioned Q-Ranker Pilot — 2026-08-12

## Decision and evidence

The 256-state, 16-rollout search probe did **not** authorize direct search
labels.  Its same-sample robust gain was `+0.0393`, but independent 8/8
selection/evaluation produced `-0.00244` with a game-clustered 95% interval of
`[-0.01465, +0.01050]`.  Greedy terminal, critic-only, and policy-plus-critic
selectors also had intervals crossing zero.  Per-row argmax distillation is
therefore rejected.

There is still a narrower, testable signal: the matched critic ranked nearby
action pairs at about 58%, and the top-six optimistic oracle was large.  The
next pilot pools noisy paired action returns across many games with a model
whose input is only the deployable observation/history.  It does not expose
hidden simulator state to the policy and it cannot emit a policy unless a
sealed-game return-gain gate passes.

## Implemented pipeline

1. `tools.searchprobe` generates raw top-six paired terminal returns.  Schema 2
   also stores the complete legal-action mask and reports even/odd cross-fit
   gains automatically.  A fresh output directory is mandatory.
2. `learn.qrank` freezes the accepted champion trunk and fits a small
   action-value head (about 2.6k parameters for an 8x32 policy).  The loss is a
   Huber loss on all paired action-return differences.
3. Complete game seeds are deterministically split into train (65%), epoch/beta
   selection (15%), and sealed test (20%).  Rows from one game can never cross
   partitions.  The test partition is untouched unless selection first passes.
4. For each candidate temperature, the selector is

   `log pi_champion(a|h) + (Q(h,a) - Q(h,a0)) / beta`.

   The ranker gate requires a positive sealed game-cluster bootstrap lower
   bound, at least `+0.01` mean return gain, 1-25% changed actions, and at least
   55% pair accuracy.  A rejected ranker is auditable but cannot be distilled.
5. `learn.qdistil` turns a gated ranker into a full legal soft target,

   `pi_target proportional to pi_champion * exp(clipped_advantage / beta)`.

   Only `head_w`, `head_b`, `pass_w`, and `pass_b` change.  The trunk is checked
   byte-for-byte against the champion.  A KL projection gate must pass before a
   deployable `.npz` is written.
6. A candidate that passes offline gates receives a fresh paired-board arena
   pilot.  Offline loss or Q accuracy never promotes a bot.

## Fixed pilot budget

Generate one new dataset, disjoint from the earlier `5.1M` development seeds:

```bash
python -m tools.searchprobe \
  --weights /home/vng205/top3-v2-shared/weights/selfplay-champion-gen1.npz \
  --out /local/data/vng205/qrank-data-gen1-20260812 \
  --games 2000 --positions-per-game 2 --topk 6 --rollouts 4 \
  --critic-depth 32 --min-turn 80 --max-turns 1200 \
  --seed0 6000000 --workers 32
```

This is 4,000 observation states and at most 96,000 terminal continuations.
Based on the measured 16-rollout probe throughput, it should take roughly
45-60 CPU minutes.  It is a bounded data-generation run, not a GPU policy run.
Do not tune the game split or gate after seeing the sealed result.

Fit the ranker with the defaults:

```bash
python -m learn.qrank \
  --data /local/data/vng205/qrank-data-gen1-20260812/rows.npz \
  --policy /home/vng205/top3-v2-shared/weights/selfplay-champion-gen1.npz \
  --out /local/data/vng205/qrank-gen1-20260812/qrank.npz
```

Only after the command prints `Q-RANKER GATE PASSED`, distil:

```bash
python -m learn.qdistil \
  --data /local/data/vng205/qrank-data-gen1-20260812/rows.npz \
  --policy /home/vng205/top3-v2-shared/weights/selfplay-champion-gen1.npz \
  --ranker /local/data/vng205/qrank-gen1-20260812/qrank.npz \
  --out /local/data/vng205/qdistil-gen1-20260812/candidate.npz
```

If either command exits nonzero, stop.  Do not loosen a gate on the same sealed
test games.  A materially changed method requires a new seed block.

## Gameplay gates

First run a construction/runtime smoke through `tools.evaluate --games 20`.
Then run a 400-game direct paired-board pilot against the accepted champion on
a fresh seed block.  Continue only if the point score exceeds 0.50, there are
no faults, and the candidate has no material regression against the required
style/safety opponents.  The 400-game pilot is triage, never promotion.

Final promotion uses `tools.evaluate` without `--games`; that consumes a unique
registered development/confirmation attempt and applies the current
`evaluation/top3-v2.json` paired superiority, style, runtime, and fault gates.
Do not copy a candidate over the shared champion until that command reports an
eligible promotion pass.

## Stop rules

- Ranker selection fails: test remains sealed; stop this model.
- Ranker sealed-test lower bound is not positive: stop this data/model recipe.
- Distillation exceeds KL or cannot reproduce changed targets: no policy file
  is written; stop.
- Direct pilot score is at or below 0.50: reject the candidate.
- Full evaluation fails any required bucket: reject the candidate.

These gates make the pipeline usable for improvement without treating a
plausible hypothesis as a promised stronger bot.
