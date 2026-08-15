# New-bot design: ensemble of champion-equal diverse nets — 2026-08-15

## Situation (measured this session)

The 8×32 champion is at its **architecture ceiling**: both training arms
(`learn.league --oracle net` continuation, `learn.selfplay` mirror curriculum)
land on a statistical tie vs the champion — the self-play `.best` gated
**984W-984L, elo 0.0 [-10,+10]**. Topology teaching was harmful; the frontier
perception plane was neutral. Global-context / receptive-field capacity was
already built (`_context_mix`, `_strategy_mix`) and a global/transformer model
was tried before with "no gain under matched compute; inference violates budget"
(`NEXT-TRAINING-AUDIT` L365). So neither more iterations nor another global net
is the lever.

## The design: a guarded ENSEMBLE (no training)

An ensemble is a *different bot*, not more of the same 8×32 training. Evidence it
should help:

* TTA (averaging one net over its 8 symmetry views — maximally correlated) is a
  measured **+31 Elo** for zero training (`bot/policy/ensemble.py` docstring).
* Independent checkpoints are *less* correlated than symmetry views, so each adds
  variance reduction the symmetry group cannot reach.
* We now have **champion-EQUAL, differently-trained members** (champion;
  `selfplay-best` = exactly 984-984; `topo-night-best` = +20 tie) — the near-peer
  regime where averaging classically beats the best single member. This is a
  better regime than when the ensemble code was written (members were then ~30
  Elo below champion and shared a castle defect).

Budget: 3 members ≈ 36 ms « 150 ms. Guard-wrapped (`shipens:`) so it is exactly
what `bot/main.py` would package. Infra exists and self-checks pass.

Risk: the three members are all champion-*derived*, so partly correlated — the
gain may be small or nil. That is why the plan below is a five-config bake-off
(rank), then a 2000-game confirmation of the leader, then an arch-diverse
fallback.

## Validation (the one manual, cluster-only step)

Node-local paths: `CH=/local/data/vng205/champion.npz`,
`SP=/local/data/vng205/champ-selfplay/selfplay.best.npz`,
`TN=/local/data/vng205/topo-night-best.npz`. On the topology build + gb312 env:

```bash
for spec in \
  "shipens:${CH}+${SP}+${TN}@logit" "shipens:${CH}+${SP}+${TN}@prob" \
  "shipens:${CH}+${SP}@logit" "shipens:${CH}+${TN}@logit" "shipens:${SP}+${TN}@logit"; do
  python -m arena.runner --a "$spec" --b "ship:${CH}" \
    --games 800 --workers 32 --time-limit-ms 150 --quiet | grep -iE "elo|fault"
done
```

* Any config with **`elo_lo > 0` and 0 faults** → confirm at `--games 2000` →
  that is a measured stronger bot; package it (`ens:`/`shipens:` list, no TTA if
  the budget is tight).
* All straddle 0 → **arch-diverse ensemble** `shipens:${CH}+cap16x96-best.npz`
  under the residual-tail build (different architectures decorrelate most — the
  strongest ensemble prior). Requires the `generals-bot-restail` build up.

## Fallback idea menu (ranked, if ensembles do not clear)

1. Gate the 16×96 SOLO vs champion (capacity, never directly gated; fast, fits
   budget).
2. Distill 36×64 (comp-eval 0.724, likely stronger) → a budget net.
3. Behaviour-clone from strong ladder/human replays → lift the imitation floor
   the whole self-play climb builds on.
4. Proper exploiter league (main-exploiters + PFSP) — the plateau-breaker;
   multi-day, fresh session.

Champion `selfplay-champion-gen1` remains accepted until a candidate's paired
lower bound clears it with zero faults and runtime margin.
