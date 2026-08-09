# generals.bot competition bot

Read `README.md` first, then `docs/superpowers/specs/2026-08-03-generals-bot-design.md`
for the rules analysis. `third_party/generals-bots` is the official starter kit —
it is the ground truth for every rule; read the engine source, not the rules page.

**Start with `docs/TOP3-V2-HANDOFF.md`**, then `docs/STATE.md` for the measured
history and non-starters. `docs/CLUSTER.md` is how to install and run on the VU
box.

Three things that cost the most time when forgotten:

* **A number at n<=36 is noise.** Ladder SE is ±58 there and ±22 at n=240. Four
  separate conclusions were drawn and later reversed off small ladder samples,
  including "loses early, wins late", which was a 43-game artifact.
* **Measure a candidate before building it.** Six signals were checked against
  ladder replays on 2026-08-05 and four were rejected in two minutes each; the
  survivors separate wins from losses by >2x, the rejects by 1.0-1.5x. The same
  discipline killed a guard trigger that looked certain and fired on 98% of turns.
* **Hand-written overrides lose to this policy.** Five measured, all neutral or
  negative, the worst at -322 Elo. It has outgrown being told what to do.

**Before proposing any ML or evaluator work, read `docs/ml-log.md`.** Six ML
attempts and nine local instruments have failed, each for a specific measured
reason. The log exists so none of them gets tried a seventh time.

## Non-negotiables

- **Never claim a change is an improvement unless the versioned promotion suite
  and a fresh direct incumbent comparison pass.** The runner/evaluator use
  paired fixed-sample evidence; do not call new output an SPRT. Use 2000 fresh
  games per bucket, inspect the pair-aware lower bound, require zero faults and
  runtime margin, and measure `.best` rather than `.live`.
- **`bot/` must stay self-contained and numpy-only.** It is the submission. It
  may not import `sim/`, `arena/`, or `analysis/`.
- **`sim/engine.py` mirrors the official engine exactly.** If you touch it, run
  `make verify`. JAX clamps out-of-range indices where numpy wraps — that is
  why every action-derived index is clipped by hand.
- **Heuristic tunables live in `bot/config.py`; learned architecture is recorded
  inside every checkpoint.** Config files are strict and versioned.

## Where things are

`bot/policy/net.py` is the submitted policy when `bot/weights.npz` exists;
`bot/policy/controller.py` is the fallback and heuristic opponent.
`bot/memory.py` is the neural temporal state, while `bot/belief.py` remains the
richer heuristic belief. `tools.evaluate.py` owns promotion.

## Current training split

The current incumbent is the C=40/context continuation checkpoint
`runs/top3-v2/incumbent-refresh-onpolicy.npz` with its matching critic. Node 1's
`learn.selfplay` run is a mirror-self-play/economy arm; its `--opp` list is
measurement-only. Node 2's `learn.league --oracle net` run is the adversarial
mixture arm; its `clone:` archive members affect training. Do not confuse a high
win rate against weak heuristic opponents with a higher ceiling, and do not
replace the incumbent before fresh direct tests. The `ceiling-r2` pilot completed
and failed its gate (`0.539` trained vs `0.504` init; `+0.071` required), so it is
not a candidate.
