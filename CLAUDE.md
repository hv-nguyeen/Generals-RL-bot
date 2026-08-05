# generals.bot competition bot

Read `README.md` first, then `docs/superpowers/specs/2026-08-03-generals-bot-design.md`
for the rules analysis. `third_party/generals-bots` is the official starter kit —
it is the ground truth for every rule; read the engine source, not the rules page.

**Start with `docs/STATE.md`** — current standing, what is running, what to do
next, and the measured non-starters. `docs/CLUSTER.md` is how to install and run
on the VU box.

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

- **Never claim a change is an improvement without an SPRT run.** The standard
  error on 60 games is ~80 Elo. `python -m arena.runner --a <new> --b <old>
  --games 400` and read the verdict line.
- **`bot/` must stay self-contained and numpy-only.** It is the submission. It
  may not import `sim/`, `arena/`, or `analysis/`.
- **`sim/engine.py` mirrors the official engine exactly.** If you touch it, run
  `make verify`. JAX clamps out-of-range indices where numpy wraps — that is
  why every action-derived index is clipped by hand.
- **Every tunable lives in `bot/config.py`.** No magic numbers in the policy;
  the tuner searches whatever is in that dataclass.

## Where things are

`bot/policy/controller.py` is the bot. `bot/belief.py` is the memory and the
enemy-general prior. `tools/sweep.py` is the fastest iteration loop.
