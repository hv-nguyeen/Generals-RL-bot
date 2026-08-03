"""Does a local instrument agree with the ladder? Test it before trusting it.

Today produced four changes that local evaluation approved and the ladder
rejected, one of them backed by two independent local instruments that agreed
with each other. The missing step was never a better instrument — it was
checking any instrument against known ground truth before using it.

We have that ground truth: five builds with measured ladder Elo. A usable
instrument must rank them correctly. This plays each build locally, scores the
positions it reaches with the field-trained win-probability model, and reports
the rank correlation against the ladder.

Read the verdict, not the numbers:

  rho > 0.8   the model tracks real strength; use it as a search objective
  rho ~ 0     no better than what we already have; do not use it
  rho < 0     actively misleading, which is what our arena turned out to be

    python -m tools.calibrate --model /local/data/vng205/value.npz --workers 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from arena import agents as agents_mod
from bot.policy.net import ValueNet
from sim import engine, mapgen

# Measured on the leaderboard. Sample sizes differ, so the ordering is more
# trustworthy than the gaps: v5 and v10 are 1.6 sigma apart, v5 and v12 are 3.6.
# NOTE these must be configs that still reconstruct the build under the current
# code. Config.from_dict fills absent keys from today's defaults, so a config
# saved before a key existed silently becomes a different bot — configs/v5.json
# predates lock_enabled and would now load with the lock ON, i.e. as v6.
LADDER = [
    ("configs/v13.json", 1737, "v5/v13  thrust 20/0.10, no lock"),
    ("configs/v9.json", 1651, "v10     + general-drain fix"),
    ("configs/v12.json", 1496, "v12     thrust 8/0.04"),
    ("configs/v7.json", 1482, "v7      lock on, thrust gated"),
    ("configs/v6.json", 1467, "v6      lock on, thrust ungated"),
]


def score_build(cfg_path: str, model: ValueNet, opponent: str, games: int,
                lo: int, hi: int, stride: int) -> tuple[float, int]:
    """Mean win probability of the positions this build reaches."""
    spec = f"ours:{cfg_path}"
    vals = []
    for seed in range(games):
        grid = mapgen.generate(50_000 + seed)
        st = engine.from_grid(grid)
        me = agents_mod.make(spec, 0, *grid.shape)
        opp = agents_mod.make(opponent, 1, *grid.shape)
        for t in range(hi):
            o0 = engine.observe(st, 0)
            if t >= lo and t % stride == 0:
                vals.append(model.win_prob(o0))
            if engine.step(st, me.act(o0), opp.act(engine.observe(st, 1))):
                break
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def spearman(a: list[float], b: list[float]) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/local/data/vng205/value.npz")
    ap.add_argument("--opponent", default="hunter:3")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--lo", type=int, default=60)
    ap.add_argument("--hi", type=int, default=400)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    model = ValueNet(args.model)
    rows = []
    print(f"scoring {len(LADDER)} builds, {args.games} games each vs {args.opponent}, "
          f"turns {args.lo}-{args.hi}\n")
    print(f"{'build':<40}{'ladder':>8}{'mean P(win)':>14}{'positions':>11}")
    for cfg, elo, label in LADDER:
        if not Path(cfg).exists():
            print(f"{label:<40}{elo:>8}   missing {cfg}")
            continue
        v, n = score_build(cfg, model, args.opponent, args.games,
                           args.lo, args.hi, args.stride)
        rows.append((label, elo, v))
        print(f"{label:<40}{elo:>8}{v:>14.4f}{n:>11}")

    # Builds that play identically inflate the correlation with duplicate rows.
    # It happens easily: v10's change was in the code, not the config, so its
    # config is indistinguishable from v13's under the current tree; and v7's
    # thrust gate never fires against this opponent, making it v6.
    seen: dict[tuple[float, int], str] = {}
    unique = []
    for label, elo, v in rows:
        key = (round(v, 6), 0)
        if key in seen:
            print(f"\n  ! {label!r} played identically to {seen[key]!r} — "
                  f"not a distinct build, dropped from the correlation")
            continue
        seen[key] = label
        unique.append((label, elo, v))
    rows = unique

    if len(rows) < 3:
        raise SystemExit("\nneed at least three DISTINCT builds to calibrate")

    spread = max(r[2] for r in rows) - min(r[2] for r in rows)
    print(f"\n  spread across builds: {spread:.4f}")
    if spread < 0.05:
        print("  ! the model barely separates builds that are hundreds of elo "
              "apart; the ordering below is close to noise")

    rho = spearman([r[1] for r in rows], [r[2] for r in rows])
    print(f"\nrank correlation with the ladder: rho = {rho:+.2f}")
    if rho > 0.8:
        print("  USABLE - the model tracks real strength. Optimise against it.")
    elif rho > 0.3:
        print("  WEAK - some signal, but not enough to ship changes on alone.")
    elif rho > -0.3:
        print("  NO SIGNAL - it cannot tell these builds apart. Do not use it.")
    else:
        print("  INVERTED - actively misleading, like the arena was today.")
    print("\nFor reference, the arena ranked v12 above v5; the ladder has them "
          "241 elo apart the other way.")


if __name__ == "__main__":
    main()
