"""Nudge a checkpoint's BUILD logit so self-play can SAMPLE castles at all.

Not a preference, an exploration fix. A behaviour-cloned policy builds nothing --
BUILD is one slot in nine per cell and rare enough in the data that argmax never
picks it -- and a policy that never samples an action gets no gradient for it.
sp46 ran 50 iterations with `bldA +0.00` on every single one: not a negative
advantage, NO advantage, because zero builds occurred.

Castles pay past ~17 tiles (measured: `bld` 0.00-0.02 at stage 3, 1.2-2.4 at
stage 4/5) and castle rate has tracked Elo across every checkpoint measured this
cycle. A policy that cannot sample them cannot discover that.

This raises `head_b` at the build slot on the INITIALISATION, then lets PPO
decide. That is the difference from `tools/buildprior.py`, which biased the
policy at inference time and was rejected on the ladder: here the bias is a
starting point the gradient is free to undo, and it will if building is bad.

The bias is in logit units on a head whose spread is typically a few units, so
+1.0 is a nudge and +3.0 is close to insisting. Start low.

    python -m tools.buildbias --net bc-8x64.npz --out bc-8x64-b1.npz --bias 1.0
"""

from __future__ import annotations

import argparse

import numpy as np

from bot import features
from bot.policy.net import arch_of


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bias", type=float, default=1.0)
    args = ap.parse_args()

    z = np.load(args.net)
    out = {k: z[k] for k in z.files}
    if "head_b" not in out:
        raise SystemExit(f"{args.net} has no head_b: not a policy checkpoint")
    hb = np.asarray(out["head_b"], np.float32).copy()
    if hb.shape != (features.PER_CELL,):
        raise SystemExit(f"head_b is {hb.shape}, expected {(features.PER_CELL,)}")

    before = float(hb[features.BUILD_OFFSET])
    hb[features.BUILD_OFFSET] = before + args.bias
    out["head_b"] = hb
    np.savez(args.out, **out)

    spread = float(np.abs(hb - hb.mean()).max())
    print(f"{arch_of(z)}  build slot {features.BUILD_OFFSET} of "
          f"{features.PER_CELL}: {before:+.4f} -> {hb[features.BUILD_OFFSET]:+.4f}")
    print(f"  head_b spread {spread:.3f} -- a bias much larger than this is "
          f"insisting, not nudging")
    print(f"wrote {args.out}")
    print("Every other weight is untouched, so the policy is unchanged except in "
          "how often it TRIES a build. If PPO drives bld back to 0.00, that is a "
          "real answer and not a failed experiment.")


if __name__ == "__main__":
    main()
