"""Do the board's symmetries disagree, and do the transposing ones disagree MORE?

Test-time augmentation averages the policy over the dihedral group and measured
+31.2 Elo on the shipped configuration. It currently withholds the four
shape-CHANGING elements on non-square boards, which is most boards -- `mapgen`
draws h and w independently from 18-21, so only about one in four is square.

Before spending an arena run on completing the group, this asks the question the
arena cannot: is a transposed view a NOISIER view of the same position, or a
BIASED one? Averaging in noise helps. Averaging in bias hurts.

Reports, per group element, how often its mapped-back argmax matches the
identity's, split by whether the element preserves the board's shape.

    agreement(g=2..5) ~ agreement(g=1,6,7)   -> noise, complete the group
    agreement(g=2..5) << agreement(g=1,6,7)  -> systematic orientation bias, stop

The overall disagreement rate is worth reading on its own: it bounds how much
ANY averaging scheme can be worth. If the orientations already agree on 99% of
moves, no weighting or softmax variant is worth an arena slot.

    python -m tools.ttaprobe --net runs/nn/sp16-c24.npz --games 40
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from bot import features, rules, symmetry
from bot.policy.net import Net
from sim import engine, mapgen


def probe(net_path: str, games: int, every: int, seed0: int) -> None:
    net = Net(net_path)
    agree = defaultdict(lambda: [0, 0])          # g -> [matches, total]
    shapes = defaultdict(int)
    boards = 0

    for s in range(seed0, seed0 + games):
        grid = mapgen.generate(s)
        st = engine.from_grid(grid)
        h, w = mapgen.dims(grid)
        shapes[(h, w)] += 1
        boards += 1
        for t in range(0, 400):
            if engine.step(st, rules.PASS_ACTION, rules.PASS_ACTION):
                break
            if t % every:
                continue
            obs = engine.observe(st, 0)
            mask = features.legal_mask(obs)
            if not mask.any():
                continue
            enc = features.encode(obs)
            flat = enc.reshape(enc.shape[0], -1)
            base = int(np.argmax(np.where(mask, net._logits_from(enc), -np.inf)))
            for g in symmetry.FULL[1:]:
                srcof, actmap = symmetry.maps(obs.H, obs.W, g)
                out = net._logits_from(flat[:, srcof].reshape(enc.shape))[actmap]
                got = int(np.argmax(np.where(mask, out, -np.inf)))
                a = agree[g]
                a[0] += got == base
                a[1] += 1

    square = sum(n for (h, w), n in shapes.items() if h == w)
    print(f"{boards} boards, {square} square ({square / boards:.0%}), "
          f"{agree[1][1]} positions\n")
    print("  g  kind             agrees with identity")
    keep, drop = [], []
    for g in symmetry.FULL[1:]:
        m, n = agree[g]
        r = m / max(n, 1)
        kind = "shape-preserving" if g in symmetry.SHAPE_PRESERVING else "TRANSPOSES"
        (keep if g in symmetry.SHAPE_PRESERVING else drop).append(r)
        print(f"  {g}  {kind:<16} {r:.3f}")

    k, d = float(np.mean(keep)), float(np.mean(drop))
    print(f"\n  shape-preserving mean {k:.3f}")
    print(f"  transposing mean     {d:.3f}   gap {k - d:+.3f}")
    print(f"\n  orientations disagree with the identity on "
          f"{1 - (k * len(keep) + d * len(drop)) / 7:.1%} of moves -- that is the "
          f"ceiling on what ANY averaging scheme can change")
    if d < k - 0.05:
        print("\n  TRANSPOSES ARE BIASED, NOT NOISY. Completing the group averages "
              "in a systematically different view; do not ship it.")
    else:
        print("\n  Transposing elements look like more of the same noise. "
              "Completing the group is worth an arena run.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--every", type=int, default=25, help="sample every Nth turn")
    ap.add_argument("--seed0", type=int, default=900_000)
    args = ap.parse_args()
    probe(args.net, args.games, args.every, args.seed0)


if __name__ == "__main__":
    main()
