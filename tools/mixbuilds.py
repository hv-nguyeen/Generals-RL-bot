"""Give the clone a castle-building prior it cannot learn on its own.

Castles are worth **+111 Elo [76.5, 148.5]** to the heuristic — `configs/v16.json`
against the same config with `castle_enabled=False`, 262W-138L over 400 games.
That is the largest single identified improvement in the project.

The policy cannot reach it by exploration. A build is legal on 19.5% of turns and
the trained net puts 0.0007 probability on one, so it samples a build about once
every nine games and the payoff arrives ~70 turns later diluted among hundreds of
other actions. Terminal-only reward cannot assign credit through that; the prior
has to come from the initialisation.

But the ladder-replay dataset barely contains builds — 34 labels in 204k, 0.02% —
almost certainly because `analysis/actions.py` recovers only ~66% of player ticks
and a build is hard to infer from a state diff. The heuristic's own games have
0.19%, ten times as many, from the same encoder.

So: keep the ladder data as the bulk, and splice in ONLY the build frames from the
distilled set, repeated until they are a few percent of the whole. Distilled data
wholesale is not an option — a clone trained purely on it reaches 0.651 top-1 and
then loses 198-2 to the bot it was copying. This takes the one behaviour that set
is uniquely good at and leaves the rest alone.

    python -m tools.distil --games 2000 --out /local/data/vng205/distil
    python -m tools.mixbuilds --base /local/data/vng205/bc20 \\
        --builds /local/data/vng205/distil --out /local/data/vng205/bc20mix
    python -m learn.train --data /local/data/vng205/bc20mix --layers 8 --channels 32 \\
        --out /local/data/vng205/clone20-build.npz

Then check the prior actually took, because that is the whole point:

    build legal on N turns, mean prob on build when legal

0.0007 is what the unmixed clone gives. Anything above ~0.02 is enough for a
build to be sampled a few times a game, which is where credit assignment starts
working.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bot import features

BUILD_SLOT = features.PER_CELL - 1     # the 9th per-cell action; see bot/features.py


def build_rows(shard: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(shard)
    y = z["y"]
    keep = (y % features.PER_CELL) == BUILD_SLOT
    return features.ensure_channels(z["x"][keep]), y[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="ladder-replay shards; the bulk")
    ap.add_argument("--builds", required=True, help="distilled shards to mine builds from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-frac", type=float, default=0.03,
                    help="build labels as a share of the mixed set. 0.03 puts a "
                         "build in every ~33 examples, which is enough to move "
                         "the prior without the clone learning to build always")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    base, builds, out = Path(args.base), Path(args.builds), Path(args.out)
    base_shards = sorted(base.glob("shard_*.npz"))
    build_shards = sorted(builds.glob("shard_*.npz"))
    if not base_shards or not build_shards:
        raise SystemExit(f"need shards in both {base} and {builds}")

    # The two sets must agree on the encoding or every spliced label is wrong.
    for d in (base, builds):
        m = d / "meta.json"
        if m.exists():
            j = json.loads(m.read_text())
            for k, want in (("n_actions", features.N_ACTIONS), ("channels", features.C)):
                if j.get(k) is not None and int(j[k]) != want:
                    raise SystemExit(f"{d} has {k}={j[k]}, this build has {want}")
        else:
            print(f"WARNING no {m}: cannot verify {d} matches this encoding")

    bx, by = [], []
    for s in build_shards:
        x, y = build_rows(s)
        if len(y):
            bx.append(x)
            by.append(y)
    if not by:
        raise SystemExit(f"no build labels in {builds} — is the teacher building at all?")
    bx, by = np.concatenate(bx), np.concatenate(by)
    print(f"{len(by)} build frames mined from {len(build_shards)} shards")

    n_base = sum(len(np.load(s)["y"]) for s in base_shards)
    # solve  reps*B / (n_base + reps*B) = f  ->  reps = f*n_base / (B*(1-f))
    reps = max(1, round(args.target_frac * n_base / (len(by) * (1 - args.target_frac))))
    print(f"{n_base} base labels; repeating builds {reps}x -> "
          f"{100 * reps * len(by) / (n_base + reps * len(by)):.2f}% of the mix")

    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("shard_*.npz"):
        stale.unlink()

    # NEVER splice into the shards learn/train.py holds out. Two failures
    # otherwise, and this project has already paid for the first: validation
    # would contain spliced frames and so could not detect that the splice had
    # failed, and at reps >= 2 the SAME frame lands in both training and
    # validation. The split is a fixed permutation, so it is reproducible here.
    from learn.train import VAL_SHARDS, VAL_SPLIT_SEED
    pick = np.random.default_rng(VAL_SPLIT_SEED).permutation(len(base_shards))
    held = set(pick[:VAL_SHARDS].tolist())
    targets = [i for i in range(len(base_shards)) if i not in held]
    if not targets:
        raise SystemExit(f"{len(base_shards)} shards is not enough to hold "
                         f"{VAL_SHARDS} out and still splice")
    per = int(np.ceil(reps * len(by) / len(targets)))
    rng = np.random.default_rng(0)
    order = rng.permutation(np.tile(np.arange(len(by)), reps))
    cut = 0
    for i, s in enumerate(base_shards):
        z = np.load(s)
        sel = order[cut:cut + per] if i in set(targets) else order[:0]
        if i in set(targets):
            cut += per
        base_x = features.ensure_channels(z["x"])
        x = np.concatenate([base_x, bx[sel]]) if len(sel) else base_x
        y = np.concatenate([z["y"], by[sel]]) if len(sel) else z["y"]
        p = rng.permutation(len(y))
        np.savez_compressed(out / f"shard_{i:04d}.npz", x=x[p], y=y[p])
    (out / "meta.json").write_text(json.dumps(
        {"n_actions": features.N_ACTIONS, "channels": features.C,
         "base": str(base), "builds": str(builds), "build_reps": reps,
         "labels": int(n_base + reps * len(by))}, indent=2) + "\n")

    got = 0
    tot = 0
    for s in sorted(out.glob("shard_*.npz")):
        y = np.load(s)["y"]
        tot += len(y)
        got += int(((y % features.PER_CELL) == BUILD_SLOT).sum())
    print(f"\nwrote {tot} labels to {out}, {100 * got / tot:.2f}% builds "
          f"(was {100 * 34 / 203924:.2f}% in the base)")
    print(f"  shards {sorted(held)} left UNTOUCHED - they are learn/train.py's "
          f"validation split, and splicing into them is how the previous "
          f"attempt hid its own failure")


if __name__ == "__main__":
    main()
