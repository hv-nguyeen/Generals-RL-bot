"""Can the network represent the heuristic at all? Label its own games and see.

Behaviour cloning on ladder replays saturates at ~0.50 top-1 across every
network size from 33k to 270k parameters. Two readings fit that equally well and
only one of them was written down:

  * the model is big enough and the labels are noisy — several moves in a
    position are reasonable and different players pick differently;
  * the model **cannot see the board**. `bot/policy/net.py` is an L-layer stack
    of 3x3 convolutions with no dilation and no pooling, so its receptive-field
    radius is exactly L. The shipped net is 4 layers: a 9x9 window on a 21x21
    board. Everything the heuristic decides on — `dist_home`, `dist_enemy_gen`,
    the distance-ordered defence sums, the threat scan — is a whole-board BFS or
    a global scalar, and none of it exists inside 9x9.

Label noise hides the second. So remove it: the heuristic is deterministic given
its config, so labelling ITS OWN games has exactly zero label noise, and any
shortfall is the model's.

    python -m tools.distil --games 300 --out /local/data/vng205/distil
    python -m learn.train --data /local/data/vng205/distil --layers 4  --channels 32
    python -m learn.train --data /local/data/vng205/distil --layers 12 --channels 128

Read the two held-out top-1 numbers together, not separately:

  radius 4 stalls, radius 12 clears ~0.90   the receptive field is the ceiling.
                                            Go deeper, or add a global path.
  both stall below ~0.7                     the OBSERVATION is the ceiling. No
                                            policy over these 12 channels can
                                            imitate the heuristic; it needs what
                                            `bot/belief.py` remembers.
  both clear ~0.90                          architecture and observation are
                                            fine and the ceiling is training.

A ceiling here is a hard bound on everything downstream: a policy that cannot
imitate the heuristic will not exceed it by self-play either.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arena import agents as agents_mod
from bot import features, rules
from sim import engine, mapgen

SHARD = 40_000          # matches learn/dataset.py so learn/train.py just works


def _one_game(job: tuple) -> tuple:
    """Both seats of one heuristic-vs-heuristic game, as (obs, action) pairs.

    Both seats are labelled: the position is symmetric and each seat's own view
    is a legitimate example, so one game yields twice the data at no cost.
    """
    seed, spec, max_turns = job
    grid = mapgen.generate(seed)
    st = engine.from_grid(grid)
    bots = [agents_mod.make(spec, i, *grid.shape, seed=seed) for i in (0, 1)]

    xs, ys = [], []
    for _ in range(max_turns):
        obs = [engine.observe(st, 0), engine.observe(st, 1)]
        acts = [bots[i].act(obs[i]) for i in (0, 1)]
        for i in (0, 1):
            idx = features.action_to_index(acts[i])
            # A pass carries no positional information and is ~a third of all
            # turns; keeping it would let the net score well by predicting
            # "wait" and tell us nothing about whether it can see the board.
            if idx is None or idx == features.PASS_INDEX:
                continue
            xs.append(features.encode(obs[i]).astype(np.float16))
            ys.append(idx)
        if engine.step(st, acts[0], acts[1]):
            break
    if not xs:
        return None
    return np.stack(xs), np.asarray(ys, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/local/data/vng205/distil")
    ap.add_argument("--spec", default="ours:configs/v16.json",
                    help="the teacher; must be DETERMINISTIC or the point is lost")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=3_000_000,
                    help="disjoint from the arena and curriculum seed blocks")
    ap.add_argument("--max-turns", type=int, default=rules.TURN_LIMIT)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        x, y = _one_game((args.seed0, args.spec, 60))
        assert x.ndim == 4 and x.shape[1] == features.C, x.shape
        assert y.dtype == np.int32 and (y < features.N_ACTIONS).all()
        assert (y != features.PASS_INDEX).all(), "passes must be dropped"
        # determinism is the whole premise: the same seed must relabel identically
        x2, y2 = _one_game((args.seed0, args.spec, 60))
        assert np.array_equal(y, y2) and np.array_equal(x, x2), "teacher is not deterministic"
        print(f"distil selfcheck OK ({len(y)} labels from one 60-turn game, "
              f"teacher deterministic)")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("shard_*.npz"):
        stale.unlink()

    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    jobs = [(args.seed0 + i, args.spec, args.max_turns) for i in range(args.games)]
    xs, ys, buffered, shard, total = [], [], 0, 0, 0

    def flush():
        nonlocal xs, ys, buffered, shard, total
        if not xs:
            return
        np.savez_compressed(out / f"shard_{shard:04d}.npz",
                            x=np.concatenate(xs), y=np.concatenate(ys))
        total += buffered
        shard += 1
        xs, ys, buffered = [], [], 0

    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context("spawn")) as pool:
        for n, r in enumerate(pool.map(_one_game, jobs, chunksize=1), 1):
            if r is None:
                continue
            x, y = r
            xs.append(x)
            ys.append(y)
            buffered += len(y)
            # Whole games stay inside one shard, so learn/train.py's shard-level
            # validation split holds out whole GAMES. Splitting a game across the
            # boundary leaks near-duplicate positions into validation.
            if buffered >= SHARD:
                flush()
            if n % 50 == 0:
                print(f"  {n}/{args.games} games, {total + buffered} labels", flush=True)
    flush()

    # Stamp the encoding these shards were built under. Without it a dataset
    # built before the action space widened trains cleanly against the new one,
    # with every label off by the reindexing -- learn/train.py checks this file
    # and only warns when it is absent.
    (out / "meta.json").write_text(json.dumps(
        {"n_actions": features.N_ACTIONS, "channels": features.C,
         "teacher": args.spec, "games": args.games, "labels": total}, indent=2) + "\n")

    print(f"\n{total} labels from {args.games} games of {args.spec} -> {out}")
    print(f"  {shard} shards; label noise is ZERO by construction (the teacher "
          f"is deterministic)")
    print("\nNow train two depths and compare held-out top-1:")
    print(f"  python -m learn.train --data {out} --layers 4  --channels 32  "
          f"--out /tmp/d4.npz")
    print(f"  python -m learn.train --data {out} --layers 12 --channels 128 "
          f"--out /tmp/d12.npz")
    print("\n  radius 4 stalls and radius 12 clears ~0.90 -> the receptive field")
    print("  is the ceiling. Both below ~0.7 -> the observation is. Both ~0.90 ->")
    print("  neither, and the ceiling is training.")


if __name__ == "__main__":
    main()
