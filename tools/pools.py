"""Prebuilt curriculum map pools: one .npz per stage, numpy only, no JAX.

    python -m tools.pools --out runs/pools --workers 32
    python -m tools.pools --selfcheck

WHY THIS EXISTS AT ALL, since `sim.mapgen.generate` is already 3.2 ms a board:
a vectorised rollout cannot call it. `generate` is a python loop over BFS
fields; the GPU path needs N boards as ONE `(N, 21, 21) int32` array, resident,
with the same shape at every curriculum stage so that swapping stages is an
argument change and not an XLA retrace. Building that on the host, once, and
caching it to disk is the whole job.

WHY NOT the starter kit's own `GeneralsEnv` pool. Three of the five documented
blockers in `learn/rlenv.py`'s neighbourhood are properties of that pool rather
than of the problem:

  * every environment's `pool_idx` starts at 0 (`generals/core/game.py:139`) and
    increments identically (`env.py:339`), so 512 parallel envs replay ONE
    environment's worth of boards;
  * `mode="competition"` sets `min_generals_distance=17` from its preset and
    never touches `max`, so `GeneralsEnv(mode="competition",
    max_generals_distance=6)` is min=17/max=6, which empties the candidate set
    and drops into a `farthest` fallback (`grid.py:356-358`) with no error --
    a curriculum built the obvious way gets silently wrong maps;
  * both distances are `static_argnames` (`grid.py:142`), so K bands are K sets
    of compiled kernels.

None of that can happen here. We index the pool ourselves so `pool_idx` is
never read; `_MODE_PRESETS` is never constructed; and every stage is the same
`(N, 21, 21)` array. The distance is MEASURED per board by
`mapgen.generals_distance`, not requested and hoped for, which is the same
discipline `learn/selfplay.selfcheck` already applies -- `generate`'s bracket is
a request and its fallback chain will seat a general outside it rather than
fail.

Boards are byte-identical to the ones the CPU rollout plays, so a CPU/GPU A/B
compares two backends on the same maps rather than on two distributions.

File format, per stage:
    grid  (N, 21, 21) int32   -2 mountain, 0 plain, 1 general A, 2 general B,
                              padded bottom/right with -2 exactly as the
                              engine's own generator pads
    h, w  (N,)        int32   the real board inside that pad
    dist  (N,)        int32   MEASURED BFS distance between the generals
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bot import features
from learn.selfplay import SP_POOL_SEED0, STAGES
from sim import mapgen

PAD = features.PAD
DEFAULT_SIZE = 32768        # 58 MB of grids a stage; ~9 replays of each board
                            # over a 1200 x 256 run, and 160 MB as device state
IN_RANGE_MIN = 0.95         # same bar selfplay.selfcheck holds `generate` to


def pool_seed(stage: int, i: int, size: int = DEFAULT_SIZE) -> int:
    """Board seed for one entry. Stages get disjoint runs so a stage's boards
    are never another stage's boards."""
    return SP_POOL_SEED0 + stage * size + i


def pool_seed_span(size: int = DEFAULT_SIZE) -> int:
    """One past the highest seed any pool would use. `selfplay.selfcheck`
    compares it against the other four blocks' origins."""
    return pool_seed(len(STAGES) - 1, size - 1, size) + 1


def _one(args) -> tuple[np.ndarray, int, int, int]:
    stage, i, size = args
    dmin, dmax, _ = STAGES[stage]
    grid = mapgen.generate(pool_seed(stage, i, size), dmin, dmax)
    h, w = grid.shape
    padded = np.full((PAD, PAD), -2, dtype=np.int32)
    padded[:h, :w] = grid
    return padded, h, w, mapgen.generals_distance(grid)


def build(stage: int, size: int = DEFAULT_SIZE, workers: int = 1) -> dict:
    """One stage's pool as plain arrays. ~105 s single-core at the default size."""
    jobs = [(stage, i, size) for i in range(size)]
    runner = (map(_one, jobs) if workers <= 1 else
              ProcessPoolExecutor(max_workers=workers).map(_one, jobs, chunksize=64))
    grid = np.empty((size, PAD, PAD), dtype=np.int32)
    h = np.empty(size, dtype=np.int32)
    w = np.empty(size, dtype=np.int32)
    dist = np.empty(size, dtype=np.int32)
    for i, (g, hh, ww, d) in enumerate(runner):
        grid[i], h[i], w[i], dist[i] = g, hh, ww, d
    return {"grid": grid, "h": h, "w": w, "dist": dist}


def in_range(stage: int, dist: np.ndarray) -> float:
    dmin, dmax, _ = STAGES[stage]
    hi = dmax if dmax is not None else 10 ** 9
    return float(((dist >= dmin) & (dist <= hi)).mean())


def path_for(out: Path | str, stage: int) -> Path:
    return Path(out) / f"stage{stage}.npz"


def load(out: Path | str, stage: int) -> dict:
    """Read one stage back. Raises rather than regenerating: a pool that is
    silently rebuilt with different code is a pool nobody can reproduce."""
    p = path_for(out, stage)
    if not p.exists():
        raise SystemExit(f"no {p} — run `python -m tools.pools --out {out}` first")
    z = np.load(p)
    return {k: z[k] for k in ("grid", "h", "w", "dist")}


def write(out: Path | str, stages=None, size: int = DEFAULT_SIZE,
          workers: int = 1) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for stage in (range(len(STAGES)) if stages is None else stages):
        t0 = time.time()
        pool = build(stage, size, workers)
        frac = in_range(stage, pool["dist"])
        dmin, dmax, _ = STAGES[stage]
        # The measured distance is the only proof a stage reached the generator.
        # Boards outside the band are KEPT, not filtered: every stage must stay
        # the same shape or the device pool retraces on promotion, and at 100%
        # in range (which is what every stage measures today) the question is
        # moot anyway.
        assert frac >= IN_RANGE_MIN, (
            f"stage {stage} ({dmin}-{dmax}): only {frac:.2f} of boards in range")
        np.savez_compressed(path_for(out, stage), **pool)
        print(f"  stage {stage} {dmin}-{dmax}: {size} boards, {frac:.0%} in range, "
              f"mean dist {pool['dist'].mean():.1f}, "
              f"sides {pool['h'].min()}-{pool['h'].max()}, {time.time() - t0:.0f}s",
              flush=True)


def selfcheck() -> None:
    """Everything provable without a GPU, a game or a full pool. ~0.3 s."""
    size = 16
    for stage in (0, len(STAGES) - 1):
        pool = build(stage, size)
        dmin, dmax, _ = STAGES[stage]
        assert pool["grid"].shape == (size, PAD, PAD), pool["grid"].shape
        for i in range(size):
            g, h, w = pool["grid"][i], int(pool["h"][i]), int(pool["w"][i])
            # the padded copy IS the generator's board, and the pad is mountain
            want = mapgen.generate(pool_seed(stage, i, size), dmin, dmax)
            assert np.array_equal(g[:h, :w], want), (stage, i)
            assert (g[h:, :] == -2).all() and (g[:, w:] == -2).all(), (stage, i)
            assert int(pool["dist"][i]) == mapgen.generals_distance(want), (stage, i)
            assert (g == 1).sum() == 1 and (g == 2).sum() == 1, (stage, i)
        frac = in_range(stage, pool["dist"])
        assert frac >= IN_RANGE_MIN, (stage, frac)
        print(f"  pool stage {stage} ({dmin}-{dmax}): {frac:.0%} in range, "
              f"mean dist {pool['dist'].mean():.1f}")

    # stage seed runs are disjoint, which is what stops two stages sharing boards
    seen = set()
    for stage in range(len(STAGES)):
        s = {pool_seed(stage, i, DEFAULT_SIZE) for i in (0, DEFAULT_SIZE - 1)}
        assert not (s & seen), stage
        seen |= s
    assert pool_seed_span() == SP_POOL_SEED0 + len(STAGES) * DEFAULT_SIZE
    print("pools selfcheck OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/pools")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--stage", type=int, default=None, help="one stage, else all")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    write(args.out, None if args.stage is None else [args.stage],
          args.size, args.workers)


if __name__ == "__main__":
    main()
