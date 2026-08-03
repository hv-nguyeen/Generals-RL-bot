"""Competition map generation.

Reproduces the *procedure* of `generals/core/grid.py::generate_grid` under the
"competition" preset — not its JAX RNG stream, which would be pointless since
the evaluator's seeds are unknown. What matters is that the distribution
matches, in particular the two spawn constraints, because the bot's
enemy-general prior is derived from them:

  1. terrain first: 24–26% of the area is mountains, placed uniformly;
  2. 9–11 castles are carved out of mountains — under the build-castles ruleset
     they are then stripped to plain, so they act as tunnels through ridges;
  3. general A is drawn from 8 candidates biased toward the carved cells;
  4. general B is drawn uniformly among cells that are at least 17 BFS steps
     from A *and* whose `room7` count is within 5 of A's.
"""

from __future__ import annotations

import numpy as np

from bot import rules
from bot.board import bfs_field_from, dilate4, room_field

SPAWN_CANDIDATES = 8


def generate(seed: int) -> np.ndarray:
    """A competition-legal grid: -2 mountain, 0 plain, 1 general A, 2 general B."""
    rng = np.random.default_rng(seed)
    h = int(rng.integers(rules.MIN_SIDE, rules.MAX_SIDE + 1))
    w = int(rng.integers(rules.MIN_SIDE, rules.MAX_SIDE + 1))
    area = h * w

    lo = int(np.floor(rules.MOUNTAIN_DENSITY[0] * area))
    hi = int(np.floor(rules.MOUNTAIN_DENSITY[1] * area))
    n_mountains = int(rng.integers(lo, hi + 1))
    n_castles = int(rng.integers(rules.NUM_CARVED_CASTLES[0], rules.NUM_CARVED_CASTLES[1] + 1))

    order = rng.permutation(area)
    mountains = np.zeros(area, dtype=bool)
    mountains[order[:n_mountains]] = True
    # Castles are carved out of mountains, then stripped to plain by the ruleset.
    carved_idx = order[:min(n_castles, n_mountains)]
    mountains[carved_idx] = False

    mountains = mountains.reshape(h, w)
    passable = ~mountains
    carved = np.zeros((h, w), dtype=bool)
    carved.flat[carved_idx] = True

    room = room_field(passable, rules.SPAWN_ROOM_RADIUS)

    # Candidates for general A, biased toward the carved-open pockets.
    near = carved.copy()
    for _ in range(6):
        near = dilate4(near) & passable
    pool = np.argwhere(near)
    if len(pool) < SPAWN_CANDIDATES:
        pool = np.argwhere(passable)
    pick_idx = rng.choice(len(pool), size=min(SPAWN_CANDIDATES, len(pool)), replace=False)
    candidates = [tuple(int(x) for x in pool[i]) for i in pick_idx]

    inf = h * w
    best = None
    for rank_order, cand in enumerate(candidates):
        field = bfs_field_from(passable, cand)
        reach = (field < inf) & passable
        far = reach & (field >= rules.MIN_GENERALS_DISTANCE)
        gap = np.abs(room - int(room[cand]))
        fair = far & (gap <= rules.SPAWN_ROOM_TOLERANCE)

        has_fair = bool(fair.any())
        has_far = bool(far.any())
        best_gap = int(gap[far].min()) if has_far else inf
        span = int(field[reach].max()) if reach.any() else -1
        rank = (2 if has_fair else 0) + (1 if has_far else 0)
        key = rank * 100_000 - best_gap * 100 + span - rank_order
        if best is None or key > best[0]:
            best = (key, cand, fair, far, reach)

    _, gen_a, fair, far, reach = best
    options = np.argwhere(fair) if fair.any() else (np.argwhere(far) if far.any() else np.argwhere(reach))
    options = options[[not np.array_equal(o, gen_a) for o in options]] if len(options) > 1 else options
    gen_b = tuple(int(x) for x in options[rng.integers(len(options))])

    grid = np.where(mountains, -2, 0).astype(np.int32)
    grid[gen_a] = 1
    grid[gen_b] = 2
    return grid


def dims(grid: np.ndarray) -> tuple[int, int]:
    return int(grid.shape[0]), int(grid.shape[1])
