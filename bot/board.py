"""Grid primitives: 4/8-connected dilation and multi-source BFS distance fields.

Everything here works on plain numpy boolean/int arrays of shape (H, W) and is
shared by the bot, the simulator and the analysis tools. Boards are at most
21x21, so the array ops are tiny and the constant factor of numpy dominates —
that is fine, a full BFS field costs well under a millisecond.
"""

from __future__ import annotations

import numpy as np

# Direction codes as defined by the wire protocol.
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def dilate4(mask: np.ndarray) -> np.ndarray:
    """Expand a boolean mask to its 4-neighbours. No wrap-around."""
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    return out


def dilate8(mask: np.ndarray) -> np.ndarray:
    """Expand a boolean mask to its full 3x3 neighbourhood. This is exactly the
    engine's visibility rule: you see every cell adjacent to a cell you own."""
    v = mask.copy()
    v[:-1, :] |= mask[1:, :]
    v[1:, :] |= mask[:-1, :]
    out = v.copy()
    out[:, :-1] |= v[:, 1:]
    out[:, 1:] |= v[:, :-1]
    return out


def bfs_field(passable: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Step distance from the nearest source to every cell, over passable ground.

    Unreachable cells get the sentinel H*W, which is larger than any real path,
    so callers can compare distances without special-casing.
    """
    h, w = passable.shape
    inf = h * w
    dist = np.full((h, w), inf, dtype=np.int32)
    reached = sources & passable
    dist[reached] = 0
    step = 1
    while True:
        grown = dilate4(reached) & passable
        newly = grown & ~reached
        if not newly.any():
            return dist
        dist[newly] = step
        reached = grown
        step += 1


def bfs_field_from(passable: np.ndarray, pos: tuple[int, int]) -> np.ndarray:
    src = np.zeros(passable.shape, dtype=bool)
    src[pos] = True
    return bfs_field(passable, src)


def _dilate4_batch(mask: np.ndarray) -> np.ndarray:
    """dilate4 over a stack of masks, shape (N, H, W)."""
    out = mask.copy()
    out[:, :-1, :] |= mask[:, 1:, :]
    out[:, 1:, :] |= mask[:, :-1, :]
    out[:, :, :-1] |= mask[:, :, 1:]
    out[:, :, 1:] |= mask[:, :, :-1]
    return out


def room_field(passable: np.ndarray, radius: int) -> np.ndarray:
    """For every cell, how many *other* passable cells lie within `radius` steps.

    This mirrors `generals/core/grid.py::room_field`, which the map generator
    uses to seat the two generals on comparable ground — and which is therefore
    the strongest static prior on where the enemy general is. One BFS per cell,
    all of them run as a single batched dilation: a few milliseconds on a 21x21
    board, computed once per game.
    """
    h, w = passable.shape
    cells = np.argwhere(passable)
    n = len(cells)
    out = np.zeros((h, w), dtype=np.int32)
    if n == 0:
        return out

    reached = np.zeros((n, h, w), dtype=bool)
    reached[np.arange(n), cells[:, 0], cells[:, 1]] = True
    for _ in range(radius):
        reached = _dilate4_batch(reached) & passable[None]
    out[cells[:, 0], cells[:, 1]] = reached.reshape(n, -1).sum(axis=1) - 1
    return out


def step_toward(field: np.ndarray, r: int, c: int) -> int | None:
    """Direction code that most decreases `field` from (r, c), or None."""
    h, w = field.shape
    best_d, best_v = None, int(field[r, c])
    for d, (dr, dc) in enumerate(DIRS):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and field[nr, nc] < best_v:
            best_d, best_v = d, int(field[nr, nc])
    return best_d


def path_between(field: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    """Walk down a BFS field from `start` to its source. Empty if unreachable."""
    h, w = field.shape
    r, c = start
    if field[r, c] >= h * w:
        return []
    path = [(r, c)]
    while field[r, c] > 0:
        d = step_toward(field, r, c)
        if d is None:
            return []
        r, c = r + DIRS[d][0], c + DIRS[d][1]
        path.append((r, c))
    return path
