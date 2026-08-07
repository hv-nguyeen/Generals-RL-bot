"""Dihedral symmetry of the board, for test-time augmentation.

The rules are symmetric under the eight rigid motions of the square, so a
rotated game is a real game and a net's answer on one orientation should equal
its answer on another. It does not: the trunk is a stack of 3x3 convolutions,
which is translation-equivariant and nothing else, so eight orientations give
eight slightly different logit vectors. Averaging them is free variance
reduction, and the budget is there -- one forward is 1.2 ms against a 150 ms
limit, so eight is ~10 ms.

This is `learn.train._dihedral_maps` reproduced here because `bot/` may not
import `learn/`. `tests/test_all.py` checks the two agree element for element.

A move action is direction-dependent and moves with the board; a BUILD is
position-only and must NOT follow the direction permutation. Getting that
wrong looks harmless -- the action map starts as the identity -- and silently
relabels every build, which is the bug the training-side comment warns about.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from bot import features

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))       # features.legal_mask order


@lru_cache(maxsize=None)
def maps(h: int, w: int, g: int):
    """(cell gather, action relabel) for group element g on an h x w board.

    g = 2*k + f: k quarter turns, then an optional left-right mirror. The board
    sits at the TOP-LEFT of the padded square, so the transform applies to the
    h x w crop and the result is re-placed at the top-left -- rotating the padded
    tensor would move the board into another corner and manufacture an
    observation that never occurs at match time.
    """
    pad = features.PAD
    if not (3 <= h <= pad and 3 <= w <= pad):
        raise ValueError(f"board {h}x{w} is not a competition board")
    ids = np.arange(h * w).reshape(h, w)
    k, f = divmod(g, 2)
    t = np.rot90(ids, k)
    if f:
        t = np.fliplr(t)
    h2, w2 = t.shape

    srcof = np.full(pad * pad, pad * pad - 1, np.int64)
    src_r, src_c = np.divmod(t.reshape(-1), w)
    dst = (np.arange(h2)[:, None] * pad + np.arange(w2)[None, :]).reshape(-1)
    srcof[dst] = src_r * pad + src_c

    fr = np.empty(h * w, np.int64)
    fc = np.empty(h * w, np.int64)
    fr[t.reshape(-1)] = np.repeat(np.arange(h2), w2)
    fc[t.reshape(-1)] = np.tile(np.arange(w2), h2)

    # A rigid motion moves every cell the same way, so read the direction
    # permutation off one interior cell rather than hardcoding eight tables.
    mid = (h // 2) * w + (w // 2)
    dperm = []
    for dr, dc in DIRS:
        nb = mid + dr * w + dc
        v = (int(fr[nb] - fr[mid]), int(fc[nb] - fc[mid]))
        dperm.append(DIRS.index(v))

    per, splits = features.PER_CELL, features.SPLITS
    actmap = np.arange(features.N_ACTIONS, dtype=np.int64)       # pass is a fixed point
    i = np.arange(h * w)
    oldbase = ((i // w) * pad + (i % w)) * per
    newbase = (fr * pad + fc) * per
    for d in range(features.DIRS_N):
        for s in range(splits):
            actmap[oldbase + d * splits + s] = newbase + dperm[d] * splits + s
    actmap[oldbase + features.BUILD_OFFSET] = newbase + features.BUILD_OFFSET
    return srcof, actmap


FULL = (0, 1, 2, 3, 4, 5, 6, 7)
SHAPE_PRESERVING = (0, 1, 6, 7)


def group(h: int, w: int, full: bool = False) -> tuple[int, ...]:
    """The elements worth using on an h x w board.

    A quarter turn of a non-square board is a TRANSPOSED board, and the original
    version of this function withheld those out of caution -- "a shape the net
    never saw in training". That was wrong twice over, and both halves are
    checkable in the repo:

    * `sim.mapgen.generate` draws h and w INDEPENDENTLY from the same range, so
      an 18x21 board transposed is a 21x18 board, which occurs with identical
      probability. The transformed observation is a legal position on a legal
      board.
    * `learn.train.augment` already applies all eight elements to non-square
      boards -- it groups by (h, w) and builds the maps for any g -- so the
      behaviour-cloned ancestor of this whole lineage trained under the full
      group.

    Boards are 18-21 per side drawn independently, so only ~1 in 4 is square:
    withholding half the group cost most of the averaging on most boards.

    Kept behind a flag rather than simply changed, because the +31.2 that is
    deployed was measured with the four-element form and an A/B has to be able
    to reproduce it exactly.
    """
    return FULL if (full or h == w) else SHAPE_PRESERVING


def average_logits(x: np.ndarray, h: int, w: int, forward, elements=None) -> np.ndarray:
    """Mean of `forward` over the dihedral group, mapped back to the original frame.

    `x` is the encoded (C, PAD, PAD) board and `forward` takes one of those and
    returns the flat logit vector. Element 0 is the identity, so a one-element
    group reproduces the plain forward exactly.
    """
    els = group(h, w) if elements is None else elements
    C = x.shape[0]
    flat = x.reshape(C, -1)
    total = None
    for g in els:
        if g == 0:
            out = forward(x)
        else:
            srcof, actmap = maps(h, w, g)
            xt = flat[:, srcof].reshape(x.shape)
            # `forward` answers in the TRANSFORMED frame; actmap sends an
            # original action to its transformed index, so gathering by it
            # brings the vector home.
            out = forward(xt)[actmap]
        total = out if total is None else total + out
    return total / len(els)
