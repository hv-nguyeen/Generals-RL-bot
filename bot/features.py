"""Board -> tensor, and action <-> flat index.

Shared by the dataset builder, the trainer and the numpy policy that runs inside
the submission, so a training example and a match-time observation are encoded
by the same code. If they ever diverge the network sees one thing in training
and another in play, which is the classic way a cloned policy quietly fails.

Boards vary from 18x18 to 21x21; everything is padded to 21x21 and channel
`VALID` marks the real region so the network can tell board edge from padding.
"""

from __future__ import annotations

import numpy as np

from bot import rules
from bot.obs import Obs

PAD = 21                      # every competition board fits in 21x21
C = 12                        # feature channels
DIRS_N = 4
SPLITS = 2
PER_CELL = DIRS_N * SPLITS    # 8 moves per source cell
N_ACTIONS = PAD * PAD * PER_CELL + 1
PASS_INDEX = N_ACTIONS - 1

(MINE, OPP, NEUTRAL, FOG, MOUNTAIN, CASTLE,
 MY_GEN, OPP_GEN, ARMY_MINE, ARMY_OPP, ARMY_NEUTRAL, VALID) = range(C)


def encode(obs: Obs) -> np.ndarray:
    """(C, PAD, PAD) float32 from one fogged observation."""
    x = np.zeros((C, PAD, PAD), dtype=np.float32)
    h, w = obs.H, obs.W
    t, o, a = obs.type_grid, obs.owner_grid, obs.army_grid

    mine = o == rules.OWNER_ME
    opp = o == rules.OWNER_OPP
    x[MINE, :h, :w] = mine
    x[OPP, :h, :w] = opp
    x[NEUTRAL, :h, :w] = (o == rules.OWNER_NEUTRAL) & (t == rules.T_PLAIN)
    x[FOG, :h, :w] = (t == rules.T_FOG) | (t == rules.T_STRUCTURE_IN_FOG)
    x[MOUNTAIN, :h, :w] = (t == rules.T_MOUNTAIN) | (t == rules.T_STRUCTURE_IN_FOG)
    x[CASTLE, :h, :w] = t == rules.T_CASTLE
    x[MY_GEN, :h, :w] = (t == rules.T_GENERAL) & mine
    x[OPP_GEN, :h, :w] = (t == rules.T_GENERAL) & opp

    # log-compressed army, split by owner so the network never has to subtract
    la = np.log1p(np.maximum(a, 0).astype(np.float32)) / 6.0
    x[ARMY_MINE, :h, :w] = np.where(mine, la, 0.0)
    x[ARMY_OPP, :h, :w] = np.where(opp, la, 0.0)
    x[ARMY_NEUTRAL, :h, :w] = np.where(~mine & ~opp, la, 0.0)
    x[VALID, :h, :w] = 1.0
    return x


def action_to_index(action) -> int:
    """Wire action -> flat class. Builds are not modelled; they map to pass."""
    kind, r, c, d, split = (int(v) for v in action)
    if kind != rules.MOVE:
        return PASS_INDEX
    if not (0 <= r < PAD and 0 <= c < PAD and 0 <= d < DIRS_N):
        return PASS_INDEX
    return ((r * PAD + c) * PER_CELL) + d * SPLITS + (1 if split else 0)


def index_to_action(idx: int):
    if idx >= PASS_INDEX:
        return rules.PASS_ACTION
    cell, rest = divmod(int(idx), PER_CELL)
    r, c = divmod(cell, PAD)
    d, split = divmod(rest, SPLITS)
    return (rules.MOVE, r, c, d, split)


def legal_mask(obs: Obs) -> np.ndarray:
    """Bool mask over the flat action space. Illegal moves are silently passes in
    the engine, so masking them keeps the network from wasting capacity."""
    m = np.zeros(N_ACTIONS, dtype=bool)
    m[PASS_INDEX] = True
    h, w = obs.H, obs.W
    o, a, t = obs.owner_grid, obs.army_grid, obs.type_grid
    src = np.argwhere((o == rules.OWNER_ME) & (a >= 2))
    for r, c in src:
        r, c = int(r), int(c)
        for d, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if t[nr, nc] == rules.T_MOUNTAIN:
                continue
            base = ((r * PAD + c) * PER_CELL) + d * SPLITS
            m[base] = True
            if a[r, c] >= 4:
                m[base + 1] = True
    return m
