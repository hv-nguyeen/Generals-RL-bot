"""The per-turn observation, exactly as the wire protocol delivers it.

The arena builds this straight from simulator state instead of going through
strings, so this class is the single definition of "what a bot sees" and the
in-process and stdio paths cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Obs:
    H: int
    W: int
    turn: int
    my_land: int
    my_army: int
    opp_land: int
    opp_army: int
    type_grid: np.ndarray   # (H, W) int8, see rules.T_*
    owner_grid: np.ndarray  # (H, W) int8, 0 neutral/unknown, 1 me, 2 opponent
    army_grid: np.ndarray   # (H, W) int32, 0 where not visible
