"""Model of the competition ruleset.

Every constant here was read out of the engine source
(`generals/core/game.py`, `generals/modifiers/*.py`, `GeneralsEnv` mode
"competition"), not off the rules page. Both the bot and the simulator use this
module, so a rule can only be wrong in one place.
"""

from __future__ import annotations

import numpy as np

# --- wire protocol cell types -------------------------------------------------
T_FOG = 0
T_PLAIN = 1
T_MOUNTAIN = 2
T_CASTLE = 3
T_GENERAL = 4
T_STRUCTURE_IN_FOG = 5

OWNER_NEUTRAL = 0
OWNER_ME = 1
OWNER_OPP = 2

# --- action kinds -------------------------------------------------------------
MOVE = 0
PASS = 1
BUILD = 2
PASS_ACTION = (PASS, 0, 0, 0, 0)

# --- castle building ----------------------------------------------------------
BASE_COST = 35
PROXIMITY_PENALTY = 14
PROXIMITY_DECAY = 2
# Farthest manhattan distance that still carries a surcharge: 14 - 2*6 = 2 > 0.
SURCHARGE_RADIUS = (PROXIMITY_PENALTY - 1) // PROXIMITY_DECAY

# --- match structure ----------------------------------------------------------
DEATHTOUCH_TURN = 800
TURN_LIMIT = 1200
MIN_GENERALS_DISTANCE = 17
SPAWN_ROOM_RADIUS = 7
SPAWN_ROOM_TOLERANCE = 5
MIN_SIDE, MAX_SIDE = 18, 21
MOUNTAIN_DENSITY = (0.24, 0.26)
NUM_CARVED_CASTLES = (9, 11)

# --- sandbox ------------------------------------------------------------------
MOVE_BUDGET_S = 0.150
FIRST_MOVE_BUDGET_S = 10.0
MAX_FAULTS = 50


def build_cost_grid(structures: np.ndarray) -> np.ndarray:
    """Castle price on every cell for the player owning `structures`.

    35 everywhere, plus max(0, 14 - 2d) for each of that player's own
    structures (general + castles) at manhattan distance d. Enemy structures
    never affect the price. Sited 7+ away from your own stuff it is a flat 35.
    """
    h, w = structures.shape
    r = SURCHARGE_RADIUS
    padded = np.zeros((h + 2 * r, w + 2 * r), dtype=np.int32)
    padded[r:r + h, r:r + w] = structures.astype(np.int32)

    cost = np.full((h, w), BASE_COST, dtype=np.int32)
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            surcharge = PROXIMITY_PENALTY - PROXIMITY_DECAY * (abs(di) + abs(dj))
            if surcharge > 0:
                cost += surcharge * padded[r + di:r + di + h, r + dj:r + dj + w]
    return cost


def structures_grow(turn: int) -> bool:
    """Generals and castles gain one army on even ticks."""
    return turn % 2 == 0


def all_grow(turn: int) -> bool:
    """Every owned tile gains one army every 50 ticks."""
    return turn % 50 == 0


def army_to_move(source_army: int, split: int) -> int:
    """How much army actually leaves the source cell."""
    a = source_army // 2 if split else source_army - 1
    return max(0, min(a, source_army - 1))


def growth_over(turn_from: int, turn_to: int, structures: int, land: int) -> int:
    """Army a player gains between two turns, holding structures/land constant.

    Used for attack-timing arithmetic ("will their general out-grow my stack on
    the way over?"), so it is deliberately an estimate: it assumes the counts do
    not change over the interval.
    """
    even_ticks = turn_to // 2 - turn_from // 2
    fifty_ticks = turn_to // 50 - turn_from // 50
    return structures * even_ticks + land * fifty_ticks


def general_army_at(turn: int) -> int:
    """Army on an untouched general at `turn` (spawns with 1, +1 every even tick)."""
    return 1 + turn // 2
