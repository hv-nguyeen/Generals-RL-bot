"""Everything the bot remembers between turns.

Three things live here that a stateless bot cannot have, and each one is worth
real Elo:

1. **The exact terrain, from turn 1.** The engine reports fogged mountains and
   fogged castles with the same code (5), which normally makes terrain
   ambiguous — but the competition ruleset strips every neutral castle before
   the game starts, so on the first frame *every* type-5 cell is a mountain.
   That hands us the complete `passable` map before we have seen anything.

2. **Enemy castle detection.** Because (1) pins the terrain, any cell that later
   turns into a type-5 must be a castle the opponent just built. It is visible
   through fog anywhere on the board, so it leaks their territory for free.

3. **The enemy general prior.** The map generator seats general B uniformly
   among the cells that are at least 17 BFS steps from A *and* within 5 of A's
   `room7` count. Both quantities are computable from (1), so the general is
   pinned to a few dozen cells before first contact, then narrowed by every cell
   we observe.
"""

from __future__ import annotations

import numpy as np

from bot import rules
from bot.board import bfs_field, bfs_field_from, room_field
from bot.obs import Obs


class Belief:
    def __init__(self, player_id: int, h: int, w: int):
        self.player_id = player_id
        self.H, self.W = h, w
        self.turn = -1

        self.mountains = np.zeros((h, w), dtype=bool)
        self.passable = np.ones((h, w), dtype=bool)
        self.my_general: tuple[int, int] | None = None
        self.enemy_general: tuple[int, int] | None = None

        # Fog memory: what a cell looked like the last time we could see it.
        self.mem_owner = np.zeros((h, w), dtype=np.int8)
        self.mem_army = np.zeros((h, w), dtype=np.int32)
        self.mem_turn = np.full((h, w), -1, dtype=np.int32)
        self.ever_seen = np.zeros((h, w), dtype=bool)
        self.ever_enemy = np.zeros((h, w), dtype=bool)

        self.enemy_castles = np.zeros((h, w), dtype=bool)
        self.my_castles = np.zeros((h, w), dtype=bool)

        self.candidates = np.zeros((h, w), dtype=bool)
        self.general_guess: tuple[int, int] | None = None
        self.dist_from_my_general = np.zeros((h, w), dtype=np.int32)
        self.room = np.zeros((h, w), dtype=np.int32)

        # Bookkeeping the policy reads.
        self.hidden_enemy_army = 0
        self.castles_built = 0
        self._initialised = False

    # -- per-turn entry point -------------------------------------------------
    def update(self, obs: Obs) -> None:
        self.turn = obs.turn
        t, o, a = obs.type_grid, obs.owner_grid, obs.army_grid
        visible = (t >= rules.T_PLAIN) & (t <= rules.T_GENERAL)

        if not self._initialised:
            self._bootstrap(obs, visible)

        # Terrain is fixed, so a *new* structure-in-fog is a freshly built castle.
        self.enemy_castles |= (t == rules.T_STRUCTURE_IN_FOG) & ~self.mountains
        self.enemy_castles |= (t == rules.T_CASTLE) & (o == rules.OWNER_OPP)
        self.my_castles = (t == rules.T_CASTLE) & (o == rules.OWNER_ME)
        # A castle we can see and no longer own has changed hands.
        self.enemy_castles &= ~self.my_castles

        self.mem_owner[visible] = o[visible]
        self.mem_army[visible] = a[visible]
        self.mem_turn[visible] = obs.turn
        self.ever_seen |= visible
        self.ever_enemy |= visible & (o == rules.OWNER_OPP)
        # Cells we can see and no longer hold enemy troops are stale evidence.
        self.ever_enemy &= ~(visible & (o != rules.OWNER_OPP))

        self.hidden_enemy_army = int(obs.opp_army) - int(a[(o == rules.OWNER_OPP)].sum())

        self._locate_enemy_general(obs, visible)

    # -- first frame ----------------------------------------------------------
    def _bootstrap(self, obs: Obs, visible: np.ndarray) -> None:
        t, o = obs.type_grid, obs.owner_grid
        self.mountains = (t == rules.T_MOUNTAIN) | (t == rules.T_STRUCTURE_IN_FOG)
        self.passable = ~self.mountains

        mine_general = np.argwhere((t == rules.T_GENERAL) & (o == rules.OWNER_ME))
        if len(mine_general):
            self.my_general = (int(mine_general[0][0]), int(mine_general[0][1]))
        else:  # should not happen; degrade to any owned cell
            owned = np.argwhere(o == rules.OWNER_ME)
            self.my_general = (int(owned[0][0]), int(owned[0][1])) if len(owned) else (0, 0)

        self.dist_from_my_general = bfs_field_from(self.passable, self.my_general)
        self.room = room_field(self.passable, rules.SPAWN_ROOM_RADIUS)

        # The generator's own two constraints on where the opponent can be.
        my_room = int(self.room[self.my_general])
        self.candidates = (
            self.passable
            & (self.dist_from_my_general >= rules.MIN_GENERALS_DISTANCE)
            & (np.abs(self.room - my_room) <= rules.SPAWN_ROOM_TOLERANCE)
        )
        if not self.candidates.any():
            # Degenerate board: fall back to the distance constraint alone.
            self.candidates = self.passable & (self.dist_from_my_general >= rules.MIN_GENERALS_DISTANCE)
        self._initialised = True

    # -- enemy general --------------------------------------------------------
    def _locate_enemy_general(self, obs: Obs, visible: np.ndarray) -> None:
        t, o = obs.type_grid, obs.owner_grid

        seen_general = np.argwhere((t == rules.T_GENERAL) & (o == rules.OWNER_OPP))
        if len(seen_general):
            self.enemy_general = (int(seen_general[0][0]), int(seen_general[0][1]))
            self.general_guess = self.enemy_general
            self.candidates[:] = False
            self.candidates[self.enemy_general] = True
            return

        # Anything we have looked at and that was not a general is ruled out.
        self.candidates &= ~(self.ever_seen & ~((t == rules.T_GENERAL) & (o == rules.OWNER_OPP)))
        if not self.candidates.any():
            # Every candidate eliminated (can happen on odd boards): re-seed from
            # the distance constraint so the bot still has something to aim at.
            self.candidates = self.passable & ~self.ever_seen & (
                self.dist_from_my_general >= rules.MIN_GENERALS_DISTANCE // 2)
            if not self.candidates.any():
                self.general_guess = None
                return

        # Evidence: their general sits inside their territory, so candidates near
        # tiles they hold (or castles they built) are far likelier.
        evidence = self.ever_enemy | self.enemy_castles
        score = 0.25 * self.dist_from_my_general.astype(np.float64)
        if evidence.any():
            d_enemy = bfs_field(self.passable, evidence).astype(np.float64)
            score -= 2.0 * np.minimum(d_enemy, self.H * self.W)
        score[~self.candidates] = -np.inf

        idx = int(np.argmax(score))
        self.general_guess = (idx // self.W, idx % self.W)

    # -- helpers the policy uses ---------------------------------------------
    def my_structures(self, obs: Obs) -> np.ndarray:
        t, o = obs.type_grid, obs.owner_grid
        return ((t == rules.T_GENERAL) | (t == rules.T_CASTLE)) & (o == rules.OWNER_ME)

    def known_enemy_structures(self) -> np.ndarray:
        s = self.enemy_castles.copy()
        if self.enemy_general is not None:
            s[self.enemy_general] = True
        return s
