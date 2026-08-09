"""Observation-only temporal memory shared by deployment and data builders.

Fog removes owner and army information from the current frame.  This state
keeps what the player previously observed, plus compact change maps.  It never
reads simulator internals, so the submitted bot and offline training can use the
same semantics.  ``learn.rlenv`` contains the JAX mirror.
"""

from __future__ import annotations

import numpy as np

from bot import rules
from bot.obs import Obs


class TemporalMemory:
    """Persistent state for one player in one game."""

    def __init__(self, h: int, w: int):
        self.H, self.W = h, w
        self.turn = -1
        self.initialised = False
        self.known_mountains = np.zeros((h, w), dtype=bool)
        self.mem_owner = np.zeros((h, w), dtype=np.int8)
        self.mem_army = np.zeros((h, w), dtype=np.int32)
        self.mem_turn = np.full((h, w), -1, dtype=np.int32)
        self.ever_seen = np.zeros((h, w), dtype=bool)
        self.ever_enemy = np.zeros((h, w), dtype=bool)
        self.enemy_castles = np.zeros((h, w), dtype=bool)
        self.my_gained = np.zeros((h, w), dtype=bool)
        self.opp_gained = np.zeros((h, w), dtype=bool)
        self.my_army_delta = np.zeros((h, w), dtype=np.int32)
        self.opp_army_delta = np.zeros((h, w), dtype=np.int32)
        self.delta_my_army = 0
        self.delta_opp_army = 0
        self.delta_land_adv = 0
        self._my_army = self._opp_army = 0
        self._my_land = self._opp_land = 0

    def copy(self) -> "TemporalMemory":
        """Independent snapshot for a counterfactual continuation.

        Engine forks without their observation history are not the same state:
        the temporal encoder would see live planes in the parent and zeros in
        the branch. Keep this method next to the state definition so future
        arrays cannot be accidentally shared between branches.
        """
        other = TemporalMemory(self.H, self.W)
        for name, value in self.__dict__.items():
            setattr(other, name, value.copy() if isinstance(value, np.ndarray)
                    else value)
        return other

    def update(self, obs: Obs) -> None:
        """Consume a frame once; repeated scoring of one turn is idempotent."""
        if obs.turn == self.turn:
            return
        if obs.H != self.H or obs.W != self.W:
            raise ValueError(
                f"memory is {self.H}x{self.W}, observation is {obs.H}x{obs.W}")

        t, o, a = obs.type_grid, obs.owner_grid, obs.army_grid
        visible = (t >= rules.T_PLAIN) & (t <= rules.T_GENERAL)
        mine = o == rules.OWNER_ME
        opp = o == rules.OWNER_OPP
        known_before = self.ever_seen.copy()

        if not self.initialised:
            # Mountains and neutral castles share the structure-in-fog token.
            # Treat both as blocked until first sight, then correct below.
            self.known_mountains = (
                (t == rules.T_MOUNTAIN) | (t == rules.T_STRUCTURE_IN_FOG))
            self.initialised = True
        # A revealed neutral/enemy castle disproves the provisional mountain;
        # an actually visible mountain confirms it.
        self.known_mountains |= t == rules.T_MOUNTAIN
        self.known_mountains &= ~(visible & (t != rules.T_MOUNTAIN))

        old_owner = self.mem_owner.copy()
        old_army = self.mem_army.copy()
        changed = visible & known_before
        self.my_gained = changed & mine & (old_owner != rules.OWNER_ME)
        self.opp_gained = changed & opp & (old_owner != rules.OWNER_OPP)
        self.my_army_delta.fill(0)
        self.opp_army_delta.fill(0)
        self.my_army_delta[changed & mine] = (
            a[changed & mine] - old_army[changed & mine])
        self.opp_army_delta[changed & opp] = (
            a[changed & opp] - old_army[changed & opp])

        self.enemy_castles |= ((t == rules.T_STRUCTURE_IN_FOG)
                               & ~self.known_mountains
                               & (self.mem_owner == rules.OWNER_OPP))
        self.enemy_castles |= (t == rules.T_CASTLE) & opp
        self.enemy_castles &= ~((t == rules.T_CASTLE) & ~opp)

        self.mem_owner[visible] = o[visible]
        self.mem_army[visible] = a[visible]
        self.mem_turn[visible] = obs.turn
        self.ever_seen |= visible
        self.ever_enemy |= visible & opp

        if self.turn >= 0:
            self.delta_my_army = int(obs.my_army) - self._my_army
            self.delta_opp_army = int(obs.opp_army) - self._opp_army
            self.delta_land_adv = ((int(obs.my_land) - self._my_land)
                                   - (int(obs.opp_land) - self._opp_land))
        else:
            self.delta_my_army = self.delta_opp_army = self.delta_land_adv = 0
        self._my_army, self._opp_army = int(obs.my_army), int(obs.opp_army)
        self._my_land, self._opp_land = int(obs.my_land), int(obs.opp_land)
        self.turn = int(obs.turn)
