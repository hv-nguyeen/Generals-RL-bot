"""Per-turn derived view of the board.

Four BFS fields and a handful of masks. Everything the move scorer and the mode
selector need is computed once here, so the scoring loop is pure arithmetic over
python lists and stays far inside the 150 ms budget.
"""

from __future__ import annotations

import numpy as np

from bot import rules
from bot.board import bfs_field, bfs_field_from, dilate4, dilate8
from bot.belief import Belief
from bot.obs import Obs


class Analysis:
    __slots__ = (
        "obs", "belief", "H", "W", "diag",
        "my_mask", "opp_mask", "neutral_mask", "fog_mask",
        "my_structures", "dist_home", "dist_unowned", "dist_enemy_gen",
        "dist_enemy_terr", "dist_gather", "reveal", "frontier_adj", "enemy_adj",
        "general_army", "my_total_army", "biggest_stack", "biggest_stack_pos",
        "_def_dists", "_def_cum", "threat_army", "threat_dist", "threat_pos",
        "unowned_near",
    )

    def __init__(self, belief: Belief, obs: Obs):
        self.obs, self.belief = obs, belief
        self.H, self.W = obs.H, obs.W
        self.diag = float(obs.H + obs.W)

        t, o, a = obs.type_grid, obs.owner_grid, obs.army_grid
        passable = belief.passable

        self.my_mask = o == rules.OWNER_ME
        self.opp_mask = o == rules.OWNER_OPP
        self.neutral_mask = (o == rules.OWNER_NEUTRAL) & (t == rules.T_PLAIN)
        self.fog_mask = (t == rules.T_FOG) | (t == rules.T_STRUCTURE_IN_FOG)
        self.my_structures = belief.my_structures(obs)

        gr, gc = belief.my_general
        self.general_army = int(a[gr, gc]) if self.my_mask[gr, gc] else 0
        self.my_total_army = int(obs.my_army)

        # --- distance fields -------------------------------------------------
        self.dist_home = belief.dist_from_my_general          # static, free
        unowned = passable & ~self.my_mask
        self.dist_unowned = bfs_field(passable, unowned) if unowned.any() \
            else np.zeros((obs.H, obs.W), np.int32)

        guess = belief.general_guess
        self.dist_enemy_gen = bfs_field_from(passable, guess) if guess else self.dist_unowned

        evidence = belief.ever_enemy | belief.enemy_castles | self.opp_mask
        self.dist_enemy_terr = bfs_field(passable, evidence) if evidence.any() \
            else self.dist_enemy_gen
        self.dist_gather = self.dist_home  # replaced by the controller when gathering

        # How much free ground is still within reach of our territory. When this
        # dries up, hoarding army is pointless and it is time to go at them.
        if self.my_mask.any():
            spread = bfs_field(passable, self.my_mask)
            self.unowned_near = int((unowned & (spread <= 8)).sum())
        else:
            self.unowned_near = 0

        # --- cheap per-cell features ----------------------------------------
        self.reveal = _count3x3(self.fog_mask)
        # How much expansion is still available *from* a cell once we hold it.
        self.frontier_adj = _count4(unowned)
        self.enemy_adj = dilate8(self.opp_mask)

        # --- stacks and threats ---------------------------------------------
        # The general's garrison is not a fist. Counting it made the general our
        # biggest stack whenever we gathered, which fired ATTACK, and the move
        # scorer exempts ATTACK from holding the general — so the base marched
        # off in one move and died a few ticks later. 324daa0 stopped the thrust
        # launching from the general; this is the same bug on the scorer path.
        movable = self.my_mask & (a >= 2)
        movable[gr, gc] = False
        if movable.any():
            idx = int(np.argmax(np.where(movable, a, 0)))
            self.biggest_stack_pos = (idx // obs.W, idx % obs.W)
            self.biggest_stack = int(a[self.biggest_stack_pos])
        else:
            self.biggest_stack_pos, self.biggest_stack = None, 0

        self._build_defense_curve(a, (gr, gc))
        self._find_threat(a)

    # -- how much army can reach the general within t turns --------------------
    def _build_defense_curve(self, army: np.ndarray, general: tuple[int, int]) -> None:
        """Order our stacks by how much army each buys per move spent.

        There is ONE move per turn, so fetching several stacks costs the SUM of
        their distances, not the max. The old version summed every tile within
        `t` steps as if they all arrived at once, which overstated our defence by
        several times: measured on real losses we were dying to 7-army stacks
        after 60+ turns of warning, because the trigger thought we had 17 army in
        hand when we could actually bring home one stack and the general.
        """
        mask = self.my_mask.copy()
        mask[general] = False
        d = np.maximum(self.dist_home[mask].astype(np.int64), 1)
        contrib = np.maximum(army[mask].astype(np.int64) - 1, 0)
        keep = contrib > 0
        d, contrib = d[keep], contrib[keep]
        # greedy: best army-per-move first, then spend the move budget in order
        order = np.argsort(-(contrib / d), kind="stable")
        self._def_dists = np.cumsum(d[order])
        self._def_cum = np.cumsum(contrib[order])

    def defense_within(self, turns: int) -> int:
        """Army we could actually have standing on the general in `turns` turns."""
        if self._def_dists.size == 0:
            return self.general_army
        k = int(np.searchsorted(self._def_dists, turns, side="right"))
        reachable = int(self._def_cum[k - 1]) if k else 0
        return self.general_army + reachable

    def max_threat_within(self, steps: int) -> int:
        """Largest army an enemy could march at our general from within `steps`."""
        near = self.opp_mask & (self.dist_home <= steps)
        if not near.any():
            return 0
        return int(np.maximum(self.obs.army_grid[near] - 1, 0).max())

    # -- the single most dangerous visible enemy stack -------------------------
    def _find_threat(self, army: np.ndarray) -> None:
        self.threat_army, self.threat_dist, self.threat_pos = 0, 0, None
        stacks = self.opp_mask & (army >= 2)
        if not stacks.any():
            return
        best = None
        for r, c in np.argwhere(stacks):
            d = int(self.dist_home[r, c])
            if d >= self.H * self.W:
                continue
            strength = int(army[r, c]) - 1
            # Rank by how much they would arrive with, not raw size.
            margin = strength - self.defense_within(d)
            if best is None or margin > best[0]:
                best = (margin, strength, d, (int(r), int(c)))
        if best is not None:
            _, self.threat_army, self.threat_dist, self.threat_pos = best


def _count3x3(mask: np.ndarray) -> np.ndarray:
    """For every cell, how many cells of `mask` lie in its 3x3 neighbourhood."""
    m = mask.astype(np.int16)
    v = m.copy()
    v[:-1, :] += m[1:, :]
    v[1:, :] += m[:-1, :]
    out = v.copy()
    out[:, :-1] += v[:, 1:]
    out[:, 1:] += v[:, :-1]
    return out


def _count4(mask: np.ndarray) -> np.ndarray:
    """For every cell, how many of its 4 neighbours are in `mask`."""
    m = mask.astype(np.int16)
    out = np.zeros_like(m)
    out[:-1, :] += m[1:, :]
    out[1:, :] += m[:-1, :]
    out[:, :-1] += m[:, 1:]
    out[:, 1:] += m[:, :-1]
    return out
