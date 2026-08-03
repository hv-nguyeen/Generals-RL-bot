"""Castle economy.

The competition strips neutral castles and lets players build their own for
35 army plus `max(0, 14 - 2d)` per own structure within 6 manhattan steps. That
makes a castle +0.5 army/turn forever in exchange for **one turn** — expanding
34 tiles buys slightly more income but costs 34 turns. Turns are the scarce
resource, so castles are the single biggest economic lever in this ruleset.

Siting rules that fall out of that:
  * only build where the surcharge is zero (7+ from our own structures), so the
    price stays a flat 35;
  * only build in the rear, out of contact — a fresh castle is left holding
    almost no army and a single enemy unit can take it;
  * prefer the cheapest safe tile closest to home, so it is defensible and the
    army does not have to walk far.

Building is opportunistic first: if a stack is already standing on a legal site
we build for free. Only when we have plenty of army but no single big stack do
we spend turns walking army to a site.
"""

from __future__ import annotations

import numpy as np

from bot import rules
from bot.config import Config
from bot.policy.analysis import Analysis


class CastlePlan:
    __slots__ = ("action", "site", "cost")

    def __init__(self, action=None, site=None, cost=0):
        self.action = action
        self.site = site
        self.cost = cost


def plan(an: Analysis, cfg: Config) -> CastlePlan:
    """Either an immediate build action, or a site worth gathering to."""
    empty = CastlePlan()
    if not cfg.castle_enabled:
        return empty
    obs = an.obs
    if obs.turn < cfg.castle_min_turn or obs.turn > cfg.castle_max_turn:
        return empty
    if obs.my_land < cfg.castle_min_land:
        return empty
    if int(an.belief.my_castles.sum()) >= cfg.max_castles:
        return empty

    cost = rules.build_cost_grid(an.my_structures)
    eligible = (
        an.my_mask
        & ~an.my_structures
        & (cost <= cfg.castle_max_cost)
        & (an.dist_enemy_terr >= cfg.castle_safe_dist)
    )
    if not eligible.any():
        return empty

    # Cheap, close to home, far from the enemy.
    score = np.where(
        eligible,
        -1.0 * an.dist_home.astype(np.float64)
        + 0.5 * np.minimum(an.dist_enemy_terr, 30).astype(np.float64)
        - 0.5 * cost.astype(np.float64),
        -np.inf,
    )

    # Can we build right now, without spending a turn walking army over?
    ready = eligible & (obs.army_grid >= cost + cfg.castle_keep)
    if ready.any():
        ready_score = np.where(ready, score, -np.inf)
        idx = int(np.argmax(ready_score))
        r, c = idx // obs.W, idx % obs.W
        return CastlePlan(action=(rules.BUILD, r, c, 0, 0), site=(r, c), cost=int(cost[r, c]))

    if obs.my_army < cfg.castle_gather_min_army:
        return empty
    idx = int(np.argmax(score))
    r, c = idx // obs.W, idx % obs.W
    return CastlePlan(site=(r, c), cost=int(cost[r, c]))
