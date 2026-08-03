"""Pure-numpy mirror of the competition transition.

Line-for-line equivalent of `generals/core/game.py::step` composed with the
`build_castles` and `deathtouch` modifiers, which is exactly what
`GeneralsEnv(mode="competition")` plays. It exists so the arena can run
thousands of games without JAX dispatch overhead, and it is kept honest by
`tools/verify_engine.py`, which steps this and the real engine in lockstep on
random games and asserts the states match.

Two fidelity details that are easy to get wrong and change outcomes:

* JAX **clamps** out-of-range indices instead of wrapping like numpy, so every
  index derived from an agent-supplied action is clipped explicitly.
* The move-order tie-break reads the *unclipped* destination when testing for a
  chase, but the *clipped* one when indexing the board. Both are reproduced.
"""

from __future__ import annotations

import numpy as np

from bot import rules
from bot.board import DIRS, dilate8
from bot.obs import Obs

PASS = rules.PASS
BUILD = rules.BUILD


class State:
    __slots__ = ("armies", "own", "neutral", "generals", "castles", "mountains",
                 "passable", "gpos", "time", "winner")

    def __init__(self, armies, own, neutral, generals, castles, mountains, passable, gpos):
        self.armies = armies
        self.own = own
        self.neutral = neutral
        self.generals = generals
        self.castles = castles
        self.mountains = mountains
        self.passable = passable
        self.gpos = gpos
        self.time = 0
        self.winner = -1

    @property
    def shape(self):
        return self.armies.shape

    def copy(self) -> "State":
        s = State(self.armies.copy(), self.own.copy(), self.neutral.copy(),
                  self.generals.copy(), self.castles.copy(), self.mountains.copy(),
                  self.passable.copy(), list(self.gpos))
        s.time, s.winner = self.time, self.winner
        return s

    def land(self) -> tuple[int, int]:
        return int(self.own[0].sum()), int(self.own[1].sum())

    def army(self) -> tuple[int, int]:
        return int((self.armies * self.own[0]).sum()), int((self.armies * self.own[1]).sum())


def from_grid(grid: np.ndarray) -> State:
    """Build a state from the generator's numeric grid (-2 mountain, 1/2 generals)."""
    g0 = grid == 1
    g1 = grid == 2
    generals = g0 | g1
    mountains = grid == -2
    passable = ~mountains
    castles = grid > 2

    own = np.stack([g0, g1])
    neutral = passable & ~g0 & ~g1
    armies = np.where(generals, 1, 0).astype(np.int32)
    armies = np.where(castles, grid, armies).astype(np.int32)

    p0 = np.argwhere(g0)[0]
    p1 = np.argwhere(g1)[0]
    return State(armies, own, neutral, generals, castles, mountains, passable,
                 [(int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1]))])


def _clip(v: int, hi: int) -> int:
    return 0 if v < 0 else (hi if v > hi else v)


def _move_params(st: State, p: int, action):
    """(valid, di, dj, army_to_move) for a move action. Mirrors _execute_move."""
    h, w = st.armies.shape
    si, sj, direction, split = int(action[1]), int(action[2]), int(action[3]), int(action[4])
    direction = _clip(direction, 3)
    in_bounds = 0 <= si < h and 0 <= sj < w
    di = si + DIRS[direction][0]
    dj = sj + DIRS[direction][1]
    dest_in_bounds = 0 <= di < h and 0 <= dj < w

    csi, csj = _clip(si, h - 1), _clip(sj, w - 1)
    cdi, cdj = _clip(di, h - 1), _clip(dj, w - 1)

    owns = bool(st.own[p, csi, csj])
    src_army = int(st.armies[csi, csj])
    amt = src_army // 2 if split == 1 else src_army - 1
    amt = max(0, min(amt, src_army - 1))

    valid = in_bounds and dest_in_bounds and owns and amt > 0 and bool(st.passable[cdi, cdj])
    return valid, di, dj, amt


def _execute(st: State, p: int, action) -> None:
    if int(action[0]) == PASS:
        return
    valid, di, dj, amt = _move_params(st, p, action)
    if not valid:
        return
    si, sj = int(action[1]), int(action[2])

    t0 = bool(st.own[0, di, dj])
    t1 = bool(st.own[1, di, dj])
    tn = bool(st.neutral[di, dj])
    moving_to_own = (p == 0 and t0) or (p == 1 and t1)

    if moving_to_own:
        st.armies[di, dj] += amt
        st.armies[si, sj] -= amt
        return

    target = int(st.armies[di, dj])
    st.armies[di, dj] = abs(target - amt)
    st.armies[si, sj] -= amt
    if amt > target:
        st.own[p, di, dj] = True
        if t0 and p == 1:
            st.own[0, di, dj] = False
        if t1 and p == 0:
            st.own[1, di, dj] = False
        if tn:
            st.neutral[di, dj] = False
        if st.generals[di, dj]:
            st.winner = p


def _touches_general(st: State, p: int, action) -> bool:
    if int(action[0]) != rules.MOVE:
        return False
    valid, di, dj, _ = _move_params(st, p, action)
    if not valid:
        return False
    gr, gc = st.gpos[1 - p]
    return di == gr and dj == gc


def _determine_order(st: State, actions) -> int:
    """Which player resolves first: chasing > reinforcing > smaller army."""
    h, w = st.armies.shape
    p0, r0, c0, d0 = (int(actions[0][0]), int(actions[0][1]),
                      int(actions[0][2]), _clip(int(actions[0][3]), 3))
    p1, r1, c1, d1 = (int(actions[1][0]), int(actions[1][1]),
                      int(actions[1][2]), _clip(int(actions[1][3]), 3))

    only_p0_passes = bool(p0 & ~p1)

    di0, dj0 = r0 + DIRS[d0][0], c0 + DIRS[d0][1]
    di1, dj1 = r1 + DIRS[d1][0], c1 + DIRS[d1][1]

    p0_chasing = di0 == r1 and dj0 == c1
    p1_chasing = di1 == r0 and dj1 == c0

    p0_reinforcing = bool(st.own[0, _clip(di0, h - 1), _clip(dj0, w - 1)])
    p1_reinforcing = bool(st.own[1, _clip(di1, h - 1), _clip(dj1, w - 1)])

    army0 = int(st.armies[_clip(r0, h - 1), _clip(c0, w - 1)])
    army1 = int(st.armies[_clip(r1, h - 1), _clip(c1, w - 1)])

    tie_chase = p0_chasing == p1_chasing
    tie_reinforce = p0_reinforcing == p1_reinforcing
    p1_first = (
        (p1_chasing and not p0_chasing)
        or (tie_chase and p1_reinforcing and not p0_reinforcing)
        or (tie_chase and tie_reinforce and army1 < army0)
        or only_p0_passes
    )
    return 1 if p1_first else 0


def _apply_build(st: State, p: int, action) -> None:
    if int(action[0]) != BUILD:
        return
    h, w = st.armies.shape
    r, c = int(action[1]), int(action[2])
    if not (0 <= r < h and 0 <= c < w):
        return
    rs, cs = _clip(r, h - 1), _clip(c, w - 1)
    if not st.own[p, rs, cs]:
        return
    if st.generals[rs, cs] or st.castles[rs, cs]:
        return
    if st.winner >= 0:
        return
    cost = int(rules.build_cost_grid((st.castles | st.generals) & st.own[p])[rs, cs])
    if int(st.armies[rs, cs]) < cost:
        return
    st.armies[rs, cs] -= cost
    st.castles[rs, cs] = True


def _transfer(st: State) -> None:
    winner = st.winner
    loser = 1 - winner
    loser_cells = st.own[loser].copy()
    st.own[winner] |= loser_cells
    st.own[loser][:] = False
    st.neutral &= ~loser_cells


def _global_update(st: State) -> None:
    owned = st.own[0].astype(np.int32) + st.own[1].astype(np.int32)
    if st.time % 50 == 0:
        st.armies += owned
    if st.time % 2 == 0:
        st.armies += (st.generals | st.castles).astype(np.int32) * owned


def step(st: State, action0, action1, deathtouch_turn: int = rules.DEATHTOUCH_TURN) -> bool:
    """Advance one turn in place. Returns True when the game is over."""
    actions = [list(action0), list(action1)]

    _apply_build(st, 0, actions[0])
    _apply_build(st, 1, actions[1])
    for i in (0, 1):
        if int(actions[i][0]) == BUILD:
            actions[i] = [PASS, 0, 0, 0, 0]

    prev_winner = st.winner
    active = prev_winner < 0 and st.time >= deathtouch_turn

    first = _determine_order(st, actions)
    second = 1 - first

    t_first = _touches_general(st, first, actions[first])
    _execute(st, first, actions[first])
    mid_winner = st.winner
    t_second = _touches_general(st, second, actions[second])
    _execute(st, second, actions[second])
    final_winner = st.winner

    if prev_winner < 0:
        st.time += 1
    if st.winner >= 0:
        _transfer(st)
    else:
        _global_update(st)

    touch = [False, False]
    touch[first] = t_first and active
    touch[second] = t_second and active
    both_captured = prev_winner < 0 and mid_winner >= 0 and final_winner != mid_winner

    both = (touch[0] and touch[1]) or both_captured
    one = (touch[0] != touch[1]) and not both_captured
    if both:
        st.winner = -1
    elif one:
        st.winner = 0 if touch[0] else 1

    if one and st.winner >= 0 and prev_winner < 0:
        _transfer(st)

    return st.winner >= 0 or both


def observe(st: State, p: int) -> Obs:
    """The exact frame the wire protocol would deliver to player `p`."""
    visible = dilate8(st.own[p])
    invisible = ~visible
    structures = st.mountains | st.castles

    h, w = st.armies.shape
    type_grid = np.full((h, w), rules.T_PLAIN, dtype=np.int8)
    type_grid[invisible & ~structures] = rules.T_FOG
    type_grid[invisible & structures] = rules.T_STRUCTURE_IN_FOG
    type_grid[st.mountains & visible] = rules.T_MOUNTAIN
    type_grid[st.castles & visible] = rules.T_CASTLE
    type_grid[st.generals & visible] = rules.T_GENERAL

    owner_grid = np.zeros((h, w), dtype=np.int8)
    owner_grid[st.own[p] & visible] = rules.OWNER_ME
    owner_grid[st.own[1 - p] & visible] = rules.OWNER_OPP

    land = st.land()
    army = st.army()
    return Obs(
        H=h, W=w, turn=int(st.time),
        my_land=land[p], my_army=army[p],
        opp_land=land[1 - p], opp_army=army[1 - p],
        type_grid=type_grid,
        owner_grid=owner_grid,
        army_grid=(st.armies * visible).astype(np.int32),
    )
