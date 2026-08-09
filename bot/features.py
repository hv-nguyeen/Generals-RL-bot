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
BASE_C = 24                   # one-frame channels; old checkpoints use this
MEMORY_C = 16                 # observation-only temporal channels
C = BASE_C + MEMORY_C
DIRS_N = 4
SPLITS = 2
BUILD_OFFSET = DIRS_N * SPLITS      # slot 8 of a cell is "build a castle here"
PER_CELL = DIRS_N * SPLITS + 1      # 8 moves + 1 build per cell
N_ACTIONS = PAD * PAD * PER_CELL + 1
PASS_INDEX = N_ACTIONS - 1

(MINE, OPP, NEUTRAL, FOG, MOUNTAIN, CASTLE,
 MY_GEN, OPP_GEN, ARMY_MINE, ARMY_OPP, ARMY_NEUTRAL, VALID,
 # broadcast scalars, constant over the board; CLOCK..HIDDEN_OPP must stay the
 # LAST channels and stay contiguous — `encode` fills them as one slice.
 CLOCK, PARITY, GROW_PHASE, DEATHTOUCH,
 ARMY_TOTAL_MINE, ARMY_TOTAL_OPP, LAND_MINE, LAND_OPP,
 GARRISON, HIDDEN_OPP, MAX_STACK_MINE, MAX_STACK_OPP) = range(BASE_C)

(MEM_MINE, MEM_OPP, MEM_NEUTRAL, MEM_ARMY_MINE, MEM_ARMY_OPP,
 MEM_STALENESS, EVER_SEEN, EVER_ENEMY, ENEMY_CASTLE,
 MY_GAINED, OPP_GAINED, MY_ARMY_DELTA, OPP_ARMY_DELTA,
 DELTA_MY_ARMY, DELTA_OPP_ARMY, DELTA_LAND_ADV) = range(BASE_C, C)


def scalar_features(turn, my_army, opp_army, my_land, opp_land,
                    garrison, hidden_opp, max_mine, max_opp, log1p=np.log1p):
    """The twelve broadcast scalars, in channel order, for CLOCK..MAX_STACK_OPP.

    ONE definition called by both encoders — `bot.features.encode` with
    `np.log1p` and `learn.rlenv.encode_jax` with `jnp.log1p` — because two
    hand-written copies of an eight-element tuple in channel order is exactly
    the drift `tools.verify_engine --encoders` exists to catch, and not writing
    it twice is cheaper than catching it.

    WHY these five numbers, all of which the wire protocol carries and the
    encoder used to throw away:

      * the clock. Deathtouch at turn 800 changes the win condition
        discontinuously, structures grow on even ticks, every tile grows every
        50, and the game is a DRAW at 1200. A network with no clock cannot tell
        turn 100 from turn 790 and so cannot predict a draw, which is a
        mechanical reason a critic stalls at competition distance where games
        end by timeout and crosses at distance 2-6 where they end decisively.
      * the totals. `opp_army` counts the opponent's FOGGED tiles too, so
        `opp_army - (visible enemy army)` is the hidden army the heuristic's
        `belief.hidden_enemy_army` runs on — by construction not computable from
        the board the network sees.
      * GARRISON and HIDDEN_OPP, added 2026-08-05 off ladder forensics. Measured
        at the tick our general is EMPTIED — not the tick it falls, which is
        tautological — hidden enemy army exceeds the garrison in 10 of 10 losses
        by 1.2x to 7.7x, while the VISIBLE threat within 3 tiles is ~0 in 9 of
        10; the killer arrives 7-36 turns later. Over a larger batch the median
        hidden army is 188 in losses and 28 in wins, and garrison alone does NOT
        discriminate (37.5 vs 34). The bot empties its general because the board
        LOOKS safe.

        The paragraph above already said this quantity was "by construction not
        computable from the board the network sees" and it was left uncomputed
        for the whole project. Both halves were always here: `opp_army` is the
        scoreboard total and the visible sum is one reduction over the encoding.

        Passed in rather than derived here because the two encoders hold the
        board differently — numpy grids in `features.encode`, starter-kit planes
        in `rlenv.encode_jax` — and this function must stay their one shared
        definition of channel order. `tools.verify_engine --encoders` compares
        them and would catch drift.

        The RATIO comes free: log1p(hidden) - log1p(garrison) is the log ratio,
        a linear combination of two inputs, which the first conv computes itself.
        No hand-built ratio channel.
      * MAX_STACK_MINE / MAX_STACK_OPP, added the same day off the same replays.
        At the tick our general is emptied, the opponent's biggest VISIBLE stack
        is a median of 16 in wins and 49 in losses (3.1x), and ours is 13 vs 6
        (2.2x). For scale: hidden army separates 6.8x, and two other candidates
        measured that night -- hidden-army DENSITY (1.5x) and BFS distance to the
        nearest reserve (1.4x) -- were rejected for being this weak.

        A global MAXIMUM is not something this architecture can reach. The army
        planes carry it per cell, but the critic mean-pools (so it gets the
        average) and the policy head is a 3x3 conv (so it sees only locally).
        "Their biggest stack anywhere is 49" is unavailable at any width.

        Ours EXCLUDES the general, because GARRISON already carries that cell and
        a max that is usually just the garrison would say nothing else.
    """
    return (turn / rules.TURN_LIMIT,
            turn % 2,                                  # structures_grow
            (turn % 50) / 50.0,                        # all_grow phase
            (turn >= rules.DEATHTOUCH_TURN) * 1.0,
            log1p(my_army) / 6.0,
            log1p(opp_army) / 6.0,
            log1p(my_land) / 6.0,
            log1p(opp_land) / 6.0,
            log1p(garrison) / 6.0,
            log1p(hidden_opp) / 6.0,
            log1p(max_mine) / 6.0,
            log1p(max_opp) / 6.0)


def memory_planes(obs: Obs, memory) -> np.ndarray:
    """Build the temporal suffix after ``memory.update(obs)``."""
    p = np.zeros((MEMORY_C, PAD, PAD), dtype=np.float32)
    h, w = obs.H, obs.W
    mine = memory.mem_owner == rules.OWNER_ME
    opp = memory.mem_owner == rules.OWNER_OPP
    neutral = memory.ever_seen & ~(mine | opp)
    p[MEM_MINE - BASE_C, :h, :w] = mine
    p[MEM_OPP - BASE_C, :h, :w] = opp
    p[MEM_NEUTRAL - BASE_C, :h, :w] = neutral
    la = np.log1p(np.maximum(memory.mem_army, 0).astype(np.float32)) / 6.0
    p[MEM_ARMY_MINE - BASE_C, :h, :w] = np.where(mine, la, 0.0)
    p[MEM_ARMY_OPP - BASE_C, :h, :w] = np.where(opp, la, 0.0)
    age = np.where(memory.ever_seen,
                   np.clip((obs.turn - memory.mem_turn) / 200.0, 0.0, 1.0),
                   1.0)
    p[MEM_STALENESS - BASE_C, :h, :w] = age
    p[EVER_SEEN - BASE_C, :h, :w] = memory.ever_seen
    p[EVER_ENEMY - BASE_C, :h, :w] = memory.ever_enemy
    p[ENEMY_CASTLE - BASE_C, :h, :w] = memory.enemy_castles
    p[MY_GAINED - BASE_C, :h, :w] = memory.my_gained
    p[OPP_GAINED - BASE_C, :h, :w] = memory.opp_gained
    p[MY_ARMY_DELTA - BASE_C, :h, :w] = np.tanh(memory.my_army_delta / 16.0)
    p[OPP_ARMY_DELTA - BASE_C, :h, :w] = np.tanh(memory.opp_army_delta / 16.0)
    p[DELTA_MY_ARMY - BASE_C, :h, :w] = np.tanh(memory.delta_my_army / 50.0)
    p[DELTA_OPP_ARMY - BASE_C, :h, :w] = np.tanh(memory.delta_opp_army / 50.0)
    p[DELTA_LAND_ADV - BASE_C, :h, :w] = np.tanh(memory.delta_land_adv / 10.0)
    return p


def encode(obs: Obs, memory=None) -> np.ndarray:
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
    x[MOUNTAIN, :h, :w] = (memory.known_mountains if memory is not None else
                            ((t == rules.T_MOUNTAIN) |
                             (t == rules.T_STRUCTURE_IN_FOG)))
    x[CASTLE, :h, :w] = t == rules.T_CASTLE
    x[MY_GEN, :h, :w] = (t == rules.T_GENERAL) & mine
    x[OPP_GEN, :h, :w] = (t == rules.T_GENERAL) & opp

    # log-compressed army, split by owner so the network never has to subtract
    la = np.log1p(np.maximum(a, 0).astype(np.float32)) / 6.0
    x[ARMY_MINE, :h, :w] = np.where(mine, la, 0.0)
    x[ARMY_OPP, :h, :w] = np.where(opp, la, 0.0)
    x[ARMY_NEUTRAL, :h, :w] = np.where(~mine & ~opp, la, 0.0)
    x[VALID, :h, :w] = 1.0

    # Broadcast over the real board only, so the padding stays all-zero in every
    # channel: `learn.train.augment` reads the bottom-right pad cell for dst
    # cells outside a rotated non-square board and expects zeros there.
    # `opp_army` is the scoreboard total and includes the opponent's fogged
    # tiles; subtracting what we can actually see leaves the army that is out
    # there unaccounted for. Clamped at 0 because the scoreboard and the
    # observation are sampled at the same tick but nothing guarantees it.
    visible_opp = float(np.where(opp, a, 0).sum())
    hidden_opp = max(float(obs.opp_army) - visible_opp, 0.0)
    is_gen = t == rules.T_GENERAL
    garrison = float(np.where(mine & is_gen, a, 0).sum())
    # Ours excludes the general: GARRISON already carries that cell, and a max
    # that is usually just the garrison would carry no extra information.
    max_mine = float(np.where(mine & ~is_gen, a, 0).max()) if (mine & ~is_gen).any() else 0.0
    max_opp = float(np.where(opp, a, 0).max()) if opp.any() else 0.0
    x[CLOCK:BASE_C, :h, :w] = np.array(
        scalar_features(obs.turn, obs.my_army, obs.opp_army,
                        obs.my_land, obs.opp_land, garrison, hidden_opp,
                        max_mine, max_opp),
        dtype=np.float32)[:, None, None]
    if memory is not None:
        x[BASE_C:] = memory_planes(obs, memory)
    return x


def action_to_index(action) -> int:
    """Wire action -> flat class.

    `idx = (r*21 + c)*9 + (8 if build else d*2 + split)`, pass last. A build is
    position-only: dir and split are ignored by the engine (see
    `generals/modifiers/build_castles.py`), so it gets one slot per cell and not
    eight. Builds used to fold into PASS here, which trained the clone to pass
    on exactly the positions where a strong player built a castle.
    """
    kind, r, c, d, split = (int(v) for v in action)
    if not (0 <= r < PAD and 0 <= c < PAD):
        return PASS_INDEX
    if kind == rules.BUILD:
        return ((r * PAD + c) * PER_CELL) + BUILD_OFFSET
    if kind != rules.MOVE or not 0 <= d < DIRS_N:
        return PASS_INDEX
    return ((r * PAD + c) * PER_CELL) + d * SPLITS + (1 if split else 0)


def index_to_action(idx: int):
    if idx >= PASS_INDEX:
        return rules.PASS_ACTION
    cell, rest = divmod(int(idx), PER_CELL)
    r, c = divmod(cell, PAD)
    if rest == BUILD_OFFSET:
        return (rules.BUILD, r, c, 0, 0)
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
            # At army 2 split/full both move one and are duplicate actions. At
            # army 3 they first differ (1 versus 2), so split is legal/useful.
            if a[r, c] >= 3:
                m[base + 1] = True

    # Builds. Computable from a FOGGED observation with no hidden information,
    # and that is provable rather than lucky: visibility is dilate8(own), so
    # every cell we own is visible to us, and the engine's price depends only on
    # `(castles | generals) & ownership[me]` -- enemy structures never enter it.
    mine = o == rules.OWNER_ME
    structs = mine & ((t == rules.T_GENERAL) | (t == rules.T_CASTLE))
    can = mine & ~structs & (a >= rules.build_cost_grid(structs))
    m[:PAD * PAD * PER_CELL].reshape(PAD, PAD, PER_CELL)[:h, :w, BUILD_OFFSET] = can
    return m
