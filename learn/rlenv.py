"""JAX side of the RL loop: the starter kit's vectorised env, encoded our way.

The one thing that must not drift is the observation encoding. `bot/features.py`
is what runs inside the submission; this is what the policy sees during
training. If they disagree by so much as a channel order, a policy that trains
beautifully plays like noise in a real match, and nothing about the failure
points at the encoder.

The engine pads every board out to 21x21 with mountains, so the padded region
has to be recovered to fill the VALID channel the same way the wire protocol
does — the network must be able to tell "board edge" from "wall".

    python -m tools.verify_engine --encoders     # proves the mirrors agree

That is the only proof, and it steps real games rather than looking at turn 0.
This module used to carry its own `verify()`: turn-0 only, 21x21 only (so four
of the twelve channels were identically zero), not jittable, never run by any
test, and it did not look at the legal mask or the action map at all. It was
deleted rather than kept beside a stronger check, because a weak check next to
a strong one is a check somebody runs by mistake.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "generals-bots"))

from bot import features, rules                              # noqa: E402


def _dilate4_jax(mask):
    """No-wrap four-neighbour dilation over (..., H, W)."""
    import jax.numpy as jnp

    p = jnp.pad(mask, [(0, 0)] * (mask.ndim - 2) + [(1, 1), (1, 1)])
    h, w = mask.shape[-2:]
    return (mask | p[..., :h, 1:w + 1] | p[..., 2:h + 2, 1:w + 1]
            | p[..., 1:h + 1, :w] | p[..., 1:h + 1, 2:w + 2])


def _bfs_field_jax(passable, sources):
    """Fixed-shape JAX mirror of ``bot.board.bfs_field``."""
    import jax
    import jax.numpy as jnp

    # The exported feature clips distance at 2*PAD, and the spawn/fallback
    # thresholds (17/8) are both below it. Computing past that cap cannot change
    # any encoded value or candidate decision, but a 441-step loop made the
    # first JAX memory update operationally unusable.
    inf = 2 * features.PAD
    reached = sources & passable
    dist = jnp.where(reached, 0, inf).astype(jnp.int32)

    def body(i, state):
        reached, dist = state
        grown = _dilate4_jax(reached) & passable
        new = grown & ~reached
        return grown, jnp.where(new, i + 1, dist)

    return jax.lax.fori_loop(0, inf, body, (reached, dist))[1]


def empty_memory_jax(batch: int):
    """Zero temporal state for ``batch`` observations."""
    import jax.numpy as jnp

    board_b = jnp.zeros((batch, features.PAD, features.PAD), jnp.bool_)
    board_i = jnp.zeros((batch, features.PAD, features.PAD), jnp.int32)
    return {
        "initialised": jnp.zeros((batch,), jnp.bool_),
        "turn": jnp.full((batch,), -1, jnp.int32),
        "known_mountains": board_b,
        "dist_home": jnp.full_like(board_i, features.PAD * features.PAD),
        "general_candidates": board_b,
        "mem_owner": board_i,
        "mem_army": board_i,
        "mem_turn": jnp.full_like(board_i, -1),
        "ever_seen": board_b,
        "ever_enemy": board_b,
        "enemy_castles": board_b,
        "my_gained": board_b,
        "opp_gained": board_b,
        "my_army_delta": board_i,
        "opp_army_delta": board_i,
        "my_army": jnp.zeros((batch,), jnp.int32),
        "opp_army": jnp.zeros((batch,), jnp.int32),
        "my_land": jnp.zeros((batch,), jnp.int32),
        "opp_land": jnp.zeros((batch,), jnp.int32),
        "delta_my_army": jnp.zeros((batch,), jnp.int32),
        "delta_opp_army": jnp.zeros((batch,), jnp.int32),
        "delta_land_adv": jnp.zeros((batch,), jnp.int32),
    }


def update_memory_jax(memory, obs):
    """Unbatched JAX mirror of ``bot.memory.TemporalMemory.update``."""
    import jax
    import jax.numpy as jnp

    visible = ~(obs.fog_cells | obs.structures_in_fog)
    mine, opp = obs.owned_cells, obs.opponent_cells
    owner = jnp.where(mine, 1, jnp.where(opp, 2, 0)).astype(jnp.int32)
    known_mountains = jnp.where(
        memory["initialised"], memory["known_mountains"],
        obs.mountains | obs.structures_in_fog)
    known_mountains = ((known_mountains | obs.mountains)
                       & ~(visible & ~obs.mountains))

    def bootstrap(_):
        passable = ~known_mountains
        general = obs.generals & mine
        fallback_idx = jnp.argmax(mine.reshape(-1).astype(jnp.int32))
        fallback = (jnp.arange(features.PAD * features.PAD) == fallback_idx).reshape(
            features.PAD, features.PAD)
        source = jnp.where(general.any(), general, fallback) & passable
        dist = _bfs_field_jax(passable, source)
        # A conservative superset of the generator prior. The full generator
        # also matches radius-7 room counts; omitting that expensive all-source
        # field never removes the true general and keeps vector rollout cheap.
        candidates = passable & (dist >= rules.MIN_GENERALS_DISTANCE)
        return dist, candidates

    dist_home, general_candidates = jax.lax.cond(
        memory["initialised"],
        lambda _: (memory["dist_home"], memory["general_candidates"]),
        bootstrap, operand=None)
    changed = visible & memory["ever_seen"]
    my_gained = changed & mine & (memory["mem_owner"] != 1)
    opp_gained = changed & opp & (memory["mem_owner"] != 2)
    delta = obs.armies.astype(jnp.int32) - memory["mem_army"]
    my_army_delta = jnp.where(changed & mine, delta, 0)
    opp_army_delta = jnp.where(changed & opp, delta, 0)
    # Match TemporalMemory: a post-initial structure token that is not a known
    # mountain is an enemy castle even if the cell was never previously seen.
    enemy_castles = (memory["enemy_castles"]
                     | (obs.structures_in_fog & ~known_mountains)
                     | (obs.castles & opp))
    enemy_castles &= ~(obs.castles & ~opp)
    mem_owner = jnp.where(visible, owner, memory["mem_owner"])
    mem_army = jnp.where(visible, obs.armies.astype(jnp.int32), memory["mem_army"])
    mem_turn = jnp.where(visible, obs.timestep.astype(jnp.int32), memory["mem_turn"])
    ever_seen = memory["ever_seen"] | visible
    ever_enemy = memory["ever_enemy"] | (visible & opp)
    seen_general = obs.generals & opp
    general_candidates = jnp.where(
        seen_general.any(), seen_general, general_candidates & ~visible)
    fallback_candidates = (~known_mountains & ~ever_seen
                           & (dist_home >= rules.MIN_GENERALS_DISTANCE // 2))
    general_candidates = jnp.where(
        general_candidates.any(), general_candidates, fallback_candidates)
    seen_before = memory["turn"] >= 0
    dma = jnp.where(seen_before, obs.owned_army_count - memory["my_army"], 0)
    doa = jnp.where(seen_before, obs.opponent_army_count - memory["opp_army"], 0)
    dla = jnp.where(
        seen_before,
        (obs.owned_land_count - memory["my_land"])
        - (obs.opponent_land_count - memory["opp_land"]), 0)
    updated = {
        "initialised": jnp.bool_(True),
        "turn": obs.timestep.astype(jnp.int32),
        "known_mountains": known_mountains,
        "dist_home": dist_home,
        "general_candidates": general_candidates,
        "mem_owner": mem_owner,
        "mem_army": mem_army,
        "mem_turn": mem_turn,
        "ever_seen": ever_seen,
        "ever_enemy": ever_enemy,
        "enemy_castles": enemy_castles,
        "my_gained": my_gained,
        "opp_gained": opp_gained,
        "my_army_delta": my_army_delta,
        "opp_army_delta": opp_army_delta,
        "my_army": obs.owned_army_count.astype(jnp.int32),
        "opp_army": obs.opponent_army_count.astype(jnp.int32),
        "my_land": obs.owned_land_count.astype(jnp.int32),
        "opp_land": obs.opponent_land_count.astype(jnp.int32),
        "delta_my_army": dma.astype(jnp.int32),
        "delta_opp_army": doa.astype(jnp.int32),
        "delta_land_adv": dla.astype(jnp.int32),
    }
    same = obs.timestep.astype(jnp.int32) == memory["turn"]
    return {k: jnp.where(same, memory[k], v) for k, v in updated.items()}


def memory_planes_jax(obs, memory, valid):
    import jax.numpy as jnp

    mine = memory["mem_owner"] == 1
    opp = memory["mem_owner"] == 2
    neutral = memory["ever_seen"] & ~(mine | opp)
    la = jnp.log1p(jnp.maximum(memory["mem_army"], 0).astype(jnp.float32)) / 6.0
    age = jnp.where(
        memory["ever_seen"],
        jnp.clip((obs.timestep - memory["mem_turn"]) / 200.0, 0.0, 1.0), 1.0)
    ones = jnp.ones_like(valid)
    out = jnp.stack([
        mine, opp, neutral, la * mine, la * opp, age,
        memory["ever_seen"], memory["ever_enemy"], memory["enemy_castles"],
        memory["my_gained"], memory["opp_gained"],
        jnp.tanh(memory["my_army_delta"] / 16.0),
        jnp.tanh(memory["opp_army_delta"] / 16.0),
        ones * jnp.tanh(memory["delta_my_army"] / 50.0),
        ones * jnp.tanh(memory["delta_opp_army"] / 50.0),
        ones * jnp.tanh(memory["delta_land_adv"] / 10.0),
    ]).astype(jnp.float32)
    return out * valid[None]


def strategic_planes_jax(obs, memory, valid):
    """JAX mirror of ``features.strategic_planes``."""
    import jax.numpy as jnp

    zeros = jnp.zeros_like(valid)
    prior = (memory["general_candidates"].astype(jnp.float32)
             if memory is not None else zeros)
    dist = (jnp.clip(memory["dist_home"].astype(jnp.float32)
                     / (2.0 * features.PAD), 0.0, 1.0)
            * (~memory["known_mountains"]).astype(jnp.float32)
            if memory is not None else zeros)
    structures = obs.owned_cells & (obs.generals | obs.castles)
    cost = jnp.log1p(build_cost_grid_jax(structures).astype(jnp.float32)) / 6.0
    return jnp.stack([prior, dist, cost]) * valid[None]


def encode_jax(obs, valid_h, valid_w, memory=None):
    """(C, 21, 21) float32 from a starter-kit Observation. Mirrors features.encode."""
    import jax.numpy as jnp

    mine = obs.owned_cells.astype(jnp.float32)
    opp = obs.opponent_cells.astype(jnp.float32)
    sif = obs.structures_in_fog.astype(jnp.float32)
    fog = jnp.clip(obs.fog_cells.astype(jnp.float32) + sif, 0, 1)
    mountain = (memory["known_mountains"].astype(jnp.float32) if memory is not None
                else jnp.clip(obs.mountains.astype(jnp.float32) + sif, 0, 1))

    la = jnp.log1p(jnp.maximum(obs.armies, 0).astype(jnp.float32)) / 6.0
    rows = jnp.arange(features.PAD)[:, None] < valid_h
    cols = jnp.arange(features.PAD)[None, :] < valid_w
    valid = (rows & cols).astype(jnp.float32)

    # The broadcast scalars come from features.scalar_features itself rather than
    # being re-derived here: they are eight expressions in a fixed channel order,
    # and a second copy of that order is the kind of drift that trains cleanly.
    # Same two reductions as `features.encode`, over planes instead of grids.
    visible_opp = (obs.armies * obs.opponent_cells).sum()
    hidden_opp = jnp.maximum(obs.opponent_army_count - visible_opp, 0.0)
    garrison = (obs.armies * obs.generals * obs.owned_cells).sum()
    # Ours excludes the general, matching `features.encode`: GARRISON already
    # carries that cell. A global max is unreachable for the architecture --
    # the critic mean-pools and the policy head is a 3x3 conv.
    max_mine = (obs.armies * obs.owned_cells * (1 - obs.generals)).max()
    max_opp = (obs.armies * obs.opponent_cells).max()

    ones = jnp.ones_like(valid)
    scalars = [(ones * s).astype(jnp.float32) for s in features.scalar_features(
        obs.timestep, obs.owned_army_count, obs.opponent_army_count,
        obs.owned_land_count, obs.opponent_land_count,
        garrison, hidden_opp, max_mine, max_opp, jnp.log1p)]

    x = jnp.stack([
        mine,
        opp,
        obs.neutral_cells.astype(jnp.float32),
        fog,
        mountain,
        obs.castles.astype(jnp.float32),
        (obs.generals * obs.owned_cells).astype(jnp.float32),
        (obs.generals * obs.opponent_cells).astype(jnp.float32),
        la * mine,
        la * opp,
        la * (1.0 - mine - opp),
        valid,
        *scalars,
    ])
    if memory is None:
        memory_x = jnp.zeros((features.MEMORY_C, features.PAD, features.PAD),
                             jnp.float32)
    else:
        memory_x = memory_planes_jax(obs, memory, valid)
    strategy_x = strategic_planes_jax(obs, memory, valid)
    x = jnp.concatenate([x, memory_x, strategy_x])
    # everything outside the real board is padding, not board state
    return x * valid[None]


def board_dims(state):
    """Recover the true (h, w) from a padded state.

    generate_grid pads bottom and right with mountains, so the real board is the
    largest prefix of rows/columns that is not entirely impassable.
    """
    import jax.numpy as jnp

    passable = state.passable
    h = jnp.max(jnp.where(passable.any(axis=1), jnp.arange(passable.shape[0]), -1)) + 1
    w = jnp.max(jnp.where(passable.any(axis=0), jnp.arange(passable.shape[1]), -1)) + 1
    return h, w


def index_to_engine_action(idx):
    """Flat class -> the engine's [pass, row, col, dir, split] array.

    Mirrors features.index_to_action. Slot 8 of a cell is a build, whose dir and
    split fields the engine ignores; they are forced to 0 anyway so a build and
    its numpy twin compare equal element for element.
    """
    import jax.numpy as jnp

    is_pass = idx >= features.PASS_INDEX
    cell = idx // features.PER_CELL
    rest = idx % features.PER_CELL
    is_build = rest >= features.BUILD_OFFSET
    r = cell // features.PAD
    c = cell % features.PAD
    d = jnp.where(is_build, 0, rest // features.SPLITS)
    split = jnp.where(is_build, 0, rest % features.SPLITS)
    kind = jnp.where(is_build, jnp.int32(2), jnp.int32(0))
    return jnp.where(
        is_pass,
        jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32),
        jnp.stack([kind, r.astype(jnp.int32), c.astype(jnp.int32),
                   d.astype(jnp.int32), split.astype(jnp.int32)]),
    )


def build_cost_grid_jax(structures):
    """(H, W) castle price for the player owning `structures`. Mirrors
    rules.build_cost_grid and generals/modifiers/build_castles.build_cost_grid.

    85 shifted adds, unrolled at trace time, and deliberately NOT one 13x13
    convolution: that kernel is symmetric under transpose, so the exact
    weight-layout bug that once cost this repo a 200-0 result would be invisible
    in it. Shifted adds have no layout to get wrong.
    """
    import jax.numpy as jnp

    h, w = structures.shape
    r = rules.SURCHARGE_RADIUS
    padded = jnp.pad(structures.astype(jnp.int32), r)
    cost = jnp.full((h, w), rules.BASE_COST, dtype=jnp.int32)
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            surcharge = rules.PROXIMITY_PENALTY - rules.PROXIMITY_DECAY * (abs(di) + abs(dj))
            if surcharge > 0:
                cost = cost + surcharge * padded[r + di:r + di + h, r + dj:r + dj + w]
    return cost


def legal_mask_jax(obs):
    """Bool mask over the flat action space, built from the same rules as
    features.legal_mask so training and match time agree on what is playable.

    Asserted equal to it — on real fogged observations from real games, with
    coverage counters on the build half — by `tools.verify_engine.check_encoders`,
    which `make test` and `make verify` both run. This docstring used to claim
    the equality and nothing compared them.

    No `valid_h`/`valid_w` argument, and the padded region needs no special
    case. Visibility IS dilate8(own), so every destination this mask can reach
    is adjacent to a cell we own and therefore visible: `structures_in_fog` is
    identically False on it, and a 21x21 pad cell next to the real board is a
    visible mountain. Both exclusions are already implied by `~obs.mountains`.
    """
    import jax.numpy as jnp

    movable = obs.owned_cells & (obs.armies >= 2)
    # Split first differs from full at army 3: floor(3/2)=1 versus 3-1=2.
    splittable = obs.owned_cells & (obs.armies >= 3)
    passable = ~obs.mountains & ~obs.structures_in_fog

    def shifted(mask, dr, dc):
        out = jnp.roll(mask, shift=(-dr, -dc), axis=(0, 1))
        if dr == 1:
            out = out.at[-1, :].set(False)
        elif dr == -1:
            out = out.at[0, :].set(False)
        if dc == 1:
            out = out.at[:, -1].set(False)
        elif dc == -1:
            out = out.at[:, 0].set(False)
        return out

    planes = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        dest_ok = shifted(passable, dr, dc)
        planes.append(movable & dest_ok)
        planes.append(splittable & dest_ok)

    # slot BUILD_OFFSET: a castle here. Own general and own castles are excluded
    # (the engine's `plain` test) and they are also what sets the price.
    structs = obs.owned_cells & (obs.generals | obs.castles)
    planes.append(obs.owned_cells & ~structs
                  & (obs.armies >= build_cost_grid_jax(structs)))

    # (H, W, PER_CELL) in the same order features.action_to_index uses
    flat = jnp.stack(planes, axis=-1).reshape(-1)
    return jnp.concatenate([flat, jnp.array([True])])
