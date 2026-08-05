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


def encode_jax(obs, valid_h, valid_w):
    """(C, 21, 21) float32 from a starter-kit Observation. Mirrors features.encode."""
    import jax.numpy as jnp

    mine = obs.owned_cells.astype(jnp.float32)
    opp = obs.opponent_cells.astype(jnp.float32)
    sif = obs.structures_in_fog.astype(jnp.float32)
    fog = jnp.clip(obs.fog_cells.astype(jnp.float32) + sif, 0, 1)
    mountain = jnp.clip(obs.mountains.astype(jnp.float32) + sif, 0, 1)

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
    splittable = obs.owned_cells & (obs.armies >= 4)
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
