"""JAX side of the RL loop: the starter kit's vectorised env, encoded our way.

The one thing that must not drift is the observation encoding. `bot/features.py`
is what runs inside the submission; this is what the policy sees during
training. If they disagree by so much as a channel order, a policy that trains
beautifully plays like noise in a real match, and nothing about the failure
points at the encoder. `verify()` asserts they agree exactly, and is run as a
test rather than trusted.

The engine pads every board out to 21x21 with mountains, so the padded region
has to be recovered to fill the VALID channel the same way the wire protocol
does — the network must be able to tell "board edge" from "wall".

    python -m learn.rlenv            # checks the two encoders agree
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "generals-bots"))

from bot import features                                     # noqa: E402


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
    """Flat class -> the engine's [pass, row, col, dir, split] array."""
    import jax.numpy as jnp

    is_pass = idx >= features.PASS_INDEX
    cell = idx // features.PER_CELL
    rest = idx % features.PER_CELL
    r = cell // features.PAD
    c = cell % features.PAD
    d = rest // features.SPLITS
    split = rest % features.SPLITS
    return jnp.where(
        is_pass,
        jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32),
        jnp.stack([jnp.int32(0), r.astype(jnp.int32), c.astype(jnp.int32),
                   d.astype(jnp.int32), split.astype(jnp.int32)]),
    )


def legal_mask_jax(obs):
    """Bool mask over the flat action space, built from the same rules as
    features.legal_mask so training and match time agree on what is playable."""
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

    per_dir = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        dest_ok = shifted(passable, dr, dc)
        per_dir.append(movable & dest_ok)
        per_dir.append(splittable & dest_ok)
    # (H, W, 8) in the same order features.action_to_index uses
    stacked = jnp.stack(per_dir, axis=-1)
    order = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
    flat = stacked[..., order].reshape(-1)
    return jnp.concatenate([flat, jnp.array([True])])


def verify(n: int = 6) -> None:
    """Assert the JAX encoder equals the numpy one used in the submission."""
    import jax.numpy as jnp
    import jax.random as jr
    from generals import GeneralsEnv, get_observation

    from sim import engine as sim_engine

    env = GeneralsEnv(mode="competition")
    key = jr.PRNGKey(0)
    worst = 0.0
    for i in range(n):
        state = env.init_state(jr.fold_in(key, i))
        h, w = board_dims(state)
        jobs = get_observation(state, 0)
        xj = np.asarray(encode_jax(jobs, h, w))

        # the same position through the submission's own path
        grid = np.where(np.asarray(state.mountains), -2, 0).astype(np.int32)
        gp = np.asarray(state.general_positions)
        grid[gp[0][0], gp[0][1]] = 1
        grid[gp[1][0], gp[1][1]] = 2
        st = sim_engine.from_grid(grid[:int(h), :int(w)])
        xn = features.encode(sim_engine.observe(st, 0))

        worst = max(worst, float(np.abs(xj - xn).max()))
    print(f"max channel disagreement over {n} boards: {worst:.2e}")
    assert worst < 1e-5, "JAX and numpy encoders disagree — BC weights will not transfer"
    print("encoders agree")


if __name__ == "__main__":
    verify()
