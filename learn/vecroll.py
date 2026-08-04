"""Vectorised self-play rollouts on the GPU, behind `learn/selfplay.py`'s
existing produce/train boundary.

    python -m learn.vecroll --selfcheck     # shapes and one vmapped step, no games

The CPU path measures ~2.8 games/s because every worker runs `bot/policy/net.py`
in numpy once per seat per turn. This runs the same game N boards wide inside
one jitted step. It replaces exactly one caller of `selfplay.play` -- the
training rollout -- and NOTHING else: comp-eval, stage-eval and the final paired
gate keep running through the process pool, because they play `arena.agents`
opponents that cannot exist inside a JAX kernel, and because they are the
ground truth this path has to be measured against.

WHAT IS DELIBERATELY NOT USED, and why each absence removes a whole failure mode:

  * `GeneralsEnv`. The transition is three lines -- `apply_build_actions` then
    `deathtouch.step` -- and that composition IS what `mode="competition"`
    plays, minus observations, rewards and auto-reset that we compute or do not
    want. Skipping the wrapper deletes: the correlated `pool_idx` (every env
    starts at 0 and increments identically, so 512 envs replay one env's
    boards), the `min=17/max=6` preset trap, per-band recompilation, `reset()`
    mutating `self.pool_size` after the pool is generated, four fog computations
    per step where two suffice, and a ~50 MB pool baked into a jitted closure.
  * `rlenv.board_dims`. It infers (h, w) from `passable.any(axis=...)`, which is
    right only because an all-mountain real row is improbable. The pool carries
    the true h and w for free.
  * auto-reset. Each column plays exactly one game to completion, so an episode
    is a whole column and the segmentation is `(turns, z)` per column -- the
    same contract `selfplay._pack` returns, with no boundary bookkeeping. The
    starter kit's `TimeStep` is not self-consistent on a terminal step
    (`observation` is post-reset, `reward` is pre-reset); there is nowhere for
    that off-by-one to hide here.

WHAT THIS COSTS: the batch runs until the LONGEST game in it finishes, so short
games sit idle. Logged every iteration as `util = sum(turns) / (n * turns_max)`.
# ponytail: no refill-on-death. The upgrade is per-column episode-start indices
# and it is not worth the boundary logic until `util` is measured below ~0.5.

TRUNCATION NEEDS NO BOOTSTRAP, and that is exact rather than an approximation:
the turn limit is a DRAW, mutual capture is a draw, terminal reward is +/-1 or
0, so V(s_T) = 0 by definition and `netoracle.gae`'s scalar-terminal contract
holds unchanged. True only while `--shape` is 0 -- and shaping is already dead
(ml-log attempts 2 and 3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "generals-bots"))

from bot import features, rules                              # noqa: E402
from learn import rlenv                                      # noqa: E402
from learn.selfplay import outcome                           # noqa: E402

MASK_BYTES = -(-features.N_ACTIONS // 8)      # what np.packbits produces
ALIVE_EVERY = 32     # a device sync in the inner loop costs more than the
                     # handful of no-op steps this leaves on the table


def transition(state, actions):
    """One competition tick for one game. == GeneralsEnv(mode="competition").step
    minus observations, reward and auto-reset. `tools/verify_engine.py` steps
    this against `sim/engine.py` and compares every field."""
    from generals.modifiers import build_castles as jbc
    from generals.modifiers import deathtouch as jdt

    state, acts = jbc.apply_build_actions(state, actions)
    return jdt.step(state, acts, rules.DEATHTOUCH_TURN)


def device_pool(pool: dict):
    """Host pool (see tools/pools.py) -> a batched GameState plus h/w on device.

    One `(N, 21, 21)` shape at every curriculum stage, so promoting a stage is
    an argument change and not an XLA retrace.
    """
    import jax
    import jax.numpy as jnp
    from generals.core import game as jgame

    states = jax.vmap(jgame.create_initial_state)(jnp.asarray(pool["grid"], jnp.int32))
    return {"states": states,
            "h": jnp.asarray(pool["h"], jnp.int32),
            "w": jnp.asarray(pool["w"], jnp.int32),
            "dist": np.asarray(pool["dist"])}


def make_step(forward):
    """The whole per-turn kernel: fog, encode, mask, sample, act, step.

    Seats are stacked into one batch -- rows [0:n] are seat 0, rows [n:2n] are
    seat 1 -- so both seats of every game go through ONE forward pass.
    """
    import jax
    import jax.numpy as jnp
    from generals.core import game as jgame

    @jax.jit
    def step(states, h, w, theta, key, t, ended, winner):
        n = h.shape[0]
        obs = jax.tree.map(
            lambda a, b: jnp.concatenate([a, b]),
            jax.vmap(jgame.get_observation, in_axes=(0, None))(states, 0),
            jax.vmap(jgame.get_observation, in_axes=(0, None))(states, 1))
        hh, ww = jnp.concatenate([h, h]), jnp.concatenate([w, w])
        x = jax.vmap(rlenv.encode_jax)(obs, hh, ww)
        mask = jax.vmap(rlenv.legal_mask_jax)(obs)
        # -1e9 and not -inf: this is the masking `netoracle.policy_loss` and
        # `ref_logp_of` use, and the stored logp has to be the density those
        # recompute or the PPO ratio is against a policy that never played.
        logits = jnp.where(mask, forward(theta, x), -1e9)
        a_idx = jax.random.categorical(key, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(2 * n), a_idx]
        acts = jax.vmap(rlenv.index_to_engine_action)(a_idx)
        states, info = jax.vmap(transition)(
            states, jnp.stack([acts[:n], acts[n:]], axis=1))

        # LATCH, do not read at the end. A mutual capture is a draw with
        # winner -1 and is_done True, and the base engine happily keeps playing
        # from there, so a final read would silently turn draws into whatever
        # happened afterwards.
        first = (ended < 0) & info.is_done
        return (states, x.astype(jnp.float16), mask, a_idx, logp,
                jnp.where(first, t, ended), jnp.where(first, info.winner, winner))

    return step


def rollout(step, pool, idx: np.ndarray, theta, key, max_turns: int) -> list[dict]:
    """`n = len(idx)` games to completion. Returns exactly what `selfplay._pack`
    returns, in the same order (seat 0 then seat 1 of each game), so everything
    from `selfplay.py`'s ingest down is untouched.

    Host memory: the step buffers are `T x 2n x 12 x 21 x 21` float16, i.e.
    10.6 kB a sample -- ~2.6 GB for a 256-game stage-5 iteration, and the same
    again in the returned trajectories, which is why the stacked buffer is freed
    before they are cut out of it.
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    n = len(idx)
    states = jax.tree_util.tree_map(lambda a: a[jnp.asarray(idx)], pool["states"])
    h, w = pool["h"][jnp.asarray(idx)], pool["w"][jnp.asarray(idx)]
    ended = jnp.full(n, -1, jnp.int32)
    winner = jnp.full(n, -1, jnp.int32)

    xs, ms, ids, lps = [], [], [], []
    steps = 0
    for t in range(max_turns):
        key, sub = jr.split(key)
        states, x, mask, a_idx, logp, ended, winner = step(
            states, h, w, theta, sub, jnp.int32(t), ended, winner)
        xs.append(np.asarray(x))
        ms.append(np.packbits(np.asarray(mask), axis=1))
        ids.append(np.asarray(a_idx))
        lps.append(np.asarray(logp))
        steps = t + 1
        if t % ALIVE_EVERY == ALIVE_EVERY - 1 and bool((ended >= 0).all()):
            break

    X = np.stack(xs); M = np.stack(ms); I = np.stack(ids); L = np.stack(lps)
    del xs, ms, ids, lps
    ended = np.asarray(ended)
    winner = np.asarray(winner)
    # `turns` counts the tick the game ended ON, matching the CPU rollout: it
    # records the observation, acts, steps, and stops.
    turns = np.where(ended >= 0, ended + 1, steps).astype(int)
    dist = pool["dist"][idx]

    out = []
    for i in range(n):
        T = int(turns[i])
        if T == 0:                      # cannot happen; _pack drops such games
            continue
        for s in (0, 1):                # both seats or neither: the buffer must
            col = s * n + i             # stay exactly zero-sum
            out.append({"z": outcome(int(winner[i]), s), "turns": T,
                        "dist": int(dist[i]), "seat": s, "opp": 0,
                        "x": X[:T, col].copy(), "idx": I[:T, col].copy(),
                        "mask": M[:T, col].copy(), "logp": L[:T, col].copy()})
    return out


def selfcheck() -> None:
    """One vmapped step on four envs and the shapes of everything it produces.

    NOT a game and not a benchmark: four envs, one tick. What it can catch is
    the plumbing -- pool indexing, seat row order, mask packing width, the
    `_pack`-shaped output -- all of which is wrong-in-silence material.
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from learn import train as bc
    from tools import pools

    n = 4
    pool = device_pool(pools.build(0, size=n))
    step = make_step(bc.forward)

    key = jr.PRNGKey(0)
    theta = bc.init_params(key, {"layers": 4, "channels": 8, "residual": False})
    states = jax.tree_util.tree_map(lambda a: a[jnp.arange(n)], pool["states"])
    out = step(states, pool["h"], pool["w"], theta, key, jnp.int32(0),
               jnp.full(n, -1, jnp.int32), jnp.full(n, -1, jnp.int32))
    states2, x, mask, a_idx, logp, ended, winner = out
    assert x.shape == (2 * n, features.C, features.PAD, features.PAD), x.shape
    assert mask.shape == (2 * n, features.N_ACTIONS), mask.shape
    assert a_idx.shape == (2 * n,) and logp.shape == (2 * n,)
    assert np.all(np.asarray(logp) <= 0.0)
    # every sampled action is legal -- the mask is applied BEFORE the sample
    m = np.asarray(mask)
    assert m[np.arange(2 * n), np.asarray(a_idx)].all()
    assert np.packbits(m, axis=1).shape == (2 * n, MASK_BYTES)
    assert int(np.asarray(states2.time)[0]) == 1
    assert np.all(np.asarray(ended) == -1) and np.all(np.asarray(winner) == -1)

    # seat rows: row i and row n + i are the SAME board seen by the two players,
    # so they must differ (own fog) but share the VALID channel
    xn = np.asarray(x)
    for i in range(n):
        assert not np.array_equal(xn[i], xn[n + i]), i
        assert np.array_equal(xn[i, features.VALID], xn[n + i, features.VALID]), i

    # and the whole rollout contract, on a 3-turn stub
    res = rollout(step, pool, np.arange(n), theta, key, max_turns=3)
    assert len(res) == 2 * n, len(res)
    assert [r["seat"] for r in res[:2]] == [0, 1]
    for r in res:
        assert r["turns"] == 3 and len(r["idx"]) == 3
        assert r["x"].shape == (3, features.C, features.PAD, features.PAD)
        assert r["x"].dtype == np.float16 and r["mask"].shape == (3, MASK_BYTES)
        assert r["logp"].dtype == np.float32 and r["z"] == 0.0
        assert 2 <= r["dist"] <= 6                  # stage 0's band, MEASURED
    for i in range(n):                              # paired, and zero-sum
        a, b = res[2 * i], res[2 * i + 1]
        assert a["z"] == -b["z"] and a["turns"] == b["turns"]
    print(f"vecroll selfcheck OK ({n} envs, {features.N_ACTIONS} actions, "
          f"{MASK_BYTES}-byte masks)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if not args.selfcheck:
        ap.error("this module is a library; --selfcheck is all it does alone")
    selfcheck()


if __name__ == "__main__":
    main()
