"""Vectorised self-play rollouts on the GPU, behind `learn/selfplay.py`'s
existing produce/train boundary.

    python -m learn.vecroll --selfcheck     # shapes, one vmapped step, scan identity

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
import time
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
# How far `rollout_scan`'s logp may sit from `rollout`'s. NOT a fudge factor:
# a scan body is a different XLA compilation, and the measured gap is 1 ulp
# (4.8e-07 over 8 games x 480 turns, XLA:CPU). 100x that, so it fails on a real
# divergence and not on a fusion change. Everything that decides a game -- the
# observation, the legal mask, the sampled action, the length -- is compared
# EXACTLY. One definition, used by `selfcheck` and by tests/test_all.py.
LOGP_TOL = 1e-5

# Seconds of DEVICE->HOST COPY in the last rollout, compute excluded. Reset at
# the top of `rollout`/`rollout_scan`, printed by `selfplay` as `d2h` inside its
# `roll` phase. A module global because the alternative is a second return value
# on both rollouts and through `play_train`, for a number that exists to be read
# once and then delete itself.
# ponytail: global. It is single-threaded, single-rollout instrumentation.
D2H = 0.0


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

    Returns the legal mask ALREADY PACKED, `(2n, MASK_BYTES)` uint8, because
    3970 bytes a row crossing PCIe to become 497 is 8x the bytes and a
    single-threaded `np.packbits` on the host every turn. `jnp.packbits` and
    `np.packbits` are both big-endian by default; `selfcheck` asserts they agree
    bit for bit rather than trusting that.
    """
    import jax
    import jax.numpy as jnp
    from generals.core import game as jgame

    @jax.jit
    def step(states, h, w, theta, key, t, ended, winner, memory):
        n = h.shape[0]
        obs = jax.tree.map(
            lambda a, b: jnp.concatenate([a, b]),
            jax.vmap(jgame.get_observation, in_axes=(0, None))(states, 0),
            jax.vmap(jgame.get_observation, in_axes=(0, None))(states, 1))
        hh, ww = jnp.concatenate([h, h]), jnp.concatenate([w, w])
        memory = jax.vmap(rlenv.update_memory_jax)(memory, obs)
        x = jax.vmap(rlenv.encode_jax)(obs, hh, ww, memory)
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
        return (states, x.astype(jnp.float16), jnp.packbits(mask, axis=1),
                a_idx, logp,
                jnp.where(first, t, ended), jnp.where(first, info.winner, winner),
                memory)

    return step


def rollout(step, pool, idx: np.ndarray, theta, key, max_turns: int) -> list[dict]:
    """`n = len(idx)` games to completion. Returns exactly what `selfplay._pack`
    returns, in the same order (seat 0 then seat 1 of each game), so everything
    from `selfplay.py`'s ingest down is untouched.

    Host memory: the step buffers are `T x 2n x C x 21 x 21` float16, i.e.
    17.6 kB a sample at `features.C` = 20, and T is the LONGEST game in the
    batch, not the mean -- a 256-game stage-5 iteration that runs near the 1200
    turn limit is `(1200, 512, 20, 21, 21)` f16 = 10.8 GB. `_join` keeps the
    per-turn parts from being live at the same time as the joined buffer; the
    joined buffer IS still live while `_pack_columns` cuts the trajectories out
    of it, and that one is inherent. `--games` and `--max-turns` are the knobs;
    this is the number that OOMs the box, several minutes into an iteration.
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    global D2H
    D2H = 0.0
    n = len(idx)
    states = jax.tree_util.tree_map(lambda a: a[jnp.asarray(idx)], pool["states"])
    h, w = pool["h"][jnp.asarray(idx)], pool["w"][jnp.asarray(idx)]
    ended = jnp.full(n, -1, jnp.int32)
    winner = jnp.full(n, -1, jnp.int32)
    memory = rlenv.empty_memory_jax(2 * n)

    xs, ms, ids, lps = [], [], [], []
    steps = 0
    for t in range(max_turns):
        key, sub = jr.split(key)
        states, x, packed, a_idx, logp, ended, winner, memory = step(
            states, h, w, theta, sub, jnp.int32(t), ended, winner, memory)
        hx, hp, hi, hl = _to_host(x, packed, a_idx, logp)
        # `[None]` so the parts are `(1, 2n, ...)` and `_join` is the same
        # concatenate the scan path uses. It is a view; it costs nothing.
        xs.append(hx[None])
        ms.append(hp[None])
        ids.append(hi[None])
        lps.append(hl[None])
        steps = t + 1
        if t % ALIVE_EVERY == ALIVE_EVERY - 1 and bool((ended >= 0).all()):
            break

    return _pack_columns(_join(xs), _join(ms), _join(ids), _join(lps),
                         ended, winner, steps, pool, idx)


def _to_host(*arrs):
    """`np.asarray` on device arrays, charging the WAIT FOR COMPUTE and the COPY
    to separate clocks. Adds to `D2H`; returns the numpy arrays.

    This exists because the iteration split alone cannot answer the question it
    was built for. `roll` lumps the env transition, two `get_observation`s, the
    encode, the net forward and the ~1.5 GB device->host copy into one number,
    and the whole "the round trip is pure waste" hypothesis is a claim about the
    last term only. Read `roll 7.0/d2h 0.3` and the copy is 4% of the rollout,
    so a device-resident buffer cannot pay for itself; read `roll 7.0/d2h 3.5`
    and it can. There is no third reading.

    Costs nothing that was not already paid: `np.asarray` blocks on the same
    computation `block_until_ready` waits for, so this moves the boundary rather
    than adding one. Do NOT "optimise" the wait away -- without it every second
    of compute lands in the copy column and the split reads as a green light.
    """
    global D2H
    import jax

    jax.block_until_ready(arrs)
    t = time.time()
    out = [np.asarray(a) for a in arrs]
    D2H += time.time() - t
    return out


def _join(parts: list) -> np.ndarray:
    """`np.concatenate` that drops each part as it copies it.

    A memory fix, not a speed one, and the arithmetic is the point:
    `np.concatenate(parts)` holds the list AND the result, so the 10.8 GB
    stage-5 buffer above peaks at 21.7 GB -- on a box that is also hosting the
    worker pool, discovered as a numpy MemoryError minutes into an iteration.
    Peak here is the result plus ONE part.

    Mutates `parts` (entries become None). Callers pass a list they own.
    """
    out = np.empty((sum(len(p) for p in parts),) + parts[0].shape[1:],
                   parts[0].dtype)
    i = 0
    for k, p in enumerate(parts):
        out[i:i + len(p)] = p
        i += len(p)
        parts[k] = None
    return out


def _turns(ended, steps: int) -> np.ndarray:
    """Length of every column's game, per GAME (length n), one derivation.

    Counts the tick the game ended ON, matching the CPU rollout: it records the
    observation, acts, steps, and stops. `steps` is the fallback for a column
    that never ended, so it MUST be the absolute number of turns stepped -- see
    `make_scan` on why that is the whole risk of the scan.

    One function because `_pack_columns` slices with it and `row_index` gathers
    with it, and a second derivation is how those two silently stop agreeing.
    """
    ended = np.asarray(ended)
    return np.where(ended >= 0, ended + 1, steps).astype(int)


def _pack_columns(X, M, I, L, ended, winner, steps: int, pool, idx) -> list[dict]:
    """Turn-major device buffers -> `selfplay._pack`'s per-trajectory dicts.

    Shared by `rollout` and `rollout_scan` so the two paths differ ONLY in where
    the per-turn loop lives. A difference between them is then unambiguously a
    rollout difference and not a slicing one, which is what
    `tests/test_all.py::test_the_scanned_rollout_matches_the_python_loop`
    compares.

    Every buffer is `(T, 2n, ...)`: turn first, then the seat-stacked column.
    """
    n = len(idx)
    ended = np.asarray(ended)
    winner = np.asarray(winner)
    turns = _turns(ended, steps)
    dist = pool["dist"][idx]

    out = []
    for i in range(n):
        T = int(turns[i])
        if T == 0:                      # cannot happen; _pack drops such games
            continue
        for s in (0, 1):                # both seats or neither: the buffer must
            col = s * n + i             # stay exactly zero-sum
            # `x` is a VIEW, not a copy, and that is the whole point: the caller
            # (`selfplay`'s `xs[k:k+m] = r.pop("x")`) immediately gathers it into
            # one contiguous buffer, so copying here writes 1.2 GB a stage-0
            # iteration only to read it again and free it. One pass over the
            # largest array in the process, removed.
            # Peak host RAM is UNCHANGED, not lowered -- the view keeps `X` alive
            # until the last column is popped, where the copies used to keep
            # themselves alive: |X| + util|X| either way. Do not sell this as a
            # memory fix; `_join` is the memory fix. The other three are 24x
            # smaller and are copied so `M`/`I`/`L` can be freed here.
            # The view CANNOT outlive `X`: `selfplay` pops it straight into `xs`
            # and never holds it past that line.
            out.append({"z": outcome(int(winner[i]), s), "turns": T,
                        "dist": int(dist[i]), "seat": s, "opp": 0,
                        "x": X[:T, col], "idx": I[:T, col].copy(),
                        "mask": M[:T, col].copy(), "logp": L[:T, col].copy()})
    return out


def row_index(ended, steps: int, n: int) -> np.ndarray:
    """The flat `t * 2n + col` row of every sample `_pack_columns` emits, in the
    order it emits them.

    Takes the rollout's OWN `(ended, steps)` -- not a length vector -- so that
    the caller cannot pick the wrong one. A `turns` argument has two plausible
    readings, per-game (length n) and per-emitted-column (n minus the dropped
    T == 0 games), they differ only when a game is dropped, and BOTH give a
    gather of the right total length. Every game after the dropped one would
    then read a different column: another game's observations paired with this
    game's outcome. `_turns` is the single derivation both sides use.

    Nothing on the hot path calls this. It is here because it is the ONE piece a
    device-resident buffer needs and the one piece that can be wrong in silence:
    today the ragged cut is a SLICE per column, and a buffer that never comes to
    the host has to GATHER instead. Build the gather from the padded `T` instead
    of from `turns` and dead columns enter the training buffer with zeroed
    observations and a real outcome attached. The critic fits them, advantages
    stay finite, `evar` stays plausible, `dnp` checks the forward and not the
    segmentation, and the scan-vs-loop test compares rollout outputs and not GAE.
    Nothing complains, ever, and the run just learns slightly the wrong thing.

    tests/test_all.py::test_the_row_index_reproduces_the_packed_columns asserts
    the gather equals the slices, on ragged lengths, with no jax and no games.

    # ponytail: the index vector only. The `--backend resident` that consumes it
    # is gated on the --epochs 0 / roll-pack-ing-trn split; build it if and only
    # if that says the iteration is transfer bound.
    """
    rows = [np.arange(T, dtype=np.int64) * (2 * n) + s * n + i
            for i, T in enumerate(int(t) for t in _turns(ended, steps))
            if T for s in (0, 1)]
    return np.concatenate(rows) if rows else np.zeros(0, np.int64)


def make_scan(step, chunk: int):
    """`step` for `chunk` turns without returning to python.

    The ONLY thing that changes from `rollout`'s loop is where the loop lives:
    the body calls the same jitted `step`, splits the same key in the same
    order, and passes the same ABSOLUTE turn index -- see the `t + 1` below,
    which is the whole risk in this file.

    NOT bit-identical, and that was measured rather than assumed. A scan body is
    a different XLA compilation from a standalone `@jax.jit`, so the conv trunk
    fuses differently and the LOGITS move by ~3.6e-07 (measured, XLA:CPU, jax
    0.11). Observations, legal masks, sampled actions and lengths came back
    identical over 8 games x 480 turns; `logp` differs by up to 1 ulp (4.8e-07).
    That is harmless -- `logp` is the PPO ratio denominator, and 5e-07 of
    relative error in it is far below any gradient -- but the drift is in the
    input to `jax.random.categorical`, so a sampled action CAN flip at a
    near-tie. `tests/test_all.py` holds the line: exact on everything that
    decides a game, bounded on `logp`.

    Read the output as `(chunk, 2n, ...)` stacked in TURN-MAJOR order, i.e. what
    `rollout`'s per-turn `xs.append` produced, already stacked on the device.
    """
    import jax
    import jax.random as jr

    @jax.jit
    def run(states, h, w, theta, key, t0, ended, winner, memory):
        def body(carry, _):
            states, key, t, ended, winner, memory = carry
            key, sub = jr.split(key)
            states, x, packed, a_idx, logp, ended, winner, memory = step(
                states, h, w, theta, sub, t, ended, winner, memory)
            # `t` is ABSOLUTE, which is why it is carried and not a scan index.
            # `step` latches `ended = where(first, t, ended)`, so restarting the
            # counter at 0 each chunk would silently shorten every game that
            # ends after chunk 0 -- shapes stay consistent, z stays right, and
            # the tail of every decided game just vanishes from the buffer.
            return (states, key, t + 1, ended, winner, memory), (x, packed, a_idx, logp)
        return jax.lax.scan(body, (states, key, t0, ended, winner, memory), None,
                            length=chunk)
    return run


def rollout_scan(run, pool, idx: np.ndarray, theta, key, max_turns: int,
                 chunk: int) -> list[dict]:
    """`rollout` with the per-turn loop replaced by a `chunk`-turn `lax.scan`.

    Two counts, because they are different numbers and the earlier version of
    this docstring divided one by the other. At stage 0 (~160 turns, chunk 40):
    kernel launches 160 -> 4 (40x), and BLOCKING host calls -- four
    `np.asarray` a turn plus the liveness probe -- 645 -> 20 (32x). The bytes
    moved are unchanged; the round trips are the diagnosis.

    What this does NOT remove: the D2H copy of each chunk does not overlap
    compute, so the device idles through 1.5 GB a stage-0 iteration, and every
    sample still round-trips host -> `_pack_columns` -> `selfplay`'s gather ->
    back to the device. That is ~1.8 s of host memcpy an iteration and it is
    why the derived ~40k env-steps/s is an upper bound this path cannot reach.

    NO autoreset, for the same reason `rollout` has none: one column is one
    whole game, so there is no episode boundary for an advantage to leak
    across. That is also why the scan is CHUNKED rather than one scan of
    `max_turns` -- you cannot break out of a `lax.scan`, and a full-length scan
    would step 1200 turns where this stops at ~160 at stage 0, i.e. it would
    report a beautiful env-steps/s while being 7.5x slower per GAME.

    Device memory for the chunk buffer: `chunk * 2 * games * 18.2 kB` (17.6 of
    `x`, 497 B of packed mask, 8 B of idx/logp) -- 373 MB at chunk 40 /
    256 games, 3.0 GB at 2048, and DOUBLE that at the peak because the previous
    chunk's output is still bound while XLA allocates the next one. Even so the
    L4 is not the binding resource: chunk 40 fits ~5000 games. HOST ram is,
    see `_join`. Lower `--scan-chunk` before lowering `--games`.
    """
    import jax
    import jax.numpy as jnp

    if chunk < 1 or max_turns % chunk:
        raise ValueError(
            f"--max-turns {max_turns} must be a positive multiple of "
            f"--scan-chunk {chunk}: a partial last chunk would step past the "
            "turn limit, and the turn limit IS the draw rule, so a game decided "
            "at turn 1205 would be recorded as a win.")

    global D2H
    D2H = 0.0
    n = len(idx)
    states = jax.tree_util.tree_map(lambda a: a[jnp.asarray(idx)], pool["states"])
    h, w = pool["h"][jnp.asarray(idx)], pool["w"][jnp.asarray(idx)]
    ended = jnp.full(n, -1, jnp.int32)
    winner = jnp.full(n, -1, jnp.int32)
    memory = rlenv.empty_memory_jax(2 * n)

    xs, ms, ids, lps = [], [], [], []
    steps = 0
    for t0 in range(0, max_turns, chunk):
        (states, key, _, ended, winner, memory), (x, m, i_, l_) = run(
            states, h, w, theta, key, jnp.int32(t0), ended, winner, memory)
        hx, hm, hi, hl = _to_host(x, m, i_, l_)
        xs.append(hx)
        ms.append(hm)
        ids.append(hi)
        lps.append(hl)
        steps = t0 + chunk
        if bool((ended >= 0).all()):
            break

    return _pack_columns(_join(xs), _join(ms), _join(ids), _join(lps),
                         ended, winner, steps, pool, idx)


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
               jnp.full(n, -1, jnp.int32), jnp.full(n, -1, jnp.int32),
               rlenv.empty_memory_jax(2 * n))
    states2, x, packed, a_idx, logp, ended, winner, memory = out
    assert x.shape == (2 * n, features.C, features.PAD, features.PAD), x.shape
    assert packed.shape == (2 * n, MASK_BYTES), packed.shape
    assert packed.dtype == np.uint8, packed.dtype
    assert a_idx.shape == (2 * n,) and logp.shape == (2 * n,)
    assert np.all(np.asarray(logp) <= 0.0)
    # every sampled action is legal -- the mask is applied BEFORE the sample
    m = np.unpackbits(np.asarray(packed), axis=1)[:, :features.N_ACTIONS].astype(bool)
    assert m[np.arange(2 * n), np.asarray(a_idx)].all()
    # `make_step` packs on the DEVICE and `selfplay.to_mask` unpacks on it, so
    # the host packbits that used to sit in the loop is gone and nothing else
    # tests its replacement. Bit order is the only thing that can differ, and an
    # asymmetric probe is what shows it: a round-trip through np.unpackbits
    # would agree with itself under either convention.
    probe = (np.arange(2 * n * features.N_ACTIONS)
             .reshape(2 * n, features.N_ACTIONS) % 3 == 0)
    assert np.array_equal(np.asarray(jnp.packbits(jnp.asarray(probe), axis=1)),
                          np.packbits(probe, axis=1)), "jnp/np packbits bit order"
    # `selfplay.to_x` / `to_mask` moved the f16->f32 widen and the unpack onto
    # the device. NO dtype changed -- f16 is a strict subset of f32, so the two
    # orderings are bit-identical and the dnp bound (1e-4 x scale, and the seam
    # test's measured 1.2e-07) is untouched. Asserted, not argued:
    xh = np.asarray(x)
    assert np.array_equal(np.asarray(jnp.asarray(xh).astype(jnp.float32)).view(np.uint32),
                          np.asarray(jnp.asarray(xh.astype(np.float32))).view(np.uint32)), \
        "device-side f16->f32 widen is not bitwise the host cast"
    assert np.array_equal(
        np.asarray(jnp.unpackbits(jnp.asarray(packed), axis=1)
                   [:, :features.N_ACTIONS] != 0), m), "device unpack != host unpack"
    assert int(np.asarray(states2.time)[0]) == 1
    assert np.all(np.asarray(ended) == -1) and np.all(np.asarray(winner) == -1)
    assert np.asarray(memory["initialised"]).all()

    # seat rows: row i and row n + i are the SAME board seen by the two players,
    # so they must differ (own fog) but share the VALID channel
    xn = np.asarray(x)
    for i in range(n):
        assert not np.array_equal(xn[i], xn[n + i]), i
        assert np.array_equal(xn[i, features.VALID], xn[n + i, features.VALID]), i

    # `_join` is on BOTH rollout paths, so the loop-vs-scan test cannot see a
    # bug in it -- it would corrupt the two sides identically. Checked here
    # against the concatenate it replaces, on uneven parts.
    parts = [np.arange(k * 6, dtype=np.int16).reshape(k, 3, 2) + 100 * j
             for j, k in enumerate((1, 4, 2, 1))]
    assert np.array_equal(_join(list(parts)), np.concatenate(parts)), "_join"

    # and the whole rollout contract, on a 3-turn stub
    global D2H
    D2H = 1e6                   # a poisoned value: `rollout` must RESET it, and
    res = rollout(step, pool, np.arange(n), theta, key, max_turns=3)
    # if it ever stops, `d2h` accumulates across iterations, grows without bound
    # and reads as "the iteration is transfer bound" while nothing else changes.
    assert 0.0 < D2H < 1e6, f"D2H {D2H} not reset-and-accumulated by rollout"
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

    # The scan, on the same 3-turn stub: plumbing only. This is DELIBERATELY the
    # weak version -- no game ends in 3 turns, so the `turns = ended + 1` branch
    # is never taken and the absolute turn counter is untested here. The real
    # check is tests/test_all.py::test_the_scanned_rollout_is_bit_identical,
    # which runs long enough for games to end and asserts that it happened.
    for ch in (3, 1):
        D2H = 1e6
        got = rollout_scan(make_scan(step, ch), pool, np.arange(n), theta, key,
                           max_turns=3, chunk=ch)
        assert 0.0 < D2H < 1e6, f"D2H {D2H} not reset by rollout_scan"
        assert len(got) == len(res), (ch, len(got))
        for a, b in zip(res, got):
            for k in ("z", "turns", "dist", "seat", "opp"):
                assert a[k] == b[k], (ch, k, a[k], b[k])
            for k in ("x", "idx", "mask"):     # exact: these decide the game
                assert np.array_equal(a[k], b[k]), (ch, k)
            # logp is bounded, not exact -- see `make_scan` on why a scan body
            # is a different compilation. Same bound as tests/test_all.py.
            assert np.abs(a["logp"] - b["logp"]).max() <= LOGP_TOL, ch
    try:
        rollout_scan(None, pool, np.arange(n), theta, key, max_turns=3, chunk=2)
        raise AssertionError("a partial last chunk was accepted")
    except ValueError:
        pass
    print(f"vecroll selfcheck OK ({n} envs, {features.N_ACTIONS} actions, "
          f"{MASK_BYTES}-byte masks, scan chunks 3/1 match)")


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
