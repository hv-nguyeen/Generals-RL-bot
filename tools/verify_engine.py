"""Differential test: our numpy mirror against the official JAX engine.

Every optimisation in this repo rests on the mirror being exactly right, so it
gets checked rather than trusted. Random games are stepped through both engines
in lockstep with the same actions — including builds, splits, out-of-range
coordinates and malformed action kinds, which is where the two implementations
would realistically diverge — and every field of the state is compared after
every step.

    python -m tools.verify_engine --games 40          # states only
    python -m tools.verify_engine --encoders          # the training seam too

`--encoders` is the second half and the one the learned path rests on. It runs
the same lockstep games on 21x21-PADDED boards and compares, on top of the
state, the four things a JAX rollout would feed a policy: the observation, the
encoder (`bot.features.encode` vs `learn.rlenv.encode_jax`), the legal mask
(`bot.features.legal_mask` vs `learn.rlenv.legal_mask_jax`) and the flat action
map, over every index. `learn.rlenv`'s docstring claimed the encoders were
"verified identical" for months; nothing ran it, and the mask disagreed.

Read the COVERAGE line, not just `ok`. An equality test over two all-False
masks passes vacuously, and the build half of the action space is all-False in
every random game — a tile never reaches the 35 army a castle costs. The
counters are what turn "the masks agree" into "the masks agree about builds".
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "generals-bots"))

from bot import features, rules             # noqa: E402
from sim import engine, mapgen              # noqa: E402


def _jax_bits():
    import jax.numpy as jnp
    from generals.core import game as jgame
    from generals.modifiers import build_castles as jbc
    from generals.modifiers import deathtouch as jdt

    def transition(state, actions):
        state, acts = jbc.apply_build_actions(state, actions)
        return jdt.step(state, acts, rules.DEATHTOUCH_TURN)

    return jnp, jgame, transition


def random_action(rng: np.random.Generator, st: engine.State, p: int) -> list[int]:
    """A deliberately adversarial action distribution."""
    h, w = st.armies.shape
    roll = rng.random()
    if roll < 0.06:
        return [1, 0, 0, 0, 0]
    if roll < 0.24:
        # Build. Aim at a cell that can actually AFFORD one when there is one:
        # an unaffordable build is silently consumed as a pass by both engines,
        # so a uniform target would exercise the price check and never the
        # castle placement it guards.
        rich = np.argwhere(st.own[p] & (st.armies >= rules.BASE_COST))
        owned = rich if len(rich) else np.argwhere(st.own[p])
        r, c = owned[rng.integers(len(owned))] if len(owned) else (0, 0)
        return [2, int(r), int(c), 0, 0]
    if roll < 0.26:                                   # build we do not own / off board
        return [2, int(rng.integers(-2, h + 2)), int(rng.integers(-2, w + 2)), 0, 0]
    if roll < 0.32:                                   # out of range on purpose
        return [0, int(rng.integers(-2, h + 2)), int(rng.integers(-2, w + 2)),
                int(rng.integers(0, 4)), int(rng.integers(0, 2))]
    movable = np.argwhere(st.own[p] & (st.armies >= 2))
    if not len(movable):
        movable = np.argwhere(st.own[p])
    if not len(movable):
        return [1, 0, 0, 0, 0]
    r, c = movable[rng.integers(len(movable))]
    return [0, int(r), int(c), int(rng.integers(0, 4)), int(rng.integers(0, 2))]


def compare(ours: engine.State, theirs, jnp) -> str | None:
    # `theirs` may be a 21x21 padded board while `ours` is the h x w original;
    # the crop is a no-op when the two are the same size.
    h, w = ours.armies.shape
    crop = lambda a: np.asarray(a)[:h, :w]                      # noqa: E731
    checks = {
        "armies": (ours.armies, crop(theirs.armies)),
        "own0": (ours.own[0], crop(theirs.ownership[0])),
        "own1": (ours.own[1], crop(theirs.ownership[1])),
        "neutral": (ours.neutral, crop(theirs.ownership_neutral)),
        "castles": (ours.castles, crop(theirs.castles)),
        "generals": (ours.generals, crop(theirs.generals)),
    }
    for name, (a, b) in checks.items():
        if not np.array_equal(np.asarray(a), b):
            bad = np.argwhere(np.asarray(a) != b)[:4]
            return f"{name} differs at {bad.tolist()}: ours={np.asarray(a)[tuple(bad[0])]} theirs={b[tuple(bad[0])]}"
    if int(ours.time) != int(theirs.time):
        return f"time {ours.time} vs {int(theirs.time)}"
    if int(ours.winner) != int(theirs.winner):
        return f"winner {ours.winner} vs {int(theirs.winner)}"
    return None


def run(games: int, max_turns: int, seed0: int, verbose: bool) -> int:
    jnp, jgame, transition = _jax_bits()
    failures = 0
    for g in range(games):
        seed = seed0 + g
        rng = np.random.default_rng(seed * 977 + 13)
        grid = mapgen.generate(seed)
        ours = engine.from_grid(grid)
        theirs = jgame.create_initial_state(jnp.asarray(grid, dtype=jnp.int32))

        # Start most games late so the deathtouch branch is actually exercised.
        if g % 2:
            ours.time = rules.DEATHTOUCH_TURN - 3
            theirs = theirs._replace(time=jnp.int32(rules.DEATHTOUCH_TURN - 3))

        for turn in range(max_turns):
            a0 = random_action(rng, ours, 0)
            a1 = random_action(rng, ours, 1)
            done = engine.step(ours, a0, a1)
            theirs, info = transition(theirs, jnp.asarray([a0, a1], dtype=jnp.int32))
            diff = compare(ours, theirs, jnp)
            if diff is not None:
                print(f"MISMATCH seed={seed} turn={turn} a0={a0} a1={a1}: {diff}")
                failures += 1
                break
            their_done = bool(info.is_done)
            if done != their_done:
                print(f"MISMATCH seed={seed} turn={turn}: done {done} vs {their_done}")
                failures += 1
                break
            if done:
                break
        if verbose:
            print(f"  seed {seed}: ok ({int(ours.time)} ticks)")
    return failures


# ---------------------------------------------------------------------------
# the training seam: observation, encoder, legal mask, action map
PAD = features.PAD

# Turns worth comparing whatever the stride says: the all-tile growth tick and
# its neighbours, and the deathtouch threshold and its neighbours. A stride that
# happens to skip turn 50 would leave the growth phase untested.
FORCED_TURNS = frozenset({49, 50, 51, 99, 100, 101, 795, 799, 800, 801})
LATE_START = 795            # 5 turns short of deathtouch


def pad_grid(grid: np.ndarray) -> np.ndarray:
    """h x w generator grid -> 21x21, padded with mountains like the engine's."""
    p = np.full((PAD, PAD), -2, dtype=np.int32)
    p[:grid.shape[0], :grid.shape[1]] = grid
    return p


def _seed_edits(st: engine.State, rng: np.random.Generator) -> list[tuple]:
    """(player, r, c, army) edits that make the build half of the space reachable.

    COVERAGE, not realism. Random play never puts 35 army on a tile, so without
    this every one of the 441 build bits is False on both sides of every
    comparison and `array_equal` is asserting that 0 == 0 — the exact shape of a
    test that passes while the thing it was written for is broken.

    Each general gets 400 (enough that captured tiles stay buildable for a
    while) and three tiles are seeded with 60: one at manhattan 1 from the
    general, one at manhattan >= 7, one anywhere. That spread is what makes the
    proximity surcharge observable — 47 next to your own structure, a flat 35
    away from it — so `costs_seen` can fail if the 13x13 surcharge kernel is
    wrong rather than merely absent.
    """
    h, w = st.armies.shape
    free = [(r, c) for r in range(h) for c in range(w)
            if st.passable[r, c] and not st.generals[r, c]]
    edits = []
    for p in (0, 1):
        gr, gc = st.gpos[p]
        edits.append((p, gr, gc, 400))
        near = [x for x in free if abs(x[0] - gr) + abs(x[1] - gc) == 1]
        far = [x for x in free if abs(x[0] - gr) + abs(x[1] - gc) >= 7]
        picked = []
        for pool in (near, far, free):
            choices = [x for x in pool if x not in picked]
            if choices:
                picked.append(choices[int(rng.integers(len(choices)))])
        edits += [(p, r, c, 60) for r, c in picked]
    return edits


def _apply_edits(st: engine.State, theirs, edits, jnp):
    """The same seeding applied to both engines, so they start identical."""
    armies = np.asarray(theirs.armies).copy()
    own = np.asarray(theirs.ownership).copy()
    neutral = np.asarray(theirs.ownership_neutral).copy()
    for p, r, c, a in edits:
        st.own[p, r, c] = True
        st.own[1 - p, r, c] = False
        st.neutral[r, c] = False
        st.armies[r, c] = a
        own[p, r, c] = True
        own[1 - p, r, c] = False
        neutral[r, c] = False
        armies[r, c] = a
    return theirs._replace(armies=jnp.asarray(armies), ownership=jnp.asarray(own),
                           ownership_neutral=jnp.asarray(neutral))


def _padding_clean(theirs, h: int, w: int) -> str | None:
    """Nothing may ever exist outside the real board. If it does, every encoder
    comparison below is comparing a padded 21x21 against an h x w crop and the
    difference lives exactly where neither side looks."""
    for name, arr in (("armies", theirs.armies), ("own0", theirs.ownership[0]),
                      ("own1", theirs.ownership[1]), ("castles", theirs.castles),
                      ("generals", theirs.generals)):
        a = np.asarray(arr)
        if a[h:, :].any() or a[:, w:].any():
            return f"{name} is non-zero in the padding"
    if np.asarray(theirs.passable)[h:, :].any() or np.asarray(theirs.passable)[:, w:].any():
        return "passable is True in the padding"
    return None


def _frame_diff(st: engine.State, theirs, p: int, fns, cov) -> str | None:
    """Compare one player's view through both encoders and both mask builders."""
    h, w = st.armies.shape
    obs_n = engine.observe(st, p)
    obs_j = fns["obs"](theirs, p)

    xn = features.encode(obs_n)
    xj = np.asarray(fns["encode"](obs_j, h, w))
    d = float(np.abs(xn - xj).max())
    cov["encoder_worst"] = max(cov["encoder_worst"], d)
    if d > 1e-6:
        ch = int(np.unravel_index(np.abs(xn - xj).argmax(), xn.shape)[0])
        return f"encoder max|d| = {d:.3e} (channel {ch})"

    # The broadcast scalars read two DIFFERENT sources -- our `engine.observe`
    # totals against the official `owned_army_count` family -- so agreeing is
    # meaningful, but only for a channel that actually moved. A scalar wired to
    # something always 0 agrees with the other side's always 0 forever.
    # This seam deliberately exercises the stateless encoder. Temporal planes
    # are validated in their own trajectory test and are zero here, so only the
    # broadcast base scalars count toward coverage.
    s = xn[features.CLOCK:features.BASE_C, 0, 0]
    cov["scalar_lo"] = np.minimum(cov["scalar_lo"], s)
    cov["scalar_hi"] = np.maximum(cov["scalar_hi"], s)

    mn = features.legal_mask(obs_n)
    mj = np.asarray(fns["mask"](obs_j))
    if not np.array_equal(mn, mj):
        bad = np.flatnonzero(mn != mj)
        return (f"legal mask differs at {len(bad)} of {features.N_ACTIONS} indices, "
                f"first {bad[:6].tolist()} (numpy={mn[bad[:6]].tolist()}, "
                f"jax={mj[bad[:6]].tolist()})")

    cov["frames"] += 1
    if st.time % 50 == 0:
        cov["growth_frames"] += 1
    if st.time >= rules.DEATHTOUCH_TURN:
        cov["deathtouch_frames"] += 1
    off = getattr(features, "BUILD_OFFSET", None)
    if off is not None:
        cells = mn[:PAD * PAD * features.PER_CELL].reshape(PAD, PAD, features.PER_CELL)
        builds = np.argwhere(cells[..., off])
        if len(builds):
            cov["legal_build_frames"] += 1
            structs = ((st.castles | st.generals) & st.own[p])[:h, :w]
            cost = rules.build_cost_grid(structs)
            for r, c in builds:
                cov["build_cells"].add((int(r), int(c)))
                cov["costs"].add(int(cost[r, c]))
    return None


def check_encoders(boards: int = 12, turns: int = 320, stride: int = 8,
                   seed0: int = 0, verbose: bool = False) -> dict:
    """Step both engines in lockstep on padded boards and compare the seam.

    Returns the coverage counters; raises AssertionError on any disagreement or
    on coverage too thin to have proved anything. Half the boards start at turn
    0 (so the 50-tick all-grow fires six times) and half at turn 795 (so
    deathtouch is live for almost the whole run).
    """
    jnp, jgame, transition = _jax_bits()
    from learn import rlenv

    # the flat action map, exhaustively — 3970 indices, pure numpy, ~20 ms
    import jax
    idx = jnp.arange(features.N_ACTIONS, dtype=jnp.int32)
    got = np.asarray(jax.vmap(rlenv.index_to_engine_action)(idx))
    want = np.asarray([features.index_to_action(i) for i in range(features.N_ACTIONS)],
                      dtype=np.int32)
    bad = np.flatnonzero((got != want).any(axis=1))
    assert not len(bad), (f"index_to_engine_action disagrees with index_to_action at "
                          f"{len(bad)} indices, first {bad[:6].tolist()}: "
                          f"jax={got[bad[0]].tolist()} numpy={want[bad[0]].tolist()}")

    # jitted: the mask alone is ~85 shifted adds and eager dispatch dominates
    # the whole run otherwise (26 s vs 6 s, measured).
    fns = {"obs": jax.jit(jgame.get_observation, static_argnums=(1,)),
           "encode": jax.jit(rlenv.encode_jax),
           "mask": jax.jit(rlenv.legal_mask_jax),
           "step": jax.jit(transition)}

    nscalar = features.BASE_C - features.CLOCK
    cov = {"frames": 0, "growth_frames": 0, "deathtouch_frames": 0,
           "legal_build_frames": 0, "builds_executed": 0, "encoder_worst": 0.0,
           "build_cells": set(), "costs": set(), "steps": 0, "shapes": set(),
           "scalar_lo": np.full(nscalar, np.inf, np.float32),
           "scalar_hi": np.full(nscalar, -np.inf, np.float32)}
    t0 = time.time()

    for b in range(boards):
        seed = seed0 + b
        rng = np.random.default_rng(seed * 977 + 13)
        grid = mapgen.generate(seed)
        h, w = grid.shape
        cov["shapes"].add((h, w))
        ours = engine.from_grid(grid)
        theirs = jgame.create_initial_state(jnp.asarray(pad_grid(grid), dtype=jnp.int32))
        theirs = _apply_edits(ours, theirs, _seed_edits(ours, rng), jnp)
        if b >= boards // 2:
            ours.time = LATE_START
            theirs = theirs._replace(time=jnp.int32(LATE_START))

        for t in range(turns):
            a0 = random_action(rng, ours, 0)
            a1 = random_action(rng, ours, 1)
            before = int(ours.castles.sum())
            done = engine.step(ours, a0, a1)
            theirs, info = fns["step"](theirs, jnp.asarray([a0, a1], dtype=jnp.int32))
            cov["steps"] += 1
            built = int(ours.castles.sum()) - before
            cov["builds_executed"] += max(0, built)

            diff = compare(ours, theirs, jnp)
            assert diff is None, f"state: seed={seed} turn={t} a0={a0} a1={a1}: {diff}"
            diff = _padding_clean(theirs, h, w)
            assert diff is None, f"padding: seed={seed} turn={t}: {diff}"
            assert done == bool(info.is_done), f"done: seed={seed} turn={t}"

            if t % stride == 0 or built or int(ours.time) in FORCED_TURNS:
                for p in (0, 1):
                    diff = _frame_diff(ours, theirs, p, fns, cov)
                    assert diff is None, (f"seam: seed={seed} turn={t} (time "
                                          f"{int(ours.time)}) player {p}: {diff}")
            if done:
                break
        if verbose:
            print(f"  board {seed} {h}x{w}: ok to turn {int(ours.time)}")

    wall = time.time() - t0
    shapes = cov["shapes"]
    assert any(a != b for a, b in shapes), f"every board was square: {sorted(shapes)}"
    assert (18, 18) in shapes or min(min(s) for s in shapes) == 18, \
        f"no 18-side board in {sorted(shapes)}"
    assert max(max(s) for s in shapes) == 21, f"no 21-side board in {sorted(shapes)}"
    assert cov["growth_frames"] >= 1, "no compared frame sat on a 50-tick growth turn"
    assert cov["deathtouch_frames"] >= 1, "no compared frame ran past turn 800"

    spread = cov["scalar_hi"] - cov["scalar_lo"]
    print(f"  seam: {cov['steps']} steps, {cov['frames']} compared frames over "
          f"{len(shapes)} board shapes, encoder max|d| {cov['encoder_worst']:.1e}, "
          f"{wall:.1f}s")
    print(f"  COVERAGE scalars: ranges {[f'{lo:.2f}-{hi:.2f}' for lo, hi in zip(cov['scalar_lo'], cov['scalar_hi'])]}")
    flat = np.flatnonzero(spread <= 0)
    assert not len(flat), (
        f"broadcast scalar channels {(features.CLOCK + flat).tolist()} never "
        f"varied over {cov['frames']} frames: their equality is vacuous")
    if hasattr(features, "BUILD_OFFSET"):
        print(f"  COVERAGE builds: {cov['legal_build_frames']} frames with a legal "
              f"build, {cov['builds_executed']} executed, "
              f"{len(cov['build_cells'])} distinct cells, costs {sorted(cov['costs'])}")
        # An equality test over an all-False mask is not a test. These four are
        # what stop the build half of the space from being "verified" vacuously.
        assert cov["legal_build_frames"] >= 200, cov["legal_build_frames"]
        assert cov["builds_executed"] >= 20, cov["builds_executed"]
        assert len(cov["build_cells"]) >= 20, len(cov["build_cells"])
        assert {35, 47} <= cov["costs"], f"surcharge never varied: {sorted(cov['costs'])}"
    else:
        print("  COVERAGE builds: NOT MODELLED in this action space (no BUILD_OFFSET)")
    return cov


def check_memory_encoders(boards: int = 4, turns: int = 240,
                          seed0: int = 20_000) -> dict:
    """Compare stateful encoders with non-vacuous temporal-plane coverage.

    Competition-distance random games often do not make contact in 120 turns,
    so both encoders can zero the entire enemy-history half and still agree.
    Short-distance boards make ownership changes and enemy sightings part of
    the mandatory green gate rather than an accidental property of a seed.
    """
    import jax
    import jax.numpy as jnp

    from bot.memory import TemporalMemory
    from learn import rlenv

    _, jgame, transition = _jax_bits()
    observe = jax.jit(jgame.get_observation, static_argnums=(1,))
    step = jax.jit(transition)
    worst, frames, activity = 0.0, 0, 0.0
    plane_peak = np.zeros(features.MEMORY_C, dtype=np.float32)
    for b in range(boards):
        grid = mapgen.generate(seed0 + b, 2, 6)
        h, w = grid.shape
        ours = engine.from_grid(grid)
        theirs = jgame.create_initial_state(jnp.asarray(pad_grid(grid), dtype=jnp.int32))
        nm = [TemporalMemory(h, w), TemporalMemory(h, w)]
        jm = [{k: v[0] for k, v in rlenv.empty_memory_jax(1).items()} for _ in range(2)]
        rng = np.random.default_rng(seed0 * 13 + b)
        for _ in range(turns + 1):
            for p in (0, 1):
                no, jo = engine.observe(ours, p), observe(theirs, p)
                nm[p].update(no)
                jm[p] = rlenv.update_memory_jax(jm[p], jo)
                xn = features.encode(no, nm[p])
                xj = np.asarray(rlenv.encode_jax(jo, h, w, jm[p]))
                d = float(np.max(np.abs(xn - xj)))
                worst = max(worst, d); frames += 1
                plane_peak = np.maximum(
                    plane_peak,
                    np.max(np.abs(xn[features.BASE_C:]), axis=(1, 2)))
                activity = max(activity, float(np.max(np.abs(
                    xn[features.MY_GAINED:features.DELTA_LAND_ADV + 1]))))
                assert d <= 1e-6, (f"temporal encoder seed={seed0 + b} "
                                   f"turn={ours.time} seat={p} max|d|={d:.3e}")
                before = {k: np.asarray(v).copy() for k, v in jm[p].items()}
                again = rlenv.update_memory_jax(jm[p], jo)
                assert all(np.array_equal(before[k], np.asarray(again[k])) for k in before)
                nm[p].update(no)
            if ours.winner >= 0:
                break
            a0, a1 = random_action(rng, ours, 0), random_action(rng, ours, 1)
            engine.step(ours, a0, a1)
            theirs, _ = step(theirs, jnp.asarray([a0, a1], dtype=jnp.int32))
    assert activity > 0.0, "temporal change planes never activated"
    required = [features.MEM_OPP, features.MEM_ARMY_OPP, features.EVER_ENEMY,
                features.ENEMY_CASTLE,
                features.OPP_GAINED, features.OPP_ARMY_DELTA,
                features.DELTA_OPP_ARMY, features.DELTA_LAND_ADV]
    missing = [ch for ch in required if plane_peak[ch - features.BASE_C] == 0]
    assert not missing, (f"temporal equality is vacuous: required enemy/change "
                         f"planes {missing} never activated")
    active = [features.BASE_C + i for i, value in enumerate(plane_peak) if value > 0]
    print(f"  temporal seam: {frames} frames, max|d| {worst:.1e}, "
          f"activity {activity:.2f}, active planes {active}")
    return {"frames": frames, "worst": worst, "activity": activity,
            "plane_peak": plane_peak}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=140)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--encoders", action="store_true",
                    help="also compare the stateless seam plus stateful temporal "
                         "memory, with non-vacuous coverage for all 16 planes")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    try:
        failures = run(args.games, args.max_turns, args.seed0, args.verbose)
        if args.encoders:
            check_encoders(verbose=args.verbose)
            check_memory_encoders()
    except ImportError as e:
        sys.exit(f"needs jax and third_party/generals-bots: {e}\n"
                 f"  uv pip install --python .venv/bin/python 'jax[cpu]'")

    if failures:
        sys.exit(f"FAILED: {failures}/{args.games} games diverged")
    print(f"ok: {args.games} games identical to the official engine")


if __name__ == "__main__":
    main()
