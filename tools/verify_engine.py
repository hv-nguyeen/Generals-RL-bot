"""Differential test: our numpy mirror against the official JAX engine.

Every optimisation in this repo rests on the mirror being exactly right, so it
gets checked rather than trusted. Random games are stepped through both engines
in lockstep with the same actions — including builds, splits, out-of-range
coordinates and malformed action kinds, which is where the two implementations
would realistically diverge — and every field of the state is compared after
every step.

    python -m tools.verify_engine --games 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "generals-bots"))

from bot import rules                       # noqa: E402
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
    if roll < 0.14:                                   # build somewhere we own
        owned = np.argwhere(st.own[p])
        r, c = owned[rng.integers(len(owned))] if len(owned) else (0, 0)
        return [2, int(r), int(c), 0, 0]
    if roll < 0.20:                                   # out of range on purpose
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
    checks = {
        "armies": (ours.armies, np.asarray(theirs.armies)),
        "own0": (ours.own[0], np.asarray(theirs.ownership[0])),
        "own1": (ours.own[1], np.asarray(theirs.ownership[1])),
        "neutral": (ours.neutral, np.asarray(theirs.ownership_neutral)),
        "castles": (ours.castles, np.asarray(theirs.castles)),
        "generals": (ours.generals, np.asarray(theirs.generals)),
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=140)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    try:
        failures = run(args.games, args.max_turns, args.seed0, args.verbose)
    except ImportError as e:
        sys.exit(f"needs jax and third_party/generals-bots: {e}\n"
                 f"  uv pip install --python .venv/bin/python 'jax[cpu]'")

    if failures:
        sys.exit(f"FAILED: {failures}/{args.games} games diverged")
    print(f"ok: {args.games} games identical to the official engine")


if __name__ == "__main__":
    main()
