"""Recover the actions players took, from god-view replays.

Replays store state per tick, not moves. Everything on the learning track —
behavioural cloning, a win-probability model, an RL warm start — needs
(observation, action) pairs, so the moves have to be reconstructed.

The trick is to use the exact simulator rather than guess from army deltas.
Step the tick with both players passing; whatever differs from the real next
state must have been touched by the two moves. That narrows the candidates to a
handful of cells, and then the pair is confirmed by replaying it: an inferred
action pair is only accepted if stepping it reproduces the observed next state
*exactly*. Anything unconfirmed is dropped rather than guessed, so the training
set contains no invented labels.

Observations are rebuilt through the same fog rule the engine uses, so a cloned
policy learns from what the player could actually see, not from god view.

    python -m analysis.actions runs/field --player ResBot --limit 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis.official import (castle_builds, castle_builds_by_cost,
                               read_replay, replay_files)
from bot import rules
from bot.board import DIRS, dilate8
from bot.obs import Obs
from sim import engine

PASS = [rules.PASS, 0, 0, 0, 0]


def _states_from(rep: dict, builds=None):
    """Materialise every tick as a simulator State (castles inferred)."""
    h, w = int(rep["dims"]["rows"]), int(rep["dims"]["cols"])
    mountains = np.zeros((h, w), dtype=bool)
    for r, c in rep["mountains"]:
        mountains[r, c] = True
    generals = np.zeros((h, w), dtype=bool)
    gpos = []
    for r, c in rep["generals"]:
        generals[r, c] = True
        gpos.append((int(r), int(c)))

    castles = np.zeros((h, w), dtype=bool)
    for r, c in rep.get("castles") or []:
        castles[r, c] = True
    if builds is None:
        builds = castle_builds(rep)

    out = []
    for t, tick in enumerate(rep["ticks"]):
        for bt, br, bc in builds:
            if bt == t:
                castles[br, bc] = True
        owners = np.asarray(tick["owners"], dtype=np.int32)
        st = engine.State(
            armies=np.asarray(tick["armies"], dtype=np.int32),
            own=np.stack([owners == 0, owners == 1]),
            neutral=(owners < 0) & ~mountains,
            generals=generals,
            castles=castles.copy(),
            mountains=mountains,
            passable=~mountains,
            gpos=list(gpos),
        )
        st.time = t
        out.append(st)
    return out, builds


def _same(a: engine.State, b: engine.State) -> bool:
    return (np.array_equal(a.armies, b.armies)
            and np.array_equal(a.own, b.own))


def _candidates(st: engine.State, p: int, touched: np.ndarray, builds_here) -> list[list[int]]:
    """Moves this player could plausibly have made, given which cells changed."""
    h, w = st.armies.shape
    out = [PASS]
    for (br, bc) in builds_here:
        if st.own[p, br, bc]:
            out.append([rules.BUILD, br, bc, 0, 0])
    srcs = np.argwhere(touched & st.own[p] & (st.armies >= 2))
    for r, c in srcs:
        r, c = int(r), int(c)
        for d, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not st.passable[nr, nc]:
                continue
            if not touched[nr, nc]:
                continue
            out.append([rules.MOVE, r, c, d, 0])
            out.append([rules.MOVE, r, c, d, 1])
    return out


def infer(rep: dict) -> tuple[list, int]:
    """Per-tick (action_p0, action_p1). Unrecoverable ticks come back as None.

    The castle set is not directly observable, and a wrong one poisons growth for
    the rest of the game, so both detectors are tried and the hypothesis that
    reconstructs more ticks wins.
    """
    best = None
    for hypothesis in (castle_builds(rep), castle_builds_by_cost(rep)):
        acts, solved = _infer_with(rep, hypothesis)
        if best is None or solved > best[1]:
            best = (acts, solved)
    return best


def _infer_with(rep: dict, builds) -> tuple[list, int]:
    states, _ = _states_from(rep, builds)
    by_tick: dict[int, list[tuple[int, int]]] = {}
    for t, r, c in builds:
        by_tick.setdefault(t, []).append((r, c))

    actions: list[tuple[list[int], list[int]] | None] = []
    solved = 0
    for t in range(len(states) - 1):
        cur, nxt = states[t], states[t + 1]
        probe = cur.copy()
        engine.step(probe, PASS, PASS)
        touched = (probe.armies != nxt.armies) | (probe.own[0] != nxt.own[0]) | (probe.own[1] != nxt.own[1])
        if not touched.any():
            actions.append((PASS, PASS))
            solved += 1
            continue
        touched = dilate8(touched)
        here = by_tick.get(t + 1, [])
        c0 = _candidates(cur, 0, touched, here)
        c1 = _candidates(cur, 1, touched, here)

        found = None
        for a0 in c0:
            for a1 in c1:
                trial = cur.copy()
                engine.step(trial, a0, a1)
                if _same(trial, nxt):
                    found = (a0, a1)
                    break
            if found:
                break
        actions.append(found)
        solved += found is not None
    return actions, solved


def observations(rep: dict, seat: int):
    """The fogged frames that player actually saw, one per tick."""
    states, _ = _states_from(rep)
    for st in states:
        yield engine.observe(st, seat)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default="runs/field")
    ap.add_argument("--player", default=None, help="only replays featuring this player")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    files = replay_files(Path(args.dir))
    total = ok = games = 0
    for f in files:
        if games >= args.limit:
            break
        rep = read_replay(f)
        if args.player and args.player not in rep["players"]:
            continue
        acts, solved = infer(rep)
        total += len(acts)
        ok += solved
        games += 1
        print(f"  {f.name:<20} {solved}/{len(acts)} ticks recovered "
              f"({100.0 * solved / max(len(acts), 1):.1f}%)")
    if total:
        print(f"\n{games} games, {ok}/{total} ticks recovered "
              f"({100.0 * ok / total:.2f}%)")
        print("Unrecovered ticks are dropped, not guessed — a cloned policy never "
              "trains on an invented label.")


if __name__ == "__main__":
    main()
