"""Replay storage.

A replay is the initial grid plus both action streams. The simulator is
deterministic, so that is enough to reconstruct every frame exactly — a few kB
per game instead of a few megabytes, which means we can keep every game of every
run and go back to the ones we lost.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim import engine


def save(path: str | Path, result: dict) -> None:
    Path(path).write_text(json.dumps({
        "seed": result["seed"],
        "spec0": result["spec0"],
        "spec1": result["spec1"],
        "winner": result["winner"],
        "reason": result["reason"],
        "turns": result["turns"],
        "grid": result["replay"]["grid"],
        "actions": result["replay"]["actions"],
    }))


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def frames(replay: dict):
    """Yield (turn, State) for every turn including the initial position."""
    grid = np.asarray(replay["grid"], dtype=np.int32)
    st = engine.from_grid(grid)
    yield 0, st.copy()
    for i, (a0, a1) in enumerate(replay["actions"], start=1):
        engine.step(st, a0, a1)
        yield i, st.copy()
