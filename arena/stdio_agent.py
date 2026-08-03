"""Drive a real `run.sh` over the wire protocol.

The in-process path is what we iterate against; this is what we *submit*. Being
able to point the arena at the packaged artifact means the thing we measure and
the thing we upload are the same thing, including the handshake, the frame
encoding and the process lifecycle.

    python -m arena.runner --a stdio:dist/generals-bot/run.sh --b greedy --games 4
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from bot import rules
from bot.obs import Obs


def encode_obs(obs: Obs) -> str:
    """Byte-identical to `competition/protocol.py::encode_observation`."""
    lines = [f"{obs.turn} {obs.my_land} {obs.my_army} {obs.opp_land} {obs.opp_army}"]
    for grid in (obs.type_grid, obs.owner_grid, obs.army_grid):
        for row in np.asarray(grid, dtype=np.int32):
            lines.append(" ".join(map(str, row.tolist())))
    return "\n".join(lines) + "\n"


class StdioAgent:
    def __init__(self, run_sh: str, player_id: int, h: int, w: int):
        path = Path(run_sh).resolve()
        self.proc = subprocess.Popen(
            ["bash", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            text=True, bufsize=1, cwd=str(path.parent),
        )
        self.proc.stdin.write(f"{player_id} {h} {w}\n")
        self.proc.stdin.flush()

    def act(self, obs: Obs, deadline=None):
        self.proc.stdin.write(encode_obs(obs))
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("agent closed stdout")
        parts = line.split()
        if len(parts) != 5:
            return rules.PASS_ACTION
        return tuple(int(x) for x in parts)

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
