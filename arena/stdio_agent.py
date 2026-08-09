"""Drive a real `run.sh` over the wire protocol.

The in-process path is what we iterate against; this is what we *submit*. Being
able to point the arena at the packaged artifact means the thing we measure and
the thing we upload are the same thing, including the handshake, the frame
encoding and the process lifecycle.

    python -m arena.runner --a stdio:dist/generals-bot/run.sh --b greedy --games 4
"""

from __future__ import annotations

import select
import subprocess
import sys
import time
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
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=str(path.parent),
        )
        self.stderr_lines: list[str] = []
        self.proc.stdin.write(f"{player_id} {h} {w}\n")
        self.proc.stdin.flush()

    def act(self, obs: Obs, deadline=None):
        self.proc.stdin.write(encode_obs(obs))
        self.proc.stdin.flush()
        deadline = deadline if deadline is not None else time.perf_counter() + 10.0
        line = None
        fatal = None
        # readline() ignored the arena deadline and could hang a worker forever.
        # Wait on stdout while draining stderr so a traceback/fallback cannot be
        # hidden behind valid PASS actions or fill the child's pipe.
        while line is None:
            remaining = max(0.0, deadline - time.perf_counter())
            if remaining == 0.0:
                self.proc.kill()
                raise TimeoutError("stdio agent missed its move deadline")
            ready, _, _ = select.select(
                [self.proc.stdout, self.proc.stderr], [], [], remaining)
            if not ready:
                self.proc.kill()
                raise TimeoutError("stdio agent missed its move deadline")
            if self.proc.stderr in ready:
                err = self.proc.stderr.readline()
                if err:
                    self.stderr_lines.append(err.rstrip())
                    print(err, end="", file=sys.stderr)
                    if "Traceback" in err or "FALLING BACK" in err:
                        fatal = err.rstrip()
            if self.proc.stdout in ready:
                line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("agent closed stdout")
        if fatal:
            raise RuntimeError(f"packaged agent reported: {fatal}")
        parts = line.split()
        if len(parts) != 5:
            raise RuntimeError(f"agent returned {len(parts)} fields, expected 5")
        try:
            return tuple(int(x) for x in parts)
        except ValueError as e:
            raise RuntimeError(f"agent returned non-integer action: {line.rstrip()}") from e

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
        if self.proc.stderr is not None:
            rest = self.proc.stderr.read()
            if rest:
                self.stderr_lines.extend(rest.splitlines())
                print(rest, end="", file=sys.stderr)
