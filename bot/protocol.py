"""Agent side of the stdin/stdout wire protocol.

Engine -> agent, once:      "<player_id> <H> <W>"
Engine -> agent, per turn:  "<turn> <my_land> <my_army> <opp_land> <opp_army>"
                            then 3 blocks of H lines of W ints: type, owner, army
Agent -> engine, per turn:  "<kind> <row> <col> <dir> <split>"

EOF on stdin means the game is over.
"""

from __future__ import annotations

import numpy as np

from bot.obs import Obs


def read_handshake(stream) -> tuple[int, int, int]:
    line = stream.readline()
    if not line:
        raise EOFError("no handshake")
    player_id, h, w = (int(x) for x in line.split())
    return player_id, h, w


def read_frame(stream, h: int, w: int) -> Obs | None:
    """Read one observation frame. Returns None on EOF (game over)."""
    header = stream.readline()
    if not header:
        return None
    turn, my_land, my_army, opp_land, opp_army = (int(x) for x in header.split())

    rows = []
    for _ in range(3 * h):
        line = stream.readline()
        if not line:
            return None
        rows.append(line)
    flat = np.array("".join(rows).split(), dtype=np.int32)
    if flat.size != 3 * h * w:
        return None
    grids = flat.reshape(3, h, w)

    return Obs(
        H=h, W=w, turn=turn,
        my_land=my_land, my_army=my_army,
        opp_land=opp_land, opp_army=opp_army,
        type_grid=grids[0].astype(np.int8),
        owner_grid=grids[1].astype(np.int8),
        army_grid=grids[2].copy(),
    )


def write_action(stream, action) -> None:
    stream.write("%d %d %d %d %d\n" % tuple(action))
    stream.flush()
