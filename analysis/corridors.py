"""Do attacks die in the same place twice, and do kills arrive down one lane?

Two claims worth testing before writing any code that acts on them:

  * our thrust takes a plain shortest path, which is deterministic, so repeated
    attacks reuse one corridor and can be intercepted on it again and again;
  * once an opponent knows where our general is, their attacks come down a
    predictable approach, which would make that corridor worth holding.

Both are checkable from replays. For each game this measures where we lost large
stacks, how often those losses repeat on cells we already lost a stack on, and
which cells enemy army traversed on its way to our general.

If the repeat rate is near chance, the idea is wrong and costs nothing. If it is
high, path variation is a mechanism-level fix rather than a tuning direction —
and those are the only local-evidence changes that have survived the ladder.

    python -m analysis.corridors /local/data/vng205/field --player H.V.Nguyen
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from analysis.official import read_replay, replay_files, summarise
from bot.board import bfs_field_from

BIG_LOSS = 12          # army lost in one tick that counts as "a stack died"


def stack_deaths(rep: dict, seat: int) -> list[tuple[int, int, int, int]]:
    """(tick, row, col, army_lost) where we lost a real stack to the opponent."""
    ticks = rep["ticks"]
    out = []
    prev_a = np.asarray(ticks[0]["armies"], dtype=np.int32)
    prev_o = np.asarray(ticks[0]["owners"], dtype=np.int32)
    for t in range(1, len(ticks)):
        a = np.asarray(ticks[t]["armies"], dtype=np.int32)
        o = np.asarray(ticks[t]["owners"], dtype=np.int32)
        # a cell we held with real army, now theirs
        lost = (prev_o == seat) & (o == 1 - seat) & (prev_a >= BIG_LOSS)
        for r, c in np.argwhere(lost):
            out.append((t, int(r), int(c), int(prev_a[r, c])))
        prev_a, prev_o = a, o
    return out


def approach_cells(rep: dict, seat: int, radius: int = 6) -> collections.Counter:
    """How often enemy army occupied each cell near our general."""
    h, w = int(rep["dims"]["rows"]), int(rep["dims"]["cols"])
    mountains = np.zeros((h, w), dtype=bool)
    for r, c in rep["mountains"]:
        mountains[r, c] = True
    gr, gc = rep["generals"][seat]
    dist = bfs_field_from(~mountains, (int(gr), int(gc)))
    near = dist <= radius

    seen = collections.Counter()
    for tick in rep["ticks"]:
        o = np.asarray(tick["owners"], dtype=np.int32)
        a = np.asarray(tick["armies"], dtype=np.int32)
        for r, c in np.argwhere((o == 1 - seat) & near & (a >= 5)):
            seen[(int(r), int(c))] += 1
    return seen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", default="/local/data/vng205/field")
    ap.add_argument("--player", required=True)
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    src = Path(args.src)
    meta = {}
    for m in src.glob("matches*.json"):
        for row in json.loads(m.read_text()):
            meta[str(row["id"])] = row

    games = repeats = deaths = 0
    per_game_repeat = []
    concentration = []
    for f in replay_files(src):
        if games >= args.limit:
            break
        rep = read_replay(f)
        if args.player not in rep["players"]:
            continue
        seat = rep["players"].index(args.player)
        games += 1

        d = stack_deaths(rep, seat)
        deaths += len(d)
        cells = collections.Counter((r, c) for _, r, c, _ in d)
        rep_here = sum(n - 1 for n in cells.values() if n > 1)
        repeats += rep_here
        if len(d) >= 2:
            per_game_repeat.append(rep_here / (len(d) - 1))

        near = approach_cells(rep, seat)
        if near:
            tot = sum(near.values())
            top = sum(n for _, n in near.most_common(3))
            concentration.append(top / tot)

    if not games:
        raise SystemExit(f"no replays featuring {args.player}")

    print(f"{games} games, {deaths} stacks of {BIG_LOSS}+ army lost\n")
    print(f"  stack losses per game            {deaths / games:.1f}")
    if per_game_repeat:
        print(f"  repeat rate on the same cell     {100 * np.mean(per_game_repeat):.1f}%")
        print("    (share of stack losses that happened on a cell where we had")
        print("     already lost a stack in the same game)")
    if concentration:
        print(f"\n  enemy approach concentration     {100 * np.mean(concentration):.1f}%")
        print("    (share of enemy presence near our general that sat on just")
        print("     three cells — high means one corridor, low means many)")

    print("\nHow to read this:")
    print("  repeat rate above ~25% means we really do walk into the same")
    print("  interception twice, and varying the thrust path is a real fix.")
    print("  Approach concentration above ~50% means their attacks funnel through")
    print("  a chokepoint that would be worth holding.")
    print("  Both near chance means the idea does not apply and costs nothing.")


if __name__ == "__main__":
    main()
