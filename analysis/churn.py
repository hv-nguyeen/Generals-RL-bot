"""Do we capture tiles we cannot hold?

A field analysis of 36 ladder games reported our own territory changing hands
~78 times in wins and ~288 times in losses, with captures equal or higher in the
losses. Churn with high capture counts has two very different causes and they
call for opposite fixes:

  * we lose big fights on a contested front, and the front sloshes back and
    forth — the answer is more army, i.e. an economy problem;
  * we take tiles with nothing left standing on them and they flip straight
    back — the answer is to not take them, i.e. a one-line scoring problem.

Those are separable from the replays. Every flip has an age (how long we held
it) and a birth army (what we left on it when we took it). If the flips are
young and their birth army is 1-2, we are spending moves to rent tiles. If they
are old and well-garrisoned, it is a real fight and this is the wrong tool.

    python -m analysis.churn /local/data/vng205/field --player H.V.Nguyen

Read the split, not the totals. `flips/game` differing between wins and losses
says almost nothing on its own — losses are longer. The number that decides it
is the birth army of flipped tiles against the birth army of tiles that held.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.official import read_replay, replay_files, summarise

YOUNG = 12          # ticks; a tile lost this fast was never really held


def flips(rep: dict, seat: int):
    """Every own-tile loss, with how long we held it and what we left on it.

    Returns (ages, birth_armies, held_birth_armies, captures, total_lost). The
    third is the control group: tiles we took and still held at the end. Without
    it a low birth army means nothing, because our birth army may simply always
    be low.

    Both sides count only tiles we actually CAPTURED (born after tick 0). Spawn
    tiles have a birth army set by the map, not by us, and the general's would
    drag the control group up on its own.
    """
    ticks = rep["ticks"]
    o = np.asarray(ticks[0]["owners"], dtype=np.int32)
    born_t = np.where(o == seat, 0, -1)
    born_a = np.where(o == seat, np.asarray(ticks[0]["armies"], dtype=np.int32), 0)

    ages, birth, captures, total_lost = [], [], 0, 0
    prev_o = o
    for t in range(1, len(ticks)):
        o = np.asarray(ticks[t]["owners"], dtype=np.int32)
        a = np.asarray(ticks[t]["armies"], dtype=np.int32)

        gained = (prev_o != seat) & (o == seat)
        captures += int(gained.sum())
        born_t = np.where(gained, t, born_t)
        born_a = np.where(gained, a, born_a)

        lost = (prev_o == seat) & (o != seat)
        total_lost += int(lost.sum())
        for r, c in np.argwhere(lost & (born_t > 0)):
            ages.append(t - int(born_t[r, c]))
            birth.append(int(born_a[r, c]))
        prev_o = o

    still = (prev_o == seat) & (born_t > 0)
    return ages, birth, [int(x) for x in born_a[still]], captures, total_lost


def fronts(rep: dict, seat: int) -> float:
    """Mean number of separate contested borders we are holding at once.

    'Avoid multiple weak fronts' is only actionable if we are in fact fighting on
    several at once. Counted as connected components of our tiles that touch an
    enemy tile, sampled every 25 ticks.
    """
    ticks = rep["ticks"]
    counts = []
    for t in range(0, len(ticks), 25):
        o = np.asarray(ticks[t]["owners"], dtype=np.int32)
        mine, theirs = o == seat, (o != seat) & (o >= 0)
        adj = np.zeros_like(theirs)
        adj[1:, :] |= theirs[:-1, :]
        adj[:-1, :] |= theirs[1:, :]
        adj[:, 1:] |= theirs[:, :-1]
        adj[:, :-1] |= theirs[:, 1:]
        border = mine & adj
        if not border.any():
            continue
        # flood fill, 4-connected
        seen = np.zeros_like(border)
        n = 0
        for start in np.argwhere(border):
            if seen[start[0], start[1]]:
                continue
            n += 1
            stack = [tuple(start)]
            seen[start[0], start[1]] = True
            while stack:
                r, c = stack.pop()
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (0 <= nr < border.shape[0] and 0 <= nc < border.shape[1]
                            and border[nr, nc] and not seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
        counts.append(n)
    return float(np.mean(counts)) if counts else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", default="/local/data/vng205/field")
    ap.add_argument("--player", required=True)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    src = Path(args.src)
    meta = {}
    for m in list(src.glob("matches.json")) + list(src.glob("matches_*.json")):
        for row in json.loads(m.read_text()):
            meta[str(row["id"])] = row

    groups: dict[str, dict] = {}
    n = 0
    for f in replay_files(src):
        if n >= args.limit:
            break
        rep = read_replay(f)
        if args.player not in rep["players"]:
            continue
        seat = rep["players"].index(args.player)
        s = summarise(rep, args.player, meta.get(str(rep.get("id"))))
        g = groups.setdefault(s["result"], dict(games=0, flips=0, recap=0, caps=0,
                                                turns=0, young=0, birth=[], held=[],
                                                fr=[]))
        n += 1
        ages, birth, held, caps, lost = flips(rep, seat)
        g["games"] += 1
        g["flips"] += lost
        g["recap"] += len(ages)
        g["caps"] += caps
        g["turns"] += s["turns"]
        g["young"] += sum(1 for a in ages if a <= YOUNG)
        g["birth"] += birth
        g["held"] += held
        g["fr"].append(fronts(rep, seat))

    if not n:
        raise SystemExit(f"no replays featuring {args.player} under {src}")

    print(f"{n} games\n")
    print(f"{'':<8}{'games':>6}{'flips/g':>9}{'caps/g':>8}{'flips/100t':>12}"
          f"{'young%':>8}{'fronts':>8}")
    for res in ("win", "loss", "draw"):
        g = groups.get(res)
        if not g or not g["games"]:
            continue
        k = g["games"]
        print(f"{res:<8}{k:>6}{g['flips'] / k:>9.0f}{g['caps'] / k:>8.0f}"
              f"{100 * g['flips'] / max(g['turns'], 1):>12.1f}"
              f"{100 * g['young'] / max(g['recap'], 1):>8.0f}"
              f"{np.mean(g['fr']):>8.1f}")
    print(f"\n  flips/100t normalises for game length; losses are longer, so the\n"
          f"  raw flips/game gap overstates the effect.\n"
          f"  young% = share of LOST CAPTURES we had held {YOUNG} ticks or less.")

    print(f"\n{'':<8}{'birth army: flipped':>22}{'held':>10}{'ratio':>8}")
    for res in ("win", "loss"):
        g = groups.get(res)
        if not g or not g["birth"] or not g["held"]:
            continue
        b, h = float(np.median(g["birth"])), float(np.median(g["held"]))
        print(f"{res:<8}{b:>22.1f}{h:>10.1f}{b / max(h, 1e-9):>8.2f}")

    print("\nHow to read this:")
    print("  Flipped tiles born with markedly LESS army than tiles that held, and")
    print("  a high young%, means we take ground we never garrison — a scoring")
    print("  fix, and a cheap one.")
    print("  Similar birth armies and old flips mean the front is genuinely being")
    print("  fought over and lost. That is an economy problem and this tool is the")
    print("  wrong instrument for it.")
    print("  Fronts near 1.0 in both rows kills the 'multiple weak fronts' theory")
    print("  outright.")


if __name__ == "__main__":
    main()
