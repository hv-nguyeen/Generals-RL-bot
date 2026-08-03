"""What did we do in the final ticks before the general fell?

Viet watched replays and reported the bot walking army *out* of its general
while an enemy stack was two tiles away. That is a specific, falsifiable claim,
and the replays can settle it: growth on the general is deterministic (+1 on
even ticks, +1 more every 50th), so any shortfall against that is army we
deliberately spent.

For each loss this reports, over the last `window` ticks:

    spent   ticks where army left the general of our own accord
    lost    total army we moved off it
    thr     the nearest enemy stack at the time
    react   ticks where we instead moved army ONTO the general

A high `spent` with a live `thr` is the bug Viet described. A high `react` means
we were trying and simply arrived short.

    python -m analysis.deathwatch runs/official6 --player H.V.Nguyen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.official import read_replay, replay_files, summarise
from bot.board import bfs_field_from


def watch(rep: dict, me: str, window: int = 15) -> dict | None:
    players = rep["players"]
    if me not in players:
        return None
    seat = players.index(me)
    opp = 1 - seat
    ticks = rep["ticks"]
    if len(ticks) < window + 2:
        return None

    h, w = int(rep["dims"]["rows"]), int(rep["dims"]["cols"])
    mountains = np.zeros((h, w), dtype=bool)
    for r, c in rep["mountains"]:
        mountains[r, c] = True
    gr, gc = rep["generals"][seat]
    dist = bfs_field_from(~mountains, (gr, gc))

    death = len(ticks) - 1
    spent = lost = react = gained = 0
    nearest_threat = 0
    trace = []
    for t in range(max(0, death - window), death):
        a0 = np.asarray(ticks[t]["armies"], dtype=np.int32)
        o0 = np.asarray(ticks[t]["owners"], dtype=np.int32)
        a1 = np.asarray(ticks[t + 1]["armies"], dtype=np.int32)
        o1 = np.asarray(ticks[t + 1]["owners"], dtype=np.int32)
        if o0[gr, gc] != seat:
            continue

        # deterministic growth on a general we still hold
        growth = (1 if (t + 1) % 2 == 0 else 0) + (1 if (t + 1) % 50 == 0 else 0)
        expected = int(a0[gr, gc]) + growth
        actual = int(a1[gr, gc])

        # biggest enemy stack within three steps, right now
        near = (o0 == opp) & (dist <= 3)
        threat = int(a0[near].max()) - 1 if near.any() else 0
        nearest_threat = max(nearest_threat, threat)

        if o1[gr, gc] == seat:          # still ours: any change was our own doing
            if actual < expected:
                # confirm the army landed on a neighbour we own, i.e. we moved it
                moved = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = gr + dr, gc + dc
                    if 0 <= nr < h and 0 <= nc < w and o1[nr, nc] == seat:
                        moved = max(moved, int(a1[nr, nc]) - int(a0[nr, nc]))
                if moved > 0:
                    spent += 1
                    lost += expected - actual
                    trace.append((t, expected, actual, threat, "OUT"))
            elif actual > expected:
                react += 1
                gained += actual - expected
                trace.append((t, expected, actual, threat, "in"))

    return {
        "id": rep.get("id"), "death": death,
        "spent": spent, "lost": lost, "react": react, "gained": gained,
        "threat": nearest_threat,
        "final_general": int(np.asarray(ticks[death - 1]["armies"])[gr, gc]),
        "trace": trace,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default="runs/official6")
    ap.add_argument("--player", required=True)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--show", type=int, default=3, help="print tick traces for N games")
    args = ap.parse_args()

    out = Path(args.dir)
    meta = {}
    for m in out.glob("matches*.json"):
        for row in json.loads(m.read_text()):
            meta[str(row["id"])] = row

    rows = []
    for f in replay_files(out):
        rep = read_replay(f)
        if args.player not in rep["players"]:
            continue
        s = summarise(rep, args.player, meta.get(str(rep.get("id"))))
        if s["result"] != "loss":
            continue
        r = watch(rep, args.player, args.window)
        if r:
            r["opponent"] = s["opponent"]
            rows.append(r)

    if not rows:
        raise SystemExit("no losses found")

    print(f"last {args.window} ticks before the general fell\n")
    print(f"{'opponent':<20}{'died':>6}{'gen':>6}{'threat':>8}"
          f"{'spent':>7}{'army out':>10}{'react':>7}{'army in':>9}")
    for r in sorted(rows, key=lambda r: -r["lost"]):
        print(f"{r['opponent'][:19]:<20}{r['death']:>6}{r['final_general']:>6}"
              f"{r['threat']:>8}{r['spent']:>7}{r['lost']:>10}{r['react']:>7}{r['gained']:>9}")

    n = len(rows)
    bad = [r for r in rows if r["spent"] > 0 and r["threat"] > 0]
    print(f"\n{n} losses")
    print(f"  moved army OFF the general with a threat within 3 steps: {len(bad)}/{n}")
    print(f"  mean army walked off in the last {args.window} ticks       : "
          f"{np.mean([r['lost'] for r in rows]):.1f}")
    print(f"  mean army brought in                                  : "
          f"{np.mean([r['gained'] for r in rows]):.1f}")
    print(f"  mean army on the general the tick before death        : "
          f"{np.mean([r['final_general'] for r in rows]):.1f}")

    for r in sorted(rows, key=lambda r: -r["lost"])[:args.show]:
        if not r["trace"]:
            continue
        print(f"\n  {r['opponent']} (#{r['id']}), died tick {r['death']}:")
        for t, exp, act, thr, kind in r["trace"]:
            print(f"    tick {t:>4}  general {exp:>4} -> {act:<4} "
                  f"{'MOVED OUT' if kind == 'OUT' else 'reinforced'}"
                  f"   nearest enemy stack {thr}")


if __name__ == "__main__":
    main()
