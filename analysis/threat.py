"""How much warning did we get before we died?

Our real losses are decapitations while ahead on macro, and three *standing*
defensive premiums (garrison, general-spend floor, threshold tuning) all measured
worse. The question that decides which defence is even the right shape is not
"how big was the stack" but "how long could we see it coming":

    window ~= 0        the stack was never observable in time. A standing
                       garrison sized to their army is genuinely required.
    window 5-15 ticks  there was time to react. Our interventions were the wrong
                       shape - the fix is a threat-triggered recall, which costs
                       nothing until it fires.
    never visible      we had the vision to see it and did not look. Buy
                       scouting, the cheapest fix of the three.

Vision is reconstructed exactly (the engine's rule is the 3x3 around owned
tiles), so "could we have seen it" is a fact here, not a guess.

    python -m analysis.threat runs/official2 --player H.V.Nguyen
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis.official import read_replay, replay_files, summarise
from bot.board import bfs_field_from, dilate8


def analyse(rep: dict, me: str) -> dict | None:
    players = rep["players"]
    if me not in players:
        return None
    seat = players.index(me)
    opp = 1 - seat
    ticks = rep["ticks"]
    if len(ticks) < 12:
        return None

    h, w = int(rep["dims"]["rows"]), int(rep["dims"]["cols"])
    mountains = np.zeros((h, w), dtype=bool)
    for r, c in rep["mountains"]:
        mountains[r, c] = True
    gr, gc = rep["generals"][seat]
    dist_home = bfs_field_from(~mountains, (gr, gc))

    death = len(ticks) - 1
    last = np.asarray(ticks[death]["armies"], dtype=np.int32)
    owners_prev = np.asarray(ticks[death - 1]["owners"], dtype=np.int32)
    armies_prev = np.asarray(ticks[death - 1]["armies"], dtype=np.int32)

    # The killer: the biggest enemy stack next to our general the tick before.
    killer = 0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = gr + dr, gc + dc
        if 0 <= nr < h and 0 <= nc < w and owners_prev[nr, nc] == opp:
            killer = max(killer, int(armies_prev[nr, nc]))
    if killer < 2:
        # not a clean decapitation (attrition loss); still report the biggest
        # enemy stack that ever stood within 3 steps of our general
        killer = 0
        for t in range(max(0, death - 40), death):
            o = np.asarray(ticks[t]["owners"], dtype=np.int32)
            a = np.asarray(ticks[t]["armies"], dtype=np.int32)
            near = (o == opp) & (dist_home <= 3)
            if near.any():
                killer = max(killer, int(a[near].max()))
    if killer < 2:
        return None

    # Walk backwards: when could we first SEE a stack at least half that size?
    threshold = max(2, killer // 2)
    first_seen = None
    first_exists = None
    for t in range(death):
        o = np.asarray(ticks[t]["owners"], dtype=np.int32)
        a = np.asarray(ticks[t]["armies"], dtype=np.int32)
        big = (o == opp) & (a >= threshold)
        if not big.any():
            continue
        if first_exists is None:
            first_exists = t
        if (big & dilate8(o == seat)).any():
            first_seen = t
            break

    my_army_home = None
    if first_seen is not None:
        o = np.asarray(ticks[first_seen]["owners"], dtype=np.int32)
        a = np.asarray(ticks[first_seen]["armies"], dtype=np.int32)
        reach = (o == seat) & (dist_home <= (death - first_seen))
        my_army_home = int(np.maximum(a[reach] - 1, 0).sum()) if reach.any() else 0

    return {
        "id": rep.get("id"),
        "opponent": players[opp],
        "death_tick": death,
        "killer_army": killer,
        "first_visible": first_seen,
        "first_existed": first_exists,
        "window": (death - first_seen) if first_seen is not None else None,
        "hidden_build_ticks": (first_seen - first_exists)
        if (first_seen is not None and first_exists is not None) else None,
        "defence_reachable": my_army_home,
        "could_have_held": (my_army_home is not None and my_army_home > killer),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default="runs/official2")
    ap.add_argument("--player", required=True)
    ap.add_argument("--only", default="loss", choices=["loss", "win", "all"])
    args = ap.parse_args()

    import json
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
        if args.only != "all" and s["result"] != args.only:
            continue
        a = analyse(rep, args.player)
        if a:
            rows.append(a)

    if not rows:
        raise SystemExit("nothing to analyse")

    print(f"{'opponent':<18}{'died':>6}{'killer':>8}{'1st seen':>10}{'window':>8}"
          f"{'hidden':>8}{'we had':>8}  held?")
    for r in sorted(rows, key=lambda r: r["death_tick"]):
        win = "never" if r["window"] is None else str(r["window"])
        print(f"{r['opponent'][:17]:<18}{r['death_tick']:>6}{r['killer_army']:>8}"
              f"{str(r['first_visible']):>10}{win:>8}"
              f"{str(r['hidden_build_ticks']):>8}{str(r['defence_reachable']):>8}"
              f"  {'yes' if r['could_have_held'] else 'no'}")

    windows = [r["window"] for r in rows if r["window"] is not None]
    never = sum(1 for r in rows if r["window"] is None)
    held = sum(1 for r in rows if r["could_have_held"])
    print(f"\n{len(rows)} games")
    print(f"  never saw it coming     : {never}")
    if windows:
        print(f"  warning window (ticks)  : median {int(np.median(windows))}, "
              f"p25 {int(np.percentile(windows, 25))}, p75 {int(np.percentile(windows, 75))}")
    print(f"  had enough army in reach: {held}/{len(rows)}  "
          f"<- these were losable-to-winnable by reacting, not by hoarding")


if __name__ == "__main__":
    main()
