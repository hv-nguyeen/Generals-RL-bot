"""Expansion fingerprints: how fast a bot takes ground, and where it loses turns.

The top of the leaderboard reaches ~110 land by turn 200 and we reach ~76. Only
one move happens per turn, so land is bounded by turns: a bot that captures on
every single turn from its first expansion is at the theoretical ceiling. The
gap therefore has to be *wasted turns*, and this measures exactly that.

Works on any (turn, State) stream, so official replays and our own arena games
are measured with the same code.

    python -m analysis.expansion official runs/official --player H.V.Nguyen
    python -m analysis.expansion ours --games 20
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def fingerprint(states, seat: int, horizon: int = 200) -> dict:
    """Expansion statistics for one player over the first `horizon` turns."""
    land, army, gen_army, best_stack = [], [], [], []
    for turn, st in states:
        if turn > horizon:
            break
        mine = st.own[seat]
        land.append(int(mine.sum()))
        army.append(int((st.armies * mine).sum()))
        gr, gc = st.gpos[seat]
        gen_army.append(int(st.armies[gr, gc]) if mine[gr, gc] else 0)
        best_stack.append(int((st.armies * mine).max()) if mine.any() else 0)

    if len(land) < 3:
        return {}
    d = np.diff(np.asarray(land))
    first = next((i + 1 for i, v in enumerate(d) if v > 0), None)
    active = d[first - 1:] if first else d
    gained = active > 0

    def at(series, t):
        return series[min(t, len(series) - 1)]

    return {
        "first_expand": first,
        "land_50": at(land, 50), "land_100": at(land, 100),
        "land_150": at(land, 150), "land_200": at(land, 200),
        "army_100": at(army, 100), "army_200": at(army, 200),
        # the headline: of the turns after we started expanding, how many
        # actually took a tile? 1.0 is the ceiling.
        "capture_rate": float(gained.mean()) if len(gained) else 0.0,
        "idle_turns": int((~gained).sum()),
        "gen_army_100": at(gen_army, 100),
        "peak_stack": max(best_stack) if best_stack else 0,
        "turns": len(land) - 1,
    }


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _show(title: str, groups: dict[str, list[dict]]) -> None:
    cols = ["first_expand", "land_50", "land_100", "land_200",
            "capture_rate", "idle_turns", "gen_army_100", "peak_stack"]
    print(f"\n{title}")
    print(f"{'who':<24}" + "".join(f"{c:>14}" for c in cols))
    print("-" * (24 + 14 * len(cols)))
    for name, rows in groups.items():
        if not rows:
            continue
        line = f"{name[:23]:<24}"
        for c in cols:
            v = _mean(rows, c)
            line += f"{v:>14.2f}" if c == "capture_rate" else f"{v:>14.1f}"
        print(line + f"   ({len(rows)} games)")


def cmd_official(args) -> None:
    from analysis.official import to_states

    groups: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(str(Path(args.dir) / "replays" / "*.json"))):
        rep = json.loads(Path(f).read_text())
        if args.player not in rep["players"]:
            continue
        me = rep["players"].index(args.player)
        opp = 1 - me
        opp_name = rep["players"][opp]
        groups.setdefault(f"us (vs {opp_name})" if args.split else "us", []).append(
            fingerprint(to_states(rep), me, args.horizon))
        groups.setdefault(opp_name, []).append(
            fingerprint(to_states(rep), opp, args.horizon))
    _show("official leaderboard replays", groups)


def cmd_ours(args) -> None:
    from arena import agents as agents_mod
    from sim import engine, mapgen

    rows_us, rows_them = [], []
    for seed in range(args.games):
        grid = mapgen.generate(seed)
        st = engine.from_grid(grid)
        players = [agents_mod.make("ours", 0, *grid.shape),
                   agents_mod.make(args.opponent, 1, *grid.shape)]
        states = [(0, st.copy())]
        for t in range(args.horizon):
            acts = [players[p].act(engine.observe(st, p)) for p in (0, 1)]
            done = engine.step(st, acts[0], acts[1])
            states.append((t + 1, st.copy()))
            if done:
                break
        rows_us.append(fingerprint(iter(states), 0, args.horizon))
        rows_them.append(fingerprint(iter(states), 1, args.horizon))
    _show(f"our arena (ours vs {args.opponent})",
          {"ours": rows_us, args.opponent: rows_them})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("official")
    o.add_argument("dir", nargs="?", default="runs/official")
    o.add_argument("--player", required=True)
    o.add_argument("--horizon", type=int, default=200)
    o.add_argument("--split", action="store_true", help="break our rows out per opponent")
    o.set_defaults(func=cmd_official)

    u = sub.add_parser("ours")
    u.add_argument("--games", type=int, default=20)
    u.add_argument("--opponent", default="greedy")
    u.add_argument("--horizon", type=int, default=200)
    u.set_defaults(func=cmd_ours)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
