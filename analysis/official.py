"""Pull our real leaderboard games off generals.bot and analyse them.

Everything else in this repo measures us against opponents we wrote ourselves,
which is exactly how you end up confident and wrong. These are the real games,
against the real field, and they arrive as full god-view state per tick.

The site is a single-page app, but the API behind it is plain JSON and needs no
auth:

    /api/leaderboard?profile=<name>            profile and standing
    /api/leaderboard?matches=1&player=<name>   match list with replay ids
    /api/leaderboard?replay=<id>               the replay itself

Replay schema (version 1):

    dims      {rows, cols}
    players   [name0, name1]        owner indices in `ticks` refer to this
    mountains [[r, c], ...]
    castles   [[r, c], ...]         initial only — built castles are NOT listed
    generals  [[r, c], ...]         index matches `players`
    ticks     [{armies: [[..]], owners: [[..]]}, ...]   owners: -1 neutral, 0, 1

Built castles have to be inferred, which matters because the castle economy is
our whole thesis: `castle_builds` finds them by the army they cost.

Needs network, so run it on the laptop, not the cluster.

    python -m analysis.official fetch --player H.V.Nguyen --out runs/official
    python -m analysis.official report runs/official
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from bot import rules
from sim import engine

API = "https://www.generals.bot/api/leaderboard"
UA = {"User-Agent": "generals-bot-analysis/1.0"}


# ---------------------------------------------------------------- fetching
def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def fetch_profile(player: str) -> dict:
    return _get(f"{API}?profile={urllib.parse.quote(player)}")


def fetch_matches(player: str) -> list[dict]:
    data = _get(f"{API}?matches=1&player={urllib.parse.quote(player)}")
    for key in ("matches", "games", "results"):
        if isinstance(data, dict) and isinstance(data.get(key), list):
            return data[key]
    return data if isinstance(data, list) else []


def fetch_replay(replay_id: int) -> dict:
    return _get(f"{API}?replay={replay_id}")


# ------------------------------------------------------------- conversion
def to_states(rep: dict):
    """Yield (tick, engine.State) so official replays feed our existing tools."""
    h, w = int(rep["dims"]["rows"]), int(rep["dims"]["cols"])
    mountains = np.zeros((h, w), dtype=bool)
    for r, c in rep["mountains"]:
        mountains[r, c] = True
    generals = np.zeros((h, w), dtype=bool)
    gpos = []
    for r, c in rep["generals"]:
        generals[r, c] = True
        gpos.append((int(r), int(c)))

    base_castles = np.zeros((h, w), dtype=bool)
    for r, c in rep.get("castles") or []:
        base_castles[r, c] = True

    builds = castle_builds(rep)
    castles = base_castles.copy()

    for t, tick in enumerate(rep["ticks"]):
        for (bt, br, bc) in builds:
            if bt == t:
                castles[br, bc] = True
        armies = np.asarray(tick["armies"], dtype=np.int32)
        owners = np.asarray(tick["owners"], dtype=np.int32)
        own = np.stack([owners == 0, owners == 1])
        st = engine.State(
            armies=armies,
            own=own,
            neutral=(owners < 0) & ~mountains,
            generals=generals,
            castles=castles.copy(),
            mountains=mountains,
            passable=~mountains,
            gpos=list(gpos),
        )
        st.time = t
        yield t, st


def castle_builds(rep: dict) -> list[tuple[int, int, int]]:
    """(tick, row, col) for every castle built during the game.

    The tick data has no castle channel, so builds are found by their price: a
    cell loses at least the 35-army base cost in one tick, keeps its owner, and
    no neighbour picks that army up (which is what a move would look like).
    """
    ticks = rep["ticks"]
    out: list[tuple[int, int, int]] = []
    if len(ticks) < 2:
        return out
    prev_a = np.asarray(ticks[0]["armies"], dtype=np.int32)
    prev_o = np.asarray(ticks[0]["owners"], dtype=np.int32)
    for t in range(1, len(ticks)):
        a = np.asarray(ticks[t]["armies"], dtype=np.int32)
        o = np.asarray(ticks[t]["owners"], dtype=np.int32)
        delta = a - prev_a
        for r, c in np.argwhere((delta <= -rules.BASE_COST) & (o == prev_o) & (o >= 0)):
            r, c = int(r), int(c)
            lost = -int(delta[r, c])
            gained = 0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < a.shape[0] and 0 <= nc < a.shape[1]:
                    gained = max(gained, int(a[nr, nc] - prev_a[nr, nc]))
            # a move would deposit what it took onto one neighbour
            if gained < lost // 2:
                out.append((t, r, c))
        prev_a, prev_o = a, o
    return out


def summarise(rep: dict, me: str, meta: dict | None = None) -> dict:
    """Per-game facts, from our seat's point of view."""
    players = rep["players"]
    seat = 0 if players[0] == me else 1
    opp_seat = 1 - seat
    ticks = rep["ticks"]
    turns = len(ticks) - 1

    land = [[], []]
    army = [[], []]
    for tick in ticks:
        a = np.asarray(tick["armies"], dtype=np.int32)
        o = np.asarray(tick["owners"], dtype=np.int32)
        for p in (0, 1):
            m = o == p
            land[p].append(int(m.sum()))
            army[p].append(int(a[m].sum()))

    # The match list carries the authoritative result; inferring it from the
    # final land count gets draws and timeouts wrong.
    winner = -1
    if meta and meta.get("winner") in ("A", "B"):
        winner_name = meta["a_name"] if meta["winner"] == "A" else meta["b_name"]
        winner = players.index(winner_name) if winner_name in players else -1
    else:
        if land[0][-1] == 0 and land[1][-1] > 0:
            winner = 1
        elif land[1][-1] == 0 and land[0][-1] > 0:
            winner = 0

    builds = castle_builds(rep)
    owners_last = np.asarray(ticks[-1]["owners"], dtype=np.int32)
    my_builds, opp_builds = [], []
    for t, r, c in builds:
        o = int(np.asarray(ticks[t]["owners"], dtype=np.int32)[r, c])
        (my_builds if o == seat else opp_builds).append(t)

    def at(series, turn):
        return series[min(turn, len(series) - 1)]

    return {
        "id": rep.get("id"),
        "opponent": players[opp_seat],
        "seat": seat,
        "turns": turns,
        "result": "win" if winner == seat else ("loss" if winner == opp_seat else "draw"),
        "land_100": [at(land[seat], 100), at(land[opp_seat], 100)],
        "land_200": [at(land[seat], 200), at(land[opp_seat], 200)],
        "army_100": [at(army[seat], 100), at(army[opp_seat], 100)],
        "my_castles": len(my_builds),
        "opp_castles": len(opp_builds),
        "my_first_castle": min(my_builds) if my_builds else None,
        "opp_first_castle": min(opp_builds) if opp_builds else None,
        "final_land": [land[seat][-1], land[opp_seat][-1]],
    }


# -------------------------------------------------------------------- CLI
def cmd_fetch(args) -> None:
    out = Path(args.out)
    (out / "replays").mkdir(parents=True, exist_ok=True)
    matches = fetch_matches(args.player)
    print(f"{len(matches)} matches for {args.player}")
    (out / "matches.json").write_text(json.dumps(matches, indent=2))

    ids = []
    for m in matches:
        mid = m.get("id") or m.get("replay") or m.get("match_id")
        if mid is not None:
            ids.append(int(mid))
    ids = ids[:args.limit] if args.limit else ids

    for i, mid in enumerate(ids, 1):
        path = out / "replays" / f"{mid}.json"
        if path.exists() and not args.force:
            continue
        try:
            rep = fetch_replay(mid)
        except Exception as e:                       # noqa: BLE001
            print(f"  {mid}: {e}")
            continue
        rep["id"] = mid
        path.write_text(json.dumps(rep))
        print(f"  [{i}/{len(ids)}] {mid} -> {path} ({path.stat().st_size // 1024} kB)")
        time.sleep(args.delay)


def cmd_report(args) -> None:
    from analysis import viewer

    out = Path(args.dir)
    reps = sorted((out / "replays").glob("*.json"))
    if not reps:
        raise SystemExit(f"no replays in {out / 'replays'} — run `fetch` first")

    meta_by_id = {}
    mpath = out / "matches.json"
    if mpath.exists():
        meta_by_id = {str(m["id"]): m for m in json.loads(mpath.read_text())}

    rows = []
    for path in reps:
        rep = json.loads(path.read_text())
        if args.player not in rep["players"]:
            continue
        s = summarise(rep, args.player, meta_by_id.get(str(rep.get("id"))))
        s["file"] = path.name
        rows.append(s)
        if args.render:
            meta = {"seed": rep.get("seed"), "spec0": rep["players"][0],
                    "spec1": rep["players"][1],
                    "winner": 0 if s["result"] == "win" and s["seat"] == 0 else -1,
                    "reason": s["result"], "turns": s["turns"]}
            data = viewer.pack_states(to_states(rep), meta, args.every)
            html = viewer._HTML.format(
                seed=meta["seed"], spec0=meta["spec0"], spec1=meta["spec1"],
                reason=meta["reason"], turns=meta["turns"], data=json.dumps(data))
            (path.with_suffix(".html")).write_text(html)

    rows.sort(key=lambda r: (r["opponent"], r["turns"]))
    print(f"{'opponent':<22}{'res':<6}{'turns':>6}{'land@100':>12}{'land@200':>12}"
          f"{'castles':>10}{'first':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['opponent'][:21]:<22}{r['result']:<6}{r['turns']:>6}"
              f"{r['land_100'][0]:>5}/{r['land_100'][1]:<6}"
              f"{r['land_200'][0]:>5}/{r['land_200'][1]:<6}"
              f"{r['my_castles']:>4}/{r['opp_castles']:<5}"
              f"{str(r['my_first_castle'] or '-'):>8}")

    wins = sum(1 for r in rows if r["result"] == "win")
    opp_build = [r for r in rows if r["opp_castles"] > 0]
    print(f"\n{len(rows)} games, {wins}W {len(rows) - wins}L/D")
    print(f"opponents that built castles: {len(opp_build)}/{len(rows)} games")
    if rows:
        print(f"mean land@100  us {np.mean([r['land_100'][0] for r in rows]):.1f}"
              f"  them {np.mean([r['land_100'][1] for r in rows]):.1f}")
        print(f"mean land@200  us {np.mean([r['land_200'][0] for r in rows]):.1f}"
              f"  them {np.mean([r['land_200'][1] for r in rows]):.1f}")
    if args.render:
        print(f"\nreplays rendered next to their json in {out / 'replays'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download our matches and replays")
    f.add_argument("--player", required=True)
    f.add_argument("--out", default="runs/official")
    f.add_argument("--limit", type=int, default=0, help="0 = all")
    f.add_argument("--delay", type=float, default=0.3, help="seconds between requests")
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("report", help="summarise downloaded replays")
    r.add_argument("dir", nargs="?", default="runs/official")
    r.add_argument("--player", required=True)
    r.add_argument("--render", action="store_true", help="also write HTML replays")
    r.add_argument("--every", type=int, default=2)
    r.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
