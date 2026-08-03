"""What is a build's Elo once you control for who it happened to play?

Raw ladder Elo from a 36-game batch is worth +-58, which is wider than every
difference we have tried to measure all day. A large part of that width is not
luck in the games — it is luck in the draw. v16 met Hunter six times and
lukbrezina six times; a build that met neither is not comparable to it.

Opponents have their own published ratings, so treat those as known and fit only
one free parameter per build: the strength that best explains our results
against the specific opponents we actually faced. That is a performance rating
with a real confidence interval, and it removes the part of the noise that comes
from matchmaking rather than from play.

Match ids increase with time, so a build is a range of ids. Give the first id
each build played and everything after it belongs to that build until the next.

    python -m analysis.ab /local/data/vng205/field --player H.V.Nguyen \\
        --builds v13:0 v16:84100

Read the overlap, not the point estimates. If two intervals overlap heavily the
ladder has not separated those builds no matter how different the Elo looks.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from analysis.official import Forbidden, fetch_profile

SCALE = math.log(10) / 400.0


def expected(mine: float, theirs: float) -> float:
    return 1.0 / (1.0 + math.exp(SCALE * (theirs - mine)))


def performance(games: list[tuple[float, float]]) -> tuple[float, float]:
    """MLE rating and its standard error from (opponent_elo, score) pairs.

    The log-likelihood is concave in our rating and its derivative is monotone,
    so bisection finds the maximum without needing a gradient step size. A build
    that won or lost every game has no finite MLE; the bracket ends pinned and
    the caller sees the +-2000 interval for what it is.
    """
    def slope(s: float) -> float:
        return sum(sc - expected(s, opp) for opp, sc in games)

    lo, hi = -2000.0, 4000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if slope(mid) > 0:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2
    info = SCALE ** 2 * sum(expected(s, o) * (1 - expected(s, o)) for o, _ in games)
    return s, (1.0 / math.sqrt(info) if info > 0 else float("inf"))


def opponent_elos(names: list[str], cache: Path) -> dict[str, float]:
    """Published rating per opponent, cached — this is the slow, polite part."""
    known: dict[str, float] = {}
    if cache.exists():
        known = json.loads(cache.read_text())
    for n in names:
        if n in known:
            continue
        try:
            known[n] = float((fetch_profile(n).get("rating") or {}).get("elo") or 1500)
        except (Forbidden, KeyError, ValueError, TypeError):
            known[n] = 1500.0
        print(f"  {n}: {known[n]:.0f}", flush=True)
        cache.write_text(json.dumps(known, indent=2))
    return known


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", default="/local/data/vng205/field")
    ap.add_argument("--player", required=True)
    ap.add_argument("--builds", nargs="+", required=True,
                    help="name:first_match_id, e.g. v13:0 v16:84100")
    args = ap.parse_args()

    src = Path(args.src)
    rows = []
    for m in list(src.glob("matches.json")) + list(src.glob("matches_*.json")):
        rows += json.loads(m.read_text())
    ours = []
    for r in rows:
        a, b = r.get("a_name"), r.get("b_name")
        if args.player not in (a, b) or r.get("id") is None:
            continue
        me = "A" if a == args.player else "B"
        opp = b if me == "A" else a
        w = r.get("winner")
        score = 1.0 if w == me else (0.0 if w in ("A", "B") else 0.5)
        ours.append((int(r["id"]), opp, score))
    if not ours:
        raise SystemExit(f"no matches for {args.player} under {src}")
    ours.sort()

    cuts = sorted(((int(b.split(":")[1]), b.split(":")[0]) for b in args.builds))
    def build_of(mid: int) -> str | None:
        name = None
        for start, n in cuts:
            if mid >= start:
                name = n
        return name

    print("fetching opponent ratings (1 req/s)")
    elos = opponent_elos(sorted({o for _, o, _ in ours}), src / "opponent_elo.json")

    fits = {}
    for _, name in cuts:
        games = [(elos[o], s) for mid, o, s in ours if build_of(mid) == name]
        if not games:
            print(f"\n{name}: no games in range")
            continue
        s, se = performance(games)
        fits[name] = (s, se, len(games))
        raw = sum(sc for _, sc in games) / len(games)
        mean_opp = sum(o for o, _ in games) / len(games)
        print(f"\n{name}: {len(games)} games, {raw:.1%} score vs mean opponent {mean_opp:.0f}")
        print(f"  performance rating {s:.0f} +- {se:.0f}")
        per: dict[str, list[float]] = {}
        for mid, o, sc in ours:
            if build_of(mid) == name:
                per.setdefault(o, []).append(sc)
        for o, scs in sorted(per.items(), key=lambda kv: -elos[kv[0]]):
            print(f"    {o[:24]:<26}{elos[o]:>6.0f}  "
                  f"{sum(scs):.1f}/{len(scs)}")

    names = [n for _, n in cuts if n in fits]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            d = fits[b][0] - fits[a][0]
            se = math.hypot(fits[a][1], fits[b][1])
            print(f"\n{b} - {a}: {d:+.0f} +- {se:.0f} elo "
                  f"({abs(d) / se if se else 0:.1f} sigma)")
            if se and abs(d) / se < 1.96:
                print("  not separated - the ladder cannot tell these apart yet")


if __name__ == "__main__":
    main()
