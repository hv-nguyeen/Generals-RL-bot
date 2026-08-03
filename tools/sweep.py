"""One-parameter sweep on a fixed set of boards.

The fastest useful loop in this repo: pick a knob, list values, get a score per
value on identical seeds. Common random numbers make small differences visible
at game counts where a naive A/B would be pure noise.

    python -m tools.sweep first_expand_turn 0,8,16,22,30 --games 120
    python -m tools.sweep expand.toward_enemy 0,1.5,4 --opponent greedy
    python -m tools.sweep castle_enabled 0,1        # bools take 0/1

Reports the land at fixed turns too, because win rate alone hides *why* a value
is better and the opening is judged on land, not on wins.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from analysis import stats as stats_mod
from arena import rating
from arena.runner import run_match, tally
from bot.config import Config


def apply(cfg: Config, name: str, raw: str) -> Config:
    cfg = Config.from_dict(asdict(cfg))
    if "." in name:
        block, attr = name.split(".")
        setattr(getattr(cfg, block), attr, float(raw))
        return cfg
    current = getattr(cfg, name)
    if isinstance(current, bool):
        setattr(cfg, name, bool(int(float(raw))))
    elif isinstance(current, int):
        setattr(cfg, name, int(round(float(raw))))
    else:
        setattr(cfg, name, float(raw))
    return cfg


def land_at(results: list[dict], turn: int) -> tuple[float, float]:
    mine, theirs, n = 0.0, 0.0, 0
    for r in results:
        a = r["a_seat"]
        for s in r["series"]:
            if s[0] == turn:
                mine += s[1] if a == 0 else s[3]
                theirs += s[3] if a == 0 else s[1]
                n += 1
                break
    return (mine / n, theirs / n) if n else (0.0, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("param")
    ap.add_argument("values", help="comma-separated")
    ap.add_argument("--base", default=None)
    ap.add_argument("--opponent", default="greedy")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--seed0", type=int, default=50_000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=900)
    ap.add_argument("--marks", default="60,120,200", help="turns to report land at")
    args = ap.parse_args()

    base = Config.load(args.base) if args.base else Config()
    marks = [int(x) for x in args.marks.split(",")]
    tmp = Path(tempfile.mkdtemp(prefix="sweep-"))

    header = (f"{args.param:>28}  score    W   D   L   elo      "
              + "  ".join(f"land@{m:<4}" for m in marks) + "  cast  turns")
    print(f"vs {args.opponent}, {args.games} games each, seeds {args.seed0}..")
    print(header)
    print("-" * len(header))

    rows = []
    for raw in args.values.split(","):
        cfg = apply(base, args.param, raw)
        path = tmp / f"{args.param.replace('.', '_')}_{raw}.json"
        cfg.save(path)
        results = run_match(f"ours:{path}", args.opponent, args.games, args.seed0,
                            args.workers, args.max_turns)
        w, d, loss = tally(results)
        summ = rating.summary(w, d, loss)
        agg = stats_mod.aggregate(results)
        lands = [land_at(results, m) for m in marks]
        land_txt = "  ".join(f"{a:5.1f}/{b:<4.0f}" for a, b in lands)
        print(f"{raw:>28}  {summ['score']:.3f}  {w:>3} {d:>3} {loss:>3}  "
              f"{summ['elo']:+7.1f}  {land_txt}  {agg['castles']:4.1f}  "
              f"{agg['mean_turns']:5.0f}")
        rows.append({"value": raw, **summ, "land": lands, "castles": agg["castles"],
                     "causes": agg["causes"]})

    best = max(rows, key=lambda r: r["score"])
    print(f"\nbest: {args.param}={best['value']}  score {best['score']:.3f}  "
          f"({best['wins']}W {best['draws']}D {best['losses']}L)")
    print("causes:", json.dumps(best["causes"]))
    print("land columns are ours/theirs")


if __name__ == "__main__":
    main()
