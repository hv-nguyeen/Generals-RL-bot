"""Where the 150 ms goes.

    python -m tools.profile_turn --games 3            # timing distribution
    python -m tools.profile_turn --games 1 --cprofile # hot functions

The distribution is what matters, not the mean: one move over budget is one
fault, and fifty faults forfeit the game.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time

import numpy as np

from arena import agents as agents_mod
from bot import rules
from sim import engine, mapgen


def collect(games: int, max_turns: int, opponent: str, seed0: int) -> list[float]:
    times: list[float] = []
    for g in range(games):
        grid = mapgen.generate(seed0 + g)
        st = engine.from_grid(grid)
        me = agents_mod.make("ours", 0, *grid.shape)
        opp = agents_mod.make(opponent, 1, *grid.shape)
        for _ in range(max_turns):
            o0, o1 = engine.observe(st, 0), engine.observe(st, 1)
            t0 = time.perf_counter()
            a0 = me.act(o0)
            times.append((time.perf_counter() - t0) * 1000.0)
            if engine.step(st, a0, opp.act(o1)):
                break
    return times


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=800)
    ap.add_argument("--opponent", default="greedy")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--cprofile", action="store_true")
    args = ap.parse_args()

    if args.cprofile:
        pr = cProfile.Profile()
        pr.enable()
        collect(args.games, args.max_turns, args.opponent, args.seed0)
        pr.disable()
        pstats.Stats(pr).sort_stats("cumulative").print_stats(22)
        return

    t = np.array(collect(args.games, args.max_turns, args.opponent, args.seed0))
    budget = rules.MOVE_BUDGET_S * 1000
    print(f"{len(t)} moves over {args.games} games")
    for label, v in (("mean", t.mean()), ("p50", np.percentile(t, 50)),
                     ("p90", np.percentile(t, 90)), ("p99", np.percentile(t, 99)),
                     ("max", t.max())):
        print(f"  {label:<5} {v:7.2f} ms")
    over = int((t > budget).sum())
    print(f"  over {budget:.0f} ms: {over} ({100.0 * over / len(t):.2f}%)")
    print(f"  headroom at p99: {budget / max(np.percentile(t, 99), 1e-9):.0f}x")


if __name__ == "__main__":
    main()
