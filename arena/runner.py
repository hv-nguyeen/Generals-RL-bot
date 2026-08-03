"""Headless match runner.

Plays agents in-process against the numpy mirror — no subprocesses, no string
encoding — which is roughly two orders of magnitude faster than the starter
kit's `matchup.py` and is what makes SPRT-gated iteration practical.

Every seed is played twice with the colours swapped, so a result can never be an
artifact of who moved first or which spawn was better.

    python -m arena.runner --a ours --b expander --games 200 --workers 12 \
        --out runs/baseline --replays
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arena import agents as agents_mod
from arena import rating
from bot import rules
from sim import engine, mapgen

SAMPLE_EVERY = 10


@dataclass
class GameSpec:
    spec0: str
    spec1: str
    seed: int
    max_turns: int = rules.TURN_LIMIT
    time_limit: float = rules.MOVE_BUDGET_S
    record: bool = False


def _valid_action(a) -> bool:
    try:
        return len(a) == 5 and all(isinstance(int(x), int) for x in a)
    except (TypeError, ValueError):
        return False


def play(g: GameSpec) -> dict:
    grid = mapgen.generate(g.seed)
    h, w = grid.shape
    st = engine.from_grid(grid)
    players = [
        agents_mod.make(g.spec0, 0, h, w, g.seed),
        agents_mod.make(g.spec1, 1, h, w, g.seed),
    ]

    faults = [0, 0]
    max_ms = [0.0, 0.0]
    total_ms = [0.0, 0.0]
    modes: list[dict[str, int]] = [{}, {}]
    first_castle = [None, None]
    actions_log: list[list[list[int]]] = []
    series: list[list[int]] = []

    winner, reason, turn = -1, "draw", 0
    while turn < g.max_turns:
        acts = []
        for p in (0, 1):
            obs = engine.observe(st, p)
            budget = rules.FIRST_MOVE_BUDGET_S if turn == 0 else g.time_limit
            t0 = time.perf_counter()
            try:
                a = players[p].act(obs, t0 + budget * 0.95)
            except Exception:                       # noqa: BLE001
                a, faults[p] = rules.PASS_ACTION, faults[p] + 1
            elapsed = time.perf_counter() - t0

            if not _valid_action(a):
                a, faults[p] = rules.PASS_ACTION, faults[p] + 1
            if elapsed > budget:
                a, faults[p] = rules.PASS_ACTION, faults[p] + 1
            ms = elapsed * 1000.0
            total_ms[p] += ms
            max_ms[p] = max(max_ms[p], ms)

            dbg = getattr(players[p], "last_debug", None)
            if dbg:
                m = dbg.get("mode", "?")
                modes[p][m] = modes[p].get(m, 0) + 1
            acts.append([int(x) for x in a])

        if faults[0] >= rules.MAX_FAULTS or faults[1] >= rules.MAX_FAULTS:
            winner = 1 if faults[0] >= rules.MAX_FAULTS else 0
            reason = "forfeit"
            break

        if g.record:
            actions_log.append(acts)

        done = engine.step(st, acts[0], acts[1])
        turn += 1

        for p in (0, 1):
            if first_castle[p] is None and bool((st.castles & st.own[p]).any()):
                first_castle[p] = turn

        if turn % SAMPLE_EVERY == 0 or done:
            land, army = st.land(), st.army()
            series.append([turn, land[0], army[0], land[1], army[1],
                           int((st.castles & st.own[0]).sum()),
                           int((st.castles & st.own[1]).sum())])

        if done:
            winner = st.winner
            reason = "capture" if winner >= 0 else "mutual"
            break

    for pl in players:
        if hasattr(pl, "close"):
            pl.close()

    land, army = st.land(), st.army()
    result = {
        "seed": g.seed, "h": int(h), "w": int(w),
        "spec0": g.spec0, "spec1": g.spec1,
        "winner": int(winner), "reason": reason, "turns": turn,
        "land": list(land), "army": list(army),
        "castles": [int((st.castles & st.own[0]).sum()), int((st.castles & st.own[1]).sum())],
        "first_castle": first_castle,
        "faults": faults,
        "max_ms": [round(x, 2) for x in max_ms],
        "mean_ms": [round(total_ms[p] / max(turn, 1), 3) for p in (0, 1)],
        "modes": modes,
        "series": series,
    }
    if g.record:
        result["replay"] = {"grid": grid.tolist(), "actions": actions_log}
    return result


def _worker(payload: dict) -> dict:
    return play(GameSpec(**payload))


def run_match(spec_a: str, spec_b: str, games: int, seed0: int = 0, workers: int = 8,
              max_turns: int = rules.TURN_LIMIT, time_limit: float = rules.MOVE_BUDGET_S,
              record: bool = False, progress=None) -> list[dict]:
    """Play `games` games, colours swapped on alternate games of each seed pair."""
    jobs = []
    for i in range(games):
        seed = seed0 + i // 2
        swapped = i % 2 == 1
        s0, s1 = (spec_b, spec_a) if swapped else (spec_a, spec_b)
        jobs.append(dict(spec0=s0, spec1=s1, seed=seed, max_turns=max_turns,
                         time_limit=time_limit, record=record))

    results = []
    if workers <= 1:
        for j in jobs:
            results.append(_worker(j))
            if progress:
                progress(len(results), len(jobs))
        return _tag(results, spec_a, spec_b)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(_worker, jobs, chunksize=1):
            results.append(r)
            if progress:
                progress(len(results), len(jobs))
    return _tag(results, spec_a, spec_b)


def _tag(results: list[dict], spec_a: str, spec_b: str) -> list[dict]:
    """Annotate each game with the outcome from A's point of view."""
    for r in results:
        a_seat = 0 if r["spec0"] == spec_a else 1
        r["a_seat"] = a_seat
        if r["winner"] < 0:
            r["a_result"] = "draw"
        else:
            r["a_result"] = "win" if r["winner"] == a_seat else "loss"
    return results


def tally(results: list[dict]) -> tuple[int, int, int]:
    w = sum(1 for r in results if r["a_result"] == "win")
    d = sum(1 for r in results if r["a_result"] == "draw")
    return w, d, len(results) - w - d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="ours", help="agent A spec")
    ap.add_argument("--b", default="expander", help="agent B spec")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=rules.TURN_LIMIT)
    ap.add_argument("--time-limit-ms", type=float, default=rules.MOVE_BUDGET_S * 1000)
    ap.add_argument("--out", default=None, help="directory to write results.jsonl into")
    ap.add_argument("--replays", action="store_true", help="store full replays (a few kB each)")
    ap.add_argument("--elo1", type=float, default=12.0, help="SPRT alternative hypothesis")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()

    def progress(done, total):
        if not args.quiet:
            print(f"\r  {done}/{total} games", end="", flush=True)

    results = run_match(args.a, args.b, args.games, args.seed0, args.workers,
                        args.max_turns, args.time_limit_ms / 1000.0,
                        args.replays, progress)
    if not args.quiet:
        print()

    w, d, loss = tally(results)
    summary = rating.summary(w, d, loss)
    test = rating.sprt(w, d, loss, 0.0, args.elo1)

    print(f"{args.a}  vs  {args.b}")
    print(f"  {w}W {d}D {loss}L  score {summary['score']:.3f}")
    print(f"  elo  {summary['elo']:+.1f}  [{summary['elo_lo']:+.1f}, {summary['elo_hi']:+.1f}]")
    print(f"  sprt llr {test['llr']:+.2f}  bounds [{test['lower']:.2f}, {test['upper']:.2f}]"
          f"  -> {test['verdict']}")
    slow = max(max(r["max_ms"]) for r in results)
    mean = float(np.mean([r["mean_ms"][r["a_seat"]] for r in results]))
    faults = sum(sum(r["faults"]) for r in results)
    print(f"  time A mean {mean:.1f} ms, slowest move in run {slow:.1f} ms, faults {faults}")
    print(f"  wall {time.time() - t0:.1f}s")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        if args.replays:
            from analysis import replay as replay_mod
            (out / "replays").mkdir(exist_ok=True)
            for i, r in enumerate(results):
                if "replay" in r:
                    path = out / "replays" / f"game_{i:04d}.json"
                    replay_mod.save(path, r)
                    r.pop("replay")
                    r["replay_path"] = str(path.relative_to(out))
        with (out / "results.jsonl").open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        (out / "summary.json").write_text(json.dumps(
            {"a": args.a, "b": args.b, **summary, "sprt": test}, indent=2) + "\n")
        print(f"  wrote {out}/results.jsonl")


if __name__ == "__main__":
    main()
