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
    maps: str | None = None      # pool of real boards; None = generate
    # Generals-distance band. None is the competition generator exactly, so an
    # unset run is byte-identical to every result measured before this existed.
    # Set it to sweep a behaviour against distance: `bld` sat at 0.01 for all of
    # stage 3 (11-17) and hit 0.5 within four iterations of stage 4 (17-24), so
    # the policy has a castle breakpoint somewhere in between and nothing could
    # measure where.
    dmin: int | None = None
    dmax: int | None = None


def _valid_action(a) -> bool:
    try:
        return len(a) == 5 and all(isinstance(int(x), int) for x in a)
    except (TypeError, ValueError):
        return False


def play(g: GameSpec) -> dict:
    grid = (mapgen.pool_grid(g.maps, g.seed) if g.maps
            else mapgen.generate(g.seed) if g.dmin is None
            else mapgen.generate(g.seed, g.dmin, g.dmax))
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
    built = [0, 0]
    captured = [0, 0]
    # Copies: the engine mutates these arrays in place, so a reference would
    # compare the board against itself and report zero of everything.
    prev_castles = st.castles.copy()
    prev_own = [st.own[0].copy(), st.own[1].copy()]
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

        # Built vs CAPTURED, exactly, and they are different strategies. Building
        # is symmetric -- both sides do it, both gain, and the win-probability
        # differential in a mirror is ~0, which is why the gradient on it is so
        # weak. Taking THEIR castle is zero-sum: they lose the income and we gain
        # it, so self-play can see it without any opponent surgery. It is also
        # cheaper, costing the garrison rather than 35 + 14 per nearby structure.
        #
        # `series` samples every 10 turns and cannot separate the two -- a build
        # and a capture inside one window look identical, and a castle taken and
        # retaken is invisible. This is per-turn and exact.
        fresh = st.castles & ~prev_castles          # tiles that became castles
        for p in (0, 1):
            # A build converts a tile you ALREADY own, so ownership does not
            # change and only the castle flag does. A capture is the opposite:
            # the tile was already a castle and changed hands. Testing for an
            # ownership change on both counted zero builds against v16, which
            # builds one castle a game.
            built[p] += int((fresh & st.own[p]).sum())
            captured[p] += int((prev_castles & st.own[p] & ~prev_own[p]).sum())
            if first_castle[p] is None and bool((st.castles & st.own[p]).any()):
                first_castle[p] = turn
        prev_castles = st.castles.copy()
        prev_own = [st.own[0].copy(), st.own[1].copy()]

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
        "built": built, "captured": captured,
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


def _jobs(spec_a: str, spec_b: str, games: int, seed0: int, max_turns: int,
          time_limit: float, record: bool, maps: str | None,
          dmin: int | None = None, dmax: int | None = None) -> list[dict]:
    """Colours swapped on alternate games of each seed pair. Order is load-bearing:
    `_tag` reads the seat off the index, so nothing may reorder these."""
    jobs = []
    for i in range(games):
        seed = seed0 + i // 2
        swapped = i % 2 == 1
        s0, s1 = (spec_b, spec_a) if swapped else (spec_a, spec_b)
        jobs.append(dict(spec0=s0, spec1=s1, seed=seed, max_turns=max_turns,
                         time_limit=time_limit, record=record, maps=maps,
                         dmin=dmin, dmax=dmax))
    return jobs


def run_match(spec_a: str, spec_b: str, games: int, seed0: int = 0, workers: int = 8,
              max_turns: int = rules.TURN_LIMIT, time_limit: float = rules.MOVE_BUDGET_S,
              record: bool = False, progress=None, maps: str | None = None,
              dmin: int | None = None, dmax: int | None = None) -> list[dict]:
    """Play `games` games, colours swapped on alternate games of each seed pair."""
    jobs = _jobs(spec_a, spec_b, games, seed0, max_turns, time_limit, record, maps,
                 dmin, dmax)

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
    """Annotate each game with the outcome from A's point of view.

    The seat comes from the job order, not from comparing spec strings: when both
    sides are the same spec (self-play, or A/B against an identical config) the
    string test always says seat 0, the colour swap silently stops happening, and
    the result is pure seat bias rather than a score. That showed up as `ours vs
    ours` scoring 0.333 on a real-board pool, which is impossible by symmetry.
    """
    for i, r in enumerate(results):
        a_seat = i % 2
        r["a_seat"] = a_seat
        if r["winner"] < 0:
            r["a_result"] = "draw"
        else:
            r["a_result"] = "win" if r["winner"] == a_seat else "loss"
    return results


def run_many(matches: list[tuple[str, str, int, int]], workers: int = 8,
             max_turns: int = rules.TURN_LIMIT, time_limit: float = rules.MOVE_BUDGET_S,
             maps: str | None = None) -> list[list[dict]]:
    """Play several matches in ONE pool. `matches` is [(spec_a, spec_b, games, seed0)];
    returns one tagged result list per match, in order.

    run_match opens a pool per call, so a caller with many small matches (scoring
    one config against a mixture of five opponents) never has more than one
    match's games in flight and leaves most of a 60-core box idle.
    """
    batches = [_jobs(a, b, g, s, max_turns, time_limit, False, maps)
               for a, b, g, s in matches]
    flat = [j for b in batches for j in b]
    if workers <= 1:
        done = [_worker(j) for j in flat]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            done = list(pool.map(_worker, flat, chunksize=1))

    out, k = [], 0
    for (a, b, _, _), batch in zip(matches, batches):
        out.append(_tag(done[k:k + len(batch)], a, b))
        k += len(batch)
    return out


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
    ap.add_argument("--maps", default=None,
                    help="play on a real-board pool from `analysis.official maps`")
    ap.add_argument("--dmin", type=int, default=None,
                    help="generals-distance band, e.g. --dmin 13 --dmax 15. Unset "
                         "is the competition generator exactly, so every result "
                         "measured before this flag existed is reproducible. Set "
                         "it to find where a behaviour turns on: `bld` is 0.01 "
                         "across stage 3 (11-17) and 0.5 four iterations into "
                         "stage 4 (17-24), and nothing could locate the knee")
    ap.add_argument("--dmax", type=int, default=None)
    ap.add_argument("--elo1", type=float, default=12.0, help="SPRT alternative hypothesis")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()

    def progress(done, total):
        if not args.quiet:
            print(f"\r  {done}/{total} games", end="", flush=True)

    results = run_match(args.a, args.b, args.games, args.seed0, args.workers,
                        args.max_turns, args.time_limit_ms / 1000.0,
                        args.replays, progress, args.maps, args.dmin, args.dmax)
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
    # A's mode histogram. Without it an override that never fires and one that
    # fires and changes nothing produce the same 0.500, and they want opposite
    # fixes -- loosen the trigger, or drop the override.
    hist: dict[str, int] = {}
    for r in results:
        for k, v in r["modes"][r["a_seat"]].items():
            hist[k] = hist.get(k, 0) + v
    if hist:
        tot = sum(hist.values())
        print("  A modes " + " ".join(
            f"{k} {v} ({100 * v / tot:.2f}%)"
            for k, v in sorted(hist.items(), key=lambda kv: -kv[1])))
    # Castles per game per side, and when the first one goes down. With --dmin/
    # --dmax this is the castle breakpoint measured directly: sweep the band and
    # read where the rate turns on, instead of inferring it from two stages of a
    # training log.
    ca = float(np.mean([r["castles"][r["a_seat"]] for r in results]))
    cb = float(np.mean([r["castles"][1 - r["a_seat"]] for r in results]))
    firsts = [r["first_castle"][r["a_seat"]] for r in results]
    firsts = [t for t in firsts if t]
    when = f", A first builds turn {np.mean(firsts):.0f} in {len(firsts)}/{len(results)} games" if firsts else ""
    print(f"  castles/game  A {ca:.2f}  B {cb:.2f}{when}")
    ba = float(np.mean([r["built"][r["a_seat"]] for r in results]))
    bb = float(np.mean([r["built"][1 - r["a_seat"]] for r in results]))
    xa = float(np.mean([r["captured"][r["a_seat"]] for r in results]))
    xb = float(np.mean([r["captured"][1 - r["a_seat"]] for r in results]))
    print(f"  built/game    A {ba:.2f}  B {bb:.2f}     captured/game  A {xa:.2f}  B {xb:.2f}")
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
