"""One command, one paste-able report.

Runs the whole diagnostic loop — gauntlet, loss causes, expansion fingerprint,
mode profile, timing — and prints plain text small enough to paste into a chat.
Everything here was previously a sequence of ad-hoc commands; the mode profile in
particular is what found that GATHER was eating 72% of the midgame.

    python -m tools.diagnose --workers 32
    python -m tools.diagnose --config runs/tune/best.json --vs configs/v2.json
    python -m tools.diagnose --quick            # fewer games, for a fast look

Reading it:
  * a mode with a big `share` and a low `cap/turn` is where the Elo is hiding;
  * `land@200` under ~105 means we are behind the top of the leaderboard on
    macro, whatever the win rate against our own baselines says;
  * any `faults` at all is a bug that will cost real games.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from analysis import expansion, stats as stats_mod
from arena import rating
from arena.runner import run_match, tally
from bot.config import Config

BAR = "=" * 78


def _git_rev() -> str:
    """Identify the build. Released tarballs carry a VERSION file because
    `git archive` strips .git, and a pasted report has to say what produced it."""
    stamp = Path(__file__).resolve().parent.parent / "VERSION"
    if stamp.exists():
        return stamp.read_text().strip()
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "?"
    except Exception:                                # noqa: BLE001
        return "?"


def _config_diff(cfg: Config) -> list[str]:
    """Only the knobs that differ from the checked-in defaults."""
    base = Config()
    bn, bv = base.flatten()
    cn, cv = cfg.flatten()
    lookup = dict(zip(bn, bv))
    return [f"{n}={v:g}" for n, v in zip(cn, cv)
            if abs(lookup.get(n, v) - v) > 1e-9]


def section(title: str) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}")


def run(args) -> None:
    spec = f"ours:{args.config}" if args.config else "ours"
    cfg = Config.load(args.config) if args.config else Config()
    opponents = [o for o in args.opponents.split(",") if o]
    started = time.time()

    print(BAR)
    print(f"DIAGNOSTIC  {spec}   git {_git_rev()}   {args.games} games/opponent")
    diff = _config_diff(cfg)
    print("config: defaults" if not diff else "config: " + " ".join(diff))

    # ---- gauntlet --------------------------------------------------------
    section("GAUNTLET")
    print(f"{'opponent':<12}{'W':>5}{'D':>4}{'L':>5}{'score':>8}{'elo':>9}"
          f"{'  95% CI':<18}{'causes'}")
    all_results = {}
    for opp in opponents:
        res = run_match(spec, opp, args.games, args.seed0, args.workers,
                        args.max_turns)
        all_results[opp] = res
        w, d, l = tally(res)
        s = rating.summary(w, d, l)
        agg = stats_mod.aggregate(res)
        causes = {k: v for k, v in sorted(agg["causes"].items(), key=lambda x: -x[1])
                  if k != "win"}
        print(f"{opp:<12}{w:>5}{d:>4}{l:>5}{s['score']:>8.3f}{s['elo']:>+9.1f}"
              f"  [{s['elo_lo']:+.0f},{s['elo_hi']:+.0f}]".ljust(18)
              + ("  " + json.dumps(causes) if causes else "  clean"))

    # ---- A/B -------------------------------------------------------------
    if args.vs:
        section(f"A/B   {spec}   vs   ours:{args.vs}")
        res = run_match(spec, f"ours:{args.vs}", args.ab_games, args.seed0 + 5000,
                        args.workers, args.max_turns)
        w, d, l = tally(res)
        s = rating.summary(w, d, l)
        t = rating.sprt(w, d, l, 0.0, args.elo1)
        print(f"  {w}W {d}D {l}L   score {s['score']:.3f}   "
              f"elo {s['elo']:+.1f} [{s['elo_lo']:+.0f},{s['elo_hi']:+.0f}]")
        print(f"  SPRT(0,{args.elo1:g}): llr {t['llr']:+.2f} "
              f"bounds [{t['lower']:.2f},{t['upper']:.2f}]  ->  {t['verdict'].upper()}")

    # ---- expansion -------------------------------------------------------
    section(f"EXPANSION  (top of the leaderboard reaches ~105-110 land by turn 200)")
    print(f"{'who':<20}{'1st':>6}{'L@50':>7}{'L@100':>7}{'L@200':>7}"
          f"{'cap/turn':>10}{'idle':>7}{'peak':>7}")
    for opp in opponents:
        rows_us, rows_them = [], []
        for seed in range(args.fp_games):
            from arena import agents as agents_mod
            from sim import engine, mapgen
            grid = mapgen.generate(args.seed0 + seed)
            st = engine.from_grid(grid)
            pl = [agents_mod.make(spec, 0, *grid.shape),
                  agents_mod.make(opp, 1, *grid.shape)]
            frames = [(0, st.copy())]
            for t in range(args.horizon):
                acts = [pl[p].act(engine.observe(st, p)) for p in (0, 1)]
                done = engine.step(st, acts[0], acts[1])
                frames.append((t + 1, st.copy()))
                if done:
                    break
            rows_us.append(expansion.fingerprint(iter(frames), 0, args.horizon))
            rows_them.append(expansion.fingerprint(iter(frames), 1, args.horizon))
        for name, rows in ((f"ours (v {opp})", rows_us), (opp, rows_them)):
            m = lambda k: expansion._mean(rows, k)          # noqa: E731
            print(f"{name[:19]:<20}{m('first_expand'):>6.0f}{m('land_50'):>7.1f}"
                  f"{m('land_100'):>7.1f}{m('land_200'):>7.1f}"
                  f"{m('capture_rate'):>10.2f}{m('idle_turns'):>7.0f}"
                  f"{m('peak_stack'):>7.1f}")

    # ---- mode profile ----------------------------------------------------
    section(f"WHERE THE TURNS GO  (turns {args.lo}-{args.hi}, vs {opponents[0]})")
    print(f"{'mode':<14}{'turns':>8}{'share':>8}{'captures':>10}{'cap/turn':>10}")
    prof = expansion.mode_profile(spec, opponents[0], args.mode_games,
                                  args.lo, args.hi, args.seed0)
    for r in prof:
        print(f"{r['mode']:<14}{r['turns']:>8}{r['share']*100:>7.0f}%"
              f"{r['captures']:>10}{r['capture_rate']:>10.2f}")
    if prof:
        overall = sum(r["captures"] for r in prof) / max(sum(r["turns"] for r in prof), 1)
        print(f"{'OVERALL':<14}{'':>8}{'':>8}{'':>10}{overall:>10.2f}")
        worst = max(prof, key=lambda r: r["share"] * (1 - r["capture_rate"]))
        if worst["share"] > 0.25 and worst["capture_rate"] < 0.3:
            print(f"\n  ** {worst['mode']} takes {worst['share']*100:.0f}% of these turns "
                  f"and captures on {worst['capture_rate']:.0%} of them — look here first.")

    # ---- timing ----------------------------------------------------------
    section("TIMING AND SAFETY")
    for opp, res in all_results.items():
        ms = [r["mean_ms"][r["a_seat"]] for r in res]
        mx = max(max(r["max_ms"]) for r in res)
        faults = sum(r["faults"][r["a_seat"]] for r in res)
        flag = "" if faults == 0 else "   <-- FAULTS, FIX BEFORE SUBMITTING"
        print(f"  vs {opp:<12} mean {np.mean(ms):.2f} ms   slowest {mx:.1f} ms   "
              f"faults {faults}{flag}")
    print(f"\n(budget is 150 ms/move, 10 s for the first; the slowest move is "
          f"almost always turn 0)")
    print(f"\nwall {time.time() - started:.0f}s")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "spec": spec, "git": _git_rev(), "config_diff": diff,
            "gauntlet": {o: dict(zip(("w", "d", "l"), tally(r)))
                         for o, r in all_results.items()},
            "mode_profile": prof,
        }, indent=2))
        print(f"machine-readable copy -> {args.json}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="config json to diagnose")
    ap.add_argument("--vs", default=None, help="second config for an A/B with SPRT")
    ap.add_argument("--opponents", default="hunter,greedy")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--ab-games", type=int, default=600)
    ap.add_argument("--fp-games", type=int, default=12)
    ap.add_argument("--mode-games", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--lo", type=int, default=100)
    ap.add_argument("--hi", type=int, default=200)
    ap.add_argument("--max-turns", type=int, default=1200)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--elo1", type=float, default=12.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--quick", action="store_true", help="fewer games, fast look")
    args = ap.parse_args()
    if args.quick:
        args.games, args.ab_games, args.fp_games, args.mode_games = 60, 200, 6, 4
    run(args)


if __name__ == "__main__":
    main()
