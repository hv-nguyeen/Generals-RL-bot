"""Autonomous expert-iteration loop: search -> distil -> gate -> promote, repeat.

No supervision. Each round generates search-improved labels from the current
BASE net, distils a fast net, and paired-gates it vs the base. Promotion is
programmatic: accept only if the paired lower-bound Elo beats the base by
`--margin` with ZERO faults inside the move budget; then that net becomes the
next round's base (search over a stronger base is stronger -> gains compound).
Stops after `--rounds`, or after `--patience` consecutive non-improving rounds.
Every checkpoint, gate, and decision is logged; `best.npz` is always the best
promoted net.

    python -m tools.expert_iter --base champion.npz --rounds 6 \
        --games 800 --topk 6 --horizon 10 --layers 8 --channels 64 \
        --ensemble champion.npz,topo-night-best.npz \
        --out-dir /local/data/vng205/ei --workers 32

`--ensemble` (optional) also reports, each round, the net as an added member vs
that shipped ensemble (elo_lo>0 there = past the current shipped bot).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ELO = re.compile(r"elo\s+([+-]?[0-9.]+)\s+\[([+-]?[0-9.]+),\s*([+-]?[0-9.]+)\]")
FAULTS = re.compile(r"faults\s+([0-9]+)")
MAXMS = re.compile(r"slowest move in run\s+([0-9.]+)\s*ms")


def _run(cmd: list[str], log) -> str:
    log.write(f"\n$ {' '.join(cmd)}\n"); log.flush()
    p = subprocess.run(cmd, capture_output=True, text=True)
    log.write(p.stdout); log.write(p.stderr); log.flush()
    if p.returncode:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd[:4])}...")
    return p.stdout + p.stderr


def _gate(a: str, b: str, games: int, workers: int, log) -> dict:
    out = _run([PY, "-m", "arena.runner", "--a", a, "--b", b,
                "--games", str(games), "--workers", str(workers),
                "--time-limit-ms", "150", "--quiet"], log)
    elo = ELO.search(out); fa = FAULTS.search(out); mx = MAXMS.search(out)
    return {"elo": float(elo.group(1)) if elo else None,
            "lo": float(elo.group(2)) if elo else None,
            "hi": float(elo.group(3)) if elo else None,
            "faults": int(fa.group(1)) if fa else None,
            "max_ms": float(mx.group(1)) if mx else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2,
                    help="non-improving rounds before ESCALATING search depth")
    ap.add_argument("--resume", action="store_true",
                    help="continue from out-dir/best.npz + history.json if present")
    ap.add_argument("--max-horizon", type=int, default=24)
    ap.add_argument("--max-games", type=int, default=3000)
    ap.add_argument("--keep-shards", action="store_true",
                    help="do NOT delete each round's search shards after distil")
    ap.add_argument("--games", type=int, default=800, help="search self-play games/round")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--gate-games", type=int, default=2000)
    ap.add_argument("--margin", type=float, default=0.0, help="required elo_lo over base")
    ap.add_argument("--max-ms", type=float, default=150.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--ensemble", default="", help="comma paths for a member-add report")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(args.out_dir / "expert_iter.log", "a", buffering=1)
    hist_path = args.out_dir / "history.json"
    best = args.out_dir / "best.npz"

    def note(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    # Resume: pick up from the best net + history so a week-long run survives a
    # crash / SSH drop / relaunch without restarting from the champion.
    history = json.loads(hist_path.read_text()) if (args.resume and hist_path.is_file()) else []
    if args.resume and best.is_file() and history:
        base = str(best)
        note(f"=== RESUME {time.strftime('%F %T')}: {len(history)} rounds done, base={best} ===")
    else:
        base = args.base
        shutil.copy(base, best)
        note(f"=== expert-iteration start {time.strftime('%F %T')} base={base} ===")

    horizon, games, stale = args.horizon, args.games, 0
    r = len(history)
    while r < args.rounds:
        r += 1
        rd = args.out_dir / f"round_{r:02d}"; rd.mkdir(exist_ok=True)
        data, net = rd / "search-bc", rd / "net.npz"
        try:
            note(f"\n--- round {r} {time.strftime('%T')} : search-gen from {Path(base).name} "
                 f"(topk {args.topk} horizon {horizon} games {games}) ---")
            _run([PY, "-m", "tools.search_gen", "--net", base, "--games", str(games),
                  "--topk", str(args.topk), "--horizon", str(horizon),
                  "--out", str(data), "--workers", str(args.workers)], logf)
            note(f"round {r}: distil -> {net.name}")
            _run([PY, "-m", "learn.train", "--data", str(data), "--out", str(net),
                  "--layers", str(args.layers), "--channels", str(args.channels),
                  "--epochs", str(args.epochs), "--augment"], logf)
            if not args.keep_shards:
                shutil.rmtree(data, ignore_errors=True)      # free disk over a long run
            g = _gate(f"ship:{net}", f"ship:{base}", args.gate_games, args.workers, logf)
        except Exception as e:                               # noqa: BLE001
            note(f"round {r}: ERROR ({e!r}); skipping, base unchanged")
            continue
        improved = (g["lo"] is not None and g["lo"] > args.margin
                    and g["faults"] == 0
                    and (g["max_ms"] is None or g["max_ms"] < args.max_ms))
        ens = None
        if args.ensemble:
            try:
                mem = "+".join(args.ensemble.split(",") + [str(net)])
                base_ens = "+".join(args.ensemble.split(","))
                ens = _gate(f"shipens:{mem}@logit", f"shipens:{base_ens}@logit",
                            args.gate_games, args.workers, logf)
            except Exception as e:                           # noqa: BLE001
                note(f"round {r}: member-gate error ({e!r})")
        history.append({"round": r, "net": str(net), "topk": args.topk,
                        "horizon": horizon, "games": games,
                        "vs_base": g, "as_member": ens, "promoted": improved})
        hist_path.write_text(json.dumps(history, indent=2))
        note(f"round {r}: vs-base elo {g['elo']} [{g['lo']},{g['hi']}] faults {g['faults']} "
             f"max_ms {g['max_ms']} -> {'PROMOTE' if improved else 'reject'}")
        if ens:
            note(f"round {r}: as member vs shipped ensemble elo {ens['elo']} "
                 f"[{ens['lo']},{ens['hi']}] faults {ens['faults']}")
        if improved:
            base = str(net); shutil.copy(net, best); stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                # Plateaued at this search depth: ESCALATE (deeper/more search
                # finds moves the shallow search missed) instead of stopping, so
                # a week-long run keeps making real attempts.
                if horizon < args.max_horizon or games < args.max_games:
                    horizon = min(int(horizon * 1.5) + 1, args.max_horizon)
                    games = min(int(games * 1.5), args.max_games)
                    stale = 0
                    note(f"plateau -> ESCALATE search to horizon {horizon}, games {games}")
                else:
                    note(f"stop: plateaued at max search depth "
                         f"(horizon {horizon}, games {games})"); break

    note(f"\n=== done round {r}. best net -> {best} ===")


if __name__ == "__main__":
    main()
