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
    ap.add_argument("--patience", type=int, default=2)
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
    base = args.base
    shutil.copy(base, args.out_dir / "best.npz")
    history, stale = [], 0

    def note(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    note(f"=== expert-iteration start {time.strftime('%F %T')} base={base} ===")
    for r in range(1, args.rounds + 1):
        rd = args.out_dir / f"round_{r:02d}"; rd.mkdir(exist_ok=True)
        data, net = rd / "search-bc", rd / "net.npz"
        note(f"\n--- round {r} {time.strftime('%T')} : search-gen from {Path(base).name} ---")
        _run([PY, "-m", "tools.search_gen", "--net", base, "--games", str(args.games),
              "--topk", str(args.topk), "--horizon", str(args.horizon),
              "--out", str(data), "--workers", str(args.workers)], logf)
        note(f"round {r}: distil -> {net.name}")
        _run([PY, "-m", "learn.train", "--data", str(data), "--out", str(net),
              "--layers", str(args.layers), "--channels", str(args.channels),
              "--epochs", str(args.epochs), "--augment"], logf)
        g = _gate(f"ship:{net}", f"ship:{base}", args.gate_games, args.workers, logf)
        improved = (g["lo"] is not None and g["lo"] > args.margin
                    and g["faults"] == 0
                    and (g["max_ms"] is None or g["max_ms"] < args.max_ms))
        ens = None
        if args.ensemble:
            mem = "+".join(args.ensemble.split(",") + [str(net)])
            base_ens = "+".join(args.ensemble.split(","))
            ens = _gate(f"shipens:{mem}@logit", f"shipens:{base_ens}@logit",
                        args.gate_games, args.workers, logf)
        rec = {"round": r, "net": str(net), "vs_base": g, "as_member": ens,
               "promoted": improved}
        history.append(rec)
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
        note(f"round {r}: vs-base elo {g['elo']} [{g['lo']},{g['hi']}] faults {g['faults']} "
             f"max_ms {g['max_ms']} -> {'PROMOTE' if improved else 'reject'}")
        if ens:
            note(f"round {r}: as member#3 vs shipped ensemble elo {ens['elo']} "
                 f"[{ens['lo']},{ens['hi']}] faults {ens['faults']}")
        if improved:
            base = str(net); shutil.copy(net, args.out_dir / "best.npz"); stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                note(f"stop: {stale} non-improving rounds"); break

    note(f"\n=== done. best net -> {args.out_dir/'best.npz'} (base lineage: {Path(base).name}) ===")


if __name__ == "__main__":
    main()
