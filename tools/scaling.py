"""Does the behaviour clone want a bigger network? Train several and look.

HOW TO READ THE TABLE. One row per architecture, all trained on the same shards
against the same three held-out shards, same seed, same lr schedule, same epoch
CEILING with the same early-stopping patience. The default sizes are chosen so
that no two of the three things you might scale move together:

    4x32 -> 8x32 and 4x64 -> 8x64   depth alone
    4x32 -> 4x64 and 8x32 -> 8x64   width alone
    9x64 -> 9x64r                   residual blocks alone, same depth and width

`val top-1` is a mean over three held-out shards and `+-` is the standard error
ACROSS those shards, not over positions. Read every difference against it. The
decision rule is `spread > 2 * the largest per-size +-`, which is a measured
noise floor rather than a threshold somebody picked; the old 0.01 constant sat
two to three times BELOW the real noise and could only ever return "flat".

* spread inside the noise -> capacity is NOT the ceiling of the CLONE. Look at
  `val nll` before you believe it: NLL can fall a long way while top-1 does not
  move, because the argmax over a near-tie flips at random, and the thing PPO
  warm starts from is the distribution and not the argmax.
* spread above the noise, and CLIMBING -> capacity was binding. Take the
  smallest size on the plateau; the games/s column is the price, every night.
* climbs only WITH --augment -> the data limit is real and symmetry relieved it.
  Keep the augmentation on for every run after this, including the PPO warm
  start.
* `best epoch` at the ceiling for the big sizes means they were still improving.
  The study is then undertrained, not answered: raise --epochs and rerun.

WHAT THE BUDGET COLUMNS MEAN. ms/move is the real numpy forward pass measured
here, one core, the same code the submission runs. It is not the constraint: the
match limit is 150 ms and 20x128 costs ~6 ms, ~60 ms on a box ten times slower.

games/s is the PPO rollout rate, and it is AMDAHL, not an inverse. The forward
pass is a few percent of a rollout turn — engine.step, two observes, the
opponent's own act, legal_mask and the pool IPC are the rest. Measured: 20
games/s at 4x32 with 60 workers is 6.25 ms of wall clock per turn against a
0.26 ms forward pass, so the fixed cost is ~6.0 ms and

    games/s = workers / (turns * (FIXED_MS + ms) / 1000)

Scaling 4x32 -> 8x96 costs ~17% of a night's games, not the 5x an inverse model
predicts. Throughput is the binding constraint relative to the move budget, but
in absolute terms it is soft, and it is not a reason to refuse to test capacity.

    python -m tools.scaling --data /local/data/vng205/bc --out /local/data/vng205/scaling \\
        --sizes 4x32,8x32,4x64,8x64,9x64,9x64r --augment

Each size is a separate `python -m learn.train`, so one blown-up run does not
take the study with it. Runs on the cluster; this trains.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from bot import features, rules
from bot.obs import Obs
from bot.policy.net import Net

# Measured: 60 workers, 20 games/s, ~480 turns/game, 0.26 ms/turn in the net.
# Everything that is NOT the forward pass costs the remainder, per turn.
ROLLOUT_WORKERS, ROLLOUT_TURNS, FIXED_MS = 60, 480, 6.0


def parse_size(s: str) -> dict:
    """`8x96` or `7x64r` (residual)."""
    r = s.endswith("r")
    layers, channels = (s[:-1] if r else s).split("x")
    return {"layers": int(layers), "channels": int(channels), "residual": r}


def games_per_s(ms: float) -> float:
    """Rollout throughput at a forward pass of `ms`, Amdahl not inverse."""
    return ROLLOUT_WORKERS / (ROLLOUT_TURNS * (FIXED_MS + ms) / 1000.0)


def ms_per_move(path: Path, reps: int = 30) -> float:
    """The submission's own forward pass on a full-size board, single core."""
    h = w = features.PAD
    ty = np.full((h, w), rules.T_PLAIN, dtype=np.int8)
    ty[0, 0] = rules.T_GENERAL
    ow = np.zeros((h, w), dtype=np.int8)
    ow[0, 0] = rules.OWNER_ME
    ar = np.zeros((h, w), dtype=np.int32)
    ar[0, 0] = 9
    obs = Obs(H=h, W=w, turn=1, my_land=1, my_army=9, opp_land=1, opp_army=1,
              type_grid=ty, owner_grid=ow, army_grid=ar)
    net = Net(str(path))
    net.logits(obs)                                  # warm the BLAS handles
    t = time.perf_counter()
    for _ in range(reps):
        net.logits(obs)
    return 1000.0 * (time.perf_counter() - t) / reps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/local/data/vng205/bc")
    ap.add_argument("--out", default="/local/data/vng205/scaling")
    ap.add_argument("--sizes", default="4x32,8x32,4x64,8x64,9x64,9x64r",
                    help="comma separated LAYERSxCHANNELS, trailing r = residual")
    ap.add_argument("--epochs", type=int, default=20,
                    help="ceiling; learn.train early stops on --patience")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in args.sizes.split(","):
        arch = parse_size(spec.strip())
        ck = out / f"clone-{spec.strip()}.npz"
        cmd = [sys.executable, "-m", "learn.train", "--data", args.data,
               "--out", str(ck), "--epochs", str(args.epochs),
               "--patience", str(args.patience),
               "--batch", str(args.batch), "--lr", str(args.lr),
               "--seed", str(args.seed), "--layers", str(arch["layers"]),
               "--channels", str(arch["channels"])]
        if arch["residual"]:
            cmd.append("--residual")
        if args.augment:
            cmd.append("--augment")
        print(f"\n=== {spec}\n{' '.join(cmd)}", flush=True)
        started = time.time()
        if subprocess.run(cmd).returncode:
            print(f"  {spec} FAILED, skipping")
            continue
        meta = json.loads(ck.with_suffix(".json").read_text())
        ms = ms_per_move(ck)
        rows.append(dict(size=spec, params=meta["params"],
                         val_top1=meta["best_val_top1"], se=meta["val_block_se"],
                         val_nll=meta["best_val_nll"], best_epoch=meta["best_epoch"],
                         ms_move=ms, games_s=games_per_s(ms),
                         minutes=(time.time() - started) / 60))

    print(f"\n{'size':>8} {'params':>10} {'val top-1':>10} {'+-':>6} {'val nll':>8} "
          f"{'epoch':>6} {'ms/move':>8} {'games/s':>8} {'min':>5}")
    for r in rows:
        print(f"{r['size']:>8} {r['params']:>10} {r['val_top1']:>10.3f} "
              f"{r['se']:>6.3f} {r['val_nll']:>8.4f} "
              f"{r['best_epoch']:>3}/{args.epochs:<2} {r['ms_move']:>8.2f} "
              f"{r['games_s']:>8.1f} {r['minutes']:>5.0f}")

    if len(rows) > 1:
        spread = max(r["val_top1"] for r in rows) - min(r["val_top1"] for r in rows)
        noise = 2 * max(r["se"] for r in rows)
        print(f"\nval top-1 spread across sizes {spread:.3f}, noise floor {noise:.3f} "
              f"(2x the widest across-shard SE)")
        print("  spread is INSIDE the noise: this study did not detect a capacity "
              "effect. That is not the same as there being none — check val nll, "
              "and see the honesty note in the docstring."
              if spread <= noise else
              "  spread is outside the noise: capacity moves the clone. Take the "
              "smallest size on the plateau, not the biggest measured.")
        if any(r["best_epoch"] >= args.epochs - 1 for r in rows):
            print("  WARNING a size stopped at the epoch ceiling — it was still "
                  "improving, so this table understates it. Raise --epochs.")
        print("\nTop-1 on median human play has a ceiling the data sets, and the "
              "question is strength. Settle it in the arena:")
        best = max(rows, key=lambda r: r["val_top1"])
        print(f"  python -m arena.runner --a clone:{out}/clone-{rows[0]['size']}.npz "
              f"--b clone:{out}/clone-{best['size']}.npz --games 400 --workers 32")

    (out / "scaling.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
