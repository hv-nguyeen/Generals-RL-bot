"""Position -> did this player go on to win. Training data for a value model.

Unlike the behaviour-cloning set this needs no action inference: the label is
the game's outcome, which the match list already records. That makes it cheap to
build and lets us use every tick of every replay.

The point is to get an evaluation function grounded in games where *real*
opponents did the punishing. Our local opponents never punish over-commitment,
so they rank a bot that overextends too highly; a model fitted to field outcomes
has seen those positions lose.

Observations are the fogged view the player actually had, so the model can be
applied to our own bot mid-game.

    python -m learn.valuedata /local/data/vng205/field --out /local/data/vng205/val
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from analysis.official import read_replay, replay_files, to_states
from bot import features
from sim import engine

SHARD = 60_000


def _winner(rep: dict, meta: dict | None) -> int:
    """Seat index of the winner, or -1 for a draw/unknown."""
    if meta and meta.get("winner") in ("A", "B"):
        name = meta["a_name"] if meta["winner"] == "A" else meta["b_name"]
        if name in rep["players"]:
            return rep["players"].index(name)
    last = np.asarray(rep["ticks"][-1]["owners"], dtype=np.int32)
    a, b = int((last == 0).sum()), int((last == 1).sum())
    if a == 0 and b > 0:
        return 1
    if b == 0 and a > 0:
        return 0
    return -1


def _one(job):
    path, meta, stride, drop_last = job
    rep = read_replay(path)
    win = _winner(rep, meta)
    if win < 0:
        return None                      # draws teach the model nothing useful
    states = [s for _, s in to_states(rep)]
    n = len(states)
    if n < 40:
        return None
    # The last few ticks are trivially decided and would dominate the loss with
    # positions no strategy question ever hinges on.
    end = max(1, n - drop_last)
    xs, ys, ts = [], [], []
    for t in range(0, end, stride):
        for seat in (0, 1):
            xs.append(features.encode(engine.observe(states[t], seat)))
            ys.append(1.0 if seat == win else 0.0)
            ts.append(t)
    if not xs:
        return None
    return (np.stack(xs).astype(np.float16),
            np.asarray(ys, dtype=np.float32),
            np.asarray(ts, dtype=np.int32))


def build(src: Path, out: Path, stride: int, drop_last: int,
          workers: int, limit: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict] = {}
    for m in src.glob("matches*.json"):
        for row in json.loads(m.read_text()):
            meta[str(row["id"])] = row

    files = replay_files(src)
    if limit:
        files = files[:limit]
    jobs = []
    for f in files:
        mid = f.name.split(".")[0]
        jobs.append((f, meta.get(mid), stride, drop_last))

    xs, ys, ts = [], [], []
    buffered = shard = kept = games = 0

    def flush():
        nonlocal xs, ys, ts, shard, kept, buffered
        if not xs:
            return
        np.savez_compressed(out / f"shard_{shard:04d}.npz",
                            x=np.concatenate(xs), y=np.concatenate(ys),
                            t=np.concatenate(ts))
        kept += buffered
        shard += 1
        xs, ys, ts, buffered = [], [], [], 0

    runner = (map(_one, jobs) if workers <= 1 else
              ProcessPoolExecutor(max_workers=workers).map(_one, jobs, chunksize=4))
    for res in runner:
        games += 1
        if res is not None:
            x, y, t = res
            xs.append(x); ys.append(y); ts.append(t)
            buffered += len(y)
            if buffered >= SHARD:
                flush()
        if games % 200 == 0:
            print(f"  {games}/{len(jobs)} replays, {kept + buffered} positions", flush=True)
    flush()

    (out / "meta.json").write_text(json.dumps(
        {"positions": kept, "games": games, "stride": stride,
         "drop_last": drop_last, "channels": features.C, "pad": features.PAD}, indent=2))
    print(f"\n{kept} positions from {games} replays -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", default="/local/data/vng205/field")
    ap.add_argument("--out", default="/local/data/vng205/val")
    ap.add_argument("--stride", type=int, default=4, help="keep every Nth tick")
    ap.add_argument("--drop-last", type=int, default=25,
                    help="ignore the final N ticks, which are already decided")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    build(Path(args.src), Path(args.out), args.stride, args.drop_last,
          args.workers, args.limit)


if __name__ == "__main__":
    main()
