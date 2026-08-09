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
from bot.memory import TemporalMemory
from sim import engine
from tools.manifest import file_set

SHARD = 60_000


def prepare_output(out: Path) -> int:
    """Clear prior shards/meta before a full rebuild, preserving other files."""
    out.mkdir(parents=True, exist_ok=True)
    stale = sorted(out.glob("shard_*.npz"))
    for path in stale:
        path.unlink()
    meta = out / "meta.json"
    if meta.exists():
        meta.unlink()
    if stale:
        print(f"removed {len(stale)} stale shards from {out}")
    return len(stale)


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
    states = [s for _, s in to_states(rep)]
    n = len(states)
    if n < 40:
        return None
    # The last few ticks are trivially decided and would dominate the loss with
    # positions no strategy question ever hinges on.
    end = max(1, n - drop_last)
    xs, ys, ts, games, seats = [], [], [], [], []
    memory = [TemporalMemory(*states[0].armies.shape) for _ in range(2)]
    game_id = int(rep.get("id") or Path(path).name.split(".")[0])
    for t in range(end):
        for seat in (0, 1):
            obs = engine.observe(states[t], seat)
            memory[seat].update(obs)
            if t % stride:
                continue
            xs.append(features.encode(obs, memory[seat]))
            ys.append(0.0 if win < 0 else (1.0 if seat == win else -1.0))
            ts.append(t)
            games.append(game_id)
            seats.append(seat)
    if not xs:
        return None
    return (np.stack(xs).astype(np.float16),
            np.asarray(ys, dtype=np.float32),
            np.asarray(ts, dtype=np.int32),
            np.asarray(games, dtype=np.int64),
            np.asarray(seats, dtype=np.int8))


def _sources(src: Path) -> list[Path]:
    """A harvest dir, or a parent of several.

    `analysis.official` stores one player per directory, each with its own
    `replays/` and match list. A field harvest of ten players is therefore ten
    of those under one root, and pointing this at the root should mean all of
    them rather than nothing.
    """
    if (src / "replays").is_dir():
        return [src]
    return sorted(d for d in src.iterdir() if (d / "replays").is_dir())


def build(src: Path, out: Path, stride: int, drop_last: int,
          workers: int, limit: int) -> None:
    dirs = _sources(src)
    if not dirs:
        raise SystemExit(f"no replays/ under {src}")

    meta: dict[str, dict] = {}
    files = []
    for d in dirs:
        for m in d.glob("matches*.json"):
            for row in json.loads(m.read_text()):
                meta[str(row["id"])] = row
        files.extend(replay_files(d))
    if len(dirs) > 1:
        print(f"{len(dirs)} sources, {len(files)} replays")
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no replay files under {src}")
    jobs = []
    for f in files:
        mid = f.name.split(".")[0]
        jobs.append((f, meta.get(mid), stride, drop_last))
    prepare_output(out)

    xs, ys, ts, game_ids, seats = [], [], [], [], []
    buffered = shard = kept = games = 0

    def flush():
        nonlocal xs, ys, ts, game_ids, seats, shard, kept, buffered
        if not xs:
            return
        np.savez_compressed(out / f"shard_{shard:04d}.npz",
                            x=np.concatenate(xs), y=np.concatenate(ys),
                            t=np.concatenate(ts), game=np.concatenate(game_ids),
                            seat=np.concatenate(seats))
        kept += buffered
        shard += 1
        xs, ys, ts, game_ids, seats, buffered = [], [], [], [], [], 0

    runner = (map(_one, jobs) if workers <= 1 else
              ProcessPoolExecutor(max_workers=workers).map(_one, jobs, chunksize=4))
    for res in runner:
        games += 1
        if res is not None:
            x, y, t, game, seat = res
            xs.append(x); ys.append(y); ts.append(t)
            game_ids.append(game); seats.append(seat)
            buffered += len(y)
            if buffered >= SHARD:
                flush()
        if games % 200 == 0:
            print(f"  {games}/{len(jobs)} replays, {kept + buffered} positions", flush=True)
    flush()

    (out / "meta.json").write_text(json.dumps(
        {"positions": kept, "games": games, "stride": stride,
         "drop_last": drop_last, "channels": features.C, "pad": features.PAD,
         "source_replays": file_set(files, src)}, indent=2))
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
