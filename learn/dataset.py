"""Replays -> (observation, action) shards for behaviour cloning.

Only ticks whose actions were *confirmed* by replaying them through the exact
simulator are kept, so no label is invented. Observations are the fogged view
that player actually had.

By default it clones the strong players only — cloning ourselves would just
reproduce our own blind spots, and the point of this policy is to be a sparring
partner that punishes what our own bots never punish.

    python -m learn.dataset /local/data/vng205/field --out /local/data/vng205/bc
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from analysis.actions import _states_from, infer
from analysis.official import read_replay, replay_files
from bot import features
from sim import engine

SHARD = 40_000


def _one(job) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Reconstruct one replay. Runs in a worker; action inference is the cost."""
    path, players, exclude = job
    rep = read_replay(path)
    names = rep["players"]
    seats = [i for i, n in enumerate(names)
             if (players is None or n in players) and n not in exclude]
    if not seats:
        return None
    acts, _ = infer(rep)
    states, _ = _states_from(rep)
    xs, ys = [], []
    seen = 0
    for t, pair in enumerate(acts):
        seen += len(seats)
        if pair is None:
            continue
        for seat in seats:
            obs = engine.observe(states[t], seat)
            xs.append(features.encode(obs))
            ys.append(features.action_to_index(pair[seat]))
    if not xs:
        return None
    return np.stack(xs).astype(np.float16), np.asarray(ys, dtype=np.int32), seen


def build(src: Path, out: Path, players: set[str] | None, exclude: set[str],
          workers: int, limit: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    files = replay_files(src)
    if limit:
        files = files[:limit]
    jobs = [(f, players, exclude) for f in files]

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    buffered = shard = kept = seen = games = 0

    def flush():
        nonlocal xs, ys, shard, kept, buffered
        if not xs:
            return
        np.savez_compressed(out / f"shard_{shard:04d}.npz",
                            x=np.concatenate(xs), y=np.concatenate(ys))
        kept += buffered
        shard += 1
        xs, ys, buffered = [], [], 0

    runner = (map(_one, jobs) if workers <= 1 else
              ProcessPoolExecutor(max_workers=workers).map(_one, jobs, chunksize=4))
    for res in runner:
        games += 1
        if res is not None:
            x, y, s_ = res
            xs.append(x)
            ys.append(y)
            buffered += len(y)
            seen += s_
            if buffered >= SHARD:
                flush()
        if games % 100 == 0:
            print(f"  {games}/{len(jobs)} replays, {kept + buffered} examples", flush=True)
    flush()

    meta = {"examples": kept, "games": games, "ticks_seen": seen,
            "channels": features.C, "pad": features.PAD,
            "n_actions": features.N_ACTIONS}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{kept} examples from {games} games -> {out}")
    print(f"  recovery {100.0 * kept / max(seen, 1):.1f}% of player-ticks")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", default="/local/data/vng205/field")
    ap.add_argument("--out", default="/local/data/vng205/bc")
    ap.add_argument("--players", default=None,
                    help="comma-separated; default = everyone except ourselves")
    ap.add_argument("--exclude", default="H.V.Nguyen,Expander (baseline),Hunter (baseline)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    players = set(p.strip() for p in args.players.split(",")) if args.players else None
    exclude = set(p.strip() for p in args.exclude.split(",") if p.strip())
    build(Path(args.src), Path(args.out), players, exclude, args.workers, args.limit)


if __name__ == "__main__":
    main()
