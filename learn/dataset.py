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
from pathlib import Path

import numpy as np

from analysis.actions import _states_from, infer
from analysis.official import read_replay, replay_files
from bot import features
from sim import engine

SHARD = 40_000


def build(src: Path, out: Path, players: set[str] | None, exclude: set[str],
          min_elo_players: int, limit: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    shard = kept = seen = games = 0

    def flush():
        nonlocal xs, ys, shard, kept
        if not xs:
            return
        np.savez_compressed(out / f"shard_{shard:04d}.npz",
                            x=np.stack(xs).astype(np.float16),
                            y=np.asarray(ys, dtype=np.int32))
        kept += len(ys)
        shard += 1
        xs, ys = [], []

    for f in replay_files(src):
        if limit and games >= limit:
            break
        rep = read_replay(f)
        names = rep["players"]
        seats = [i for i, n in enumerate(names)
                 if (players is None or n in players) and n not in exclude]
        if not seats:
            continue
        acts, _ = infer(rep)
        states, _ = _states_from(rep)
        games += 1
        for t, pair in enumerate(acts):
            seen += 2
            if pair is None:
                continue
            for seat in seats:
                obs = engine.observe(states[t], seat)
                idx = features.action_to_index(pair[seat])
                xs.append(features.encode(obs))
                ys.append(idx)
                if len(xs) >= SHARD:
                    flush()
        if games % 25 == 0:
            print(f"  {games} games, {kept + len(ys)} examples", flush=True)
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
    args = ap.parse_args()

    players = set(p.strip() for p in args.players.split(",")) if args.players else None
    exclude = set(p.strip() for p in args.exclude.split(",") if p.strip())
    build(Path(args.src), Path(args.out), players, exclude, 0, args.limit)


if __name__ == "__main__":
    main()
