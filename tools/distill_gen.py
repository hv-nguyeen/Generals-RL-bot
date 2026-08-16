"""Self-play of a TEACHER policy -> (observation, teacher-move) shards.

Distillation, not RL. We already have a measured-stronger teacher (the +17
ensemble). Behaviour-cloning a single net to imitate its moves is SUPERVISED —
no critic, no PPO, so the cold-critic collapse that killed every self-play run
here cannot happen. The output is a single trained net that plays like the
ensemble (stronger than the champion) at one net's inference cost.

Shards match `learn/dataset.py` exactly (`x` float16 (N,C,21,21), `y` int32
action index, plus meta.json), so `learn/train.py` distils them into ANY arch
(`--layers/--channels/--residual`) with no other change.

    python -m tools.distill_gen --teacher 'shipens:champ.npz+topo.npz@logit' \
        --opponents 'self,greedy,hunter,expander,ours:configs/v16.json' \
        --games 4000 --out /local/data/vng205/distill-bc --workers 32

`self` means the teacher on both seats (every move is a teacher label, 2x data,
on the teacher's own state distribution). Named opponents add off-distribution
states so the student is robust where the teacher rarely goes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from arena import agents
from bot import features
from bot.memory import TemporalMemory
from sim import engine, mapgen

SHARD = 200_000                        # examples per shard, like learn/dataset.py
_CTX: dict = {}


def _one(job):
    """One game. Record every move made by a TEACHER-controlled seat."""
    seed, opp = job
    grid = mapgen.generate(seed)
    h, w = grid.shape
    st = engine.from_grid(grid)
    teacher = _CTX["teacher"]
    a0 = agents.make(teacher, 0, h, w, seed)
    # seat 1 is the teacher too when opp == "self", else the named opponent
    teach1 = opp == "self"
    a1 = agents.make(teacher if teach1 else opp, 1, h, w, seed)
    mem = [TemporalMemory(h, w), TemporalMemory(h, w)]
    xs, ys = [], []
    for _ in range(_CTX["max_turns"]):
        obs0 = engine.observe(st, 0)
        obs1 = engine.observe(st, 1)
        try:
            act0 = a0.act(obs0)
        except Exception:                       # noqa: BLE001 - one bad game ≠ poison
            break
        try:
            act1 = a1.act(obs1)
        except Exception:                       # noqa: BLE001
            act1 = (1, 0, 0, 0, 0)              # PASS
        # Label seat 0 (always teacher); seat 1 only when it is the teacher.
        mem[0].update(obs0)
        xs.append(features.encode(obs0, mem[0]).astype(np.float16))
        ys.append(features.action_to_index(act0))
        if teach1:
            mem[1].update(obs1)
            xs.append(features.encode(obs1, mem[1]).astype(np.float16))
            ys.append(features.action_to_index(act1))
        if engine.step(st, act0, act1):
            break
    if not xs:
        return None
    return np.stack(xs), np.asarray(ys, dtype=np.int32)


def _init(teacher, max_turns):
    _CTX.update(teacher=teacher, max_turns=max_turns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True, help="agent spec, e.g. shipens:a.npz+b.npz@logit")
    ap.add_argument("--opponents", default="self",
                    help="comma list; 'self' = teacher vs teacher (labels both seats)")
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-turns", type=int, default=1200)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    opps = [o.strip() for o in args.opponents.split(",") if o.strip()]
    jobs = [(args.seed0 + i, opps[i % len(opps)]) for i in range(args.games)]

    args.out.mkdir(parents=True, exist_ok=True)
    for s in args.out.glob("shard_*.npz"):
        s.unlink()

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    shard = kept = buffered = games = 0

    def flush():
        nonlocal xs, ys, shard, kept, buffered
        if not xs:
            return
        np.savez_compressed(args.out / f"shard_{shard:04d}.npz",
                            x=np.concatenate(xs), y=np.concatenate(ys))
        kept += buffered
        shard += 1
        xs, ys, buffered = [], [], 0

    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                               initargs=(args.teacher, args.max_turns))
    for res in pool.map(_one, jobs, chunksize=2):
        games += 1
        if res is not None:
            x, y = res
            xs.append(x); ys.append(y); buffered += len(y)
            if buffered >= SHARD:
                flush()
        if games % 200 == 0:
            print(f"  {games}/{len(jobs)} games, {kept + buffered} examples", flush=True)
    flush()
    pool.shutdown()

    (args.out / "meta.json").write_text(json.dumps(
        {"examples": kept, "games": games, "channels": features.C,
         "pad": features.PAD, "n_actions": features.N_ACTIONS,
         "teacher": args.teacher, "opponents": opps}, indent=2))
    print(f"\n{kept} examples from {games} games -> {args.out}")


if __name__ == "__main__":
    main()
