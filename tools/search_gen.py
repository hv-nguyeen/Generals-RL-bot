"""Search-based EXPERT ITERATION: generate (obs -> search-improved move) shards.

AlphaZero's trick, adapted to Generals. A raw policy net is mediocre; adding
LOOKAHEAD makes it much stronger. For each self-play state we take the net's
top-K legal moves, roll each one `horizon` plies through the EXACT engine (the
opponent replies each ply, value at the leaf), and pick the best. That
search-improved move is a stronger label than the raw net's pick. We record
`(features.encode(obs), search_move)` in `learn/train.py`'s shard format, so a
plain fast net can be DISTILLED to play the search policy -- search runs only
here (offline); the shipped net does none, so it stays in the 150ms budget.
Iterate: distil -> stronger base net -> stronger search -> distil again.

Reuses `netoracle._counterfactual_score` (the real-engine rollout) and
`_branch_score` (leaf value: resolved W/L dominates, else safety/army/land).
Search needs the TRUE engine state, which self-play has; the student only ever
sees the fogged observation, exactly as at play time.

    python -m tools.search_gen --net /local/data/vng205/champion.npz \
        --games 1000 --topk 6 --horizon 12 --out /local/data/vng205/search-bc --workers 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bot import features, rules
from bot.memory import TemporalMemory
from bot.policy.net import ClonePolicy, Net
from learn.netoracle import _counterfactual_score
from sim import engine, mapgen

SHARD = 200_000
_CTX: dict = {}


def _search_move(st, seat, net, foe, foe_action, memory, topk, horizon):
    """Argmax over the net's top-K legal moves by real-engine rollout value."""
    obs = engine.observe(st, seat)
    mask = features.legal_mask(obs)
    logits = np.where(mask, net.logits(obs, memory=memory), -np.inf)
    legal = np.flatnonzero(mask)
    if legal.size <= 1:
        return int(np.argmax(logits)), obs
    cand = legal[np.argsort(logits[legal])[-max(1, topk):]]
    best_idx, best_val = int(cand[-1]), -1e18       # default: top policy move
    for c in cand:
        try:
            v = _counterfactual_score(st, memory, foe, net, seat,
                                      features.index_to_action(int(c)),
                                      foe_action, horizon)
        except Exception:                            # noqa: BLE001
            continue
        if v > best_val:
            best_val, best_idx = v, int(c)
    return best_idx, obs


def _one(job):
    seed = job
    grid = mapgen.generate(seed)
    h, w = grid.shape
    st = engine.from_grid(grid)
    net = Net(_CTX["weights"])
    # Opponent = the base policy (self-play against the pre-search net).
    foe = ClonePolicy(1, h, w, _CTX["weights"], tta=False)
    mem = TemporalMemory(h, w)
    topk, horizon = _CTX["topk"], _CTX["horizon"]
    xs, ys = [], []
    for _ in range(_CTX["max_turns"]):
        obs0 = engine.observe(st, 0)
        mem.update(obs0)
        try:
            foe_action = foe.act(engine.observe(st, 1))
        except Exception:                            # noqa: BLE001
            foe_action = rules.PASS_ACTION
        idx, _ = _search_move(st, 0, net, foe, foe_action, mem, topk, horizon)
        xs.append(features.encode(obs0, mem).astype(np.float16))
        ys.append(idx)
        if engine.step(st, features.index_to_action(idx), foe_action):
            break
    if not xs:
        return None
    return np.stack(xs), np.asarray(ys, dtype=np.int32)


def _init(weights, topk, horizon, max_turns):
    _CTX.update(weights=weights, topk=topk, horizon=horizon, max_turns=max_turns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, help="base policy weights .npz")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-turns", type=int, default=1200)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for s in args.out.glob("shard_*.npz"):
        s.unlink()

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    shard = kept = buffered = games = flips = 0

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
                               initargs=(args.net, args.topk, args.horizon, args.max_turns))
    for res in pool.map(_one, range(args.seed0, args.seed0 + args.games), chunksize=1):
        games += 1
        if res is not None:
            x, y = res
            xs.append(x); ys.append(y); buffered += len(y)
            if buffered >= SHARD:
                flush()
        if games % 50 == 0:
            print(f"  {games}/{args.games} games, {kept + buffered} examples", flush=True)
    flush()
    pool.shutdown()

    (args.out / "meta.json").write_text(json.dumps(
        {"examples": kept, "games": games, "channels": features.C,
         "pad": features.PAD, "n_actions": features.N_ACTIONS,
         "base_net": args.net, "topk": args.topk, "horizon": args.horizon}, indent=2))
    print(f"\n{kept} search-labelled examples from {games} games -> {args.out}")


if __name__ == "__main__":
    main()
