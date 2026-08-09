"""Stage-4 value-training data from a FROZEN policy's own self-play.

`learn/valuedata.py` builds the same shards from ladder replays, and that critic
did not transfer: commit 547da6e records 0.759 BCE on field positions buying
NEGATIVE evar on self-play returns. The positions were someone else's.

This writes the same format from the policy's own games at the distance band it
will actually train on, so "on-distribution" is true by construction rather than
by hope. The output goes straight into `learn.valuetrain` and then into
`learn.selfplay --init-critic`, which already accepts a standalone critic
(`load_critic` falls back to bare keys for exactly this).

WHY THIS EXISTS. Three runs on 2026-08-08 -- sp70, sp72, sp73 -- all failed the
same way: `sp43.resume.npz`'s critic warmed at stage 0, and at stage 4 its `evar`
sat BELOW `sc`, the control that cannot see the board. The policy never unfroze
in any of them, because `do_policy = warmed`. Warming a critic online, against a
moving policy, on 460-turn games, is the hardest version of the problem. Fitting
it offline against fixed Monte-Carlo returns is the easiest, and the label is
exact: with gamma=1 and terminal-only reward the target IS the game's outcome.

It also sidesteps the curriculum trap. Reaching stage 4 by promotion means
training the policy through stages 2-3 first, where castles are correctly bad
(`bldA` -0.9 to -3.0) and self-play deletes them -- so a build bias would be
gone before it arrived. Here the policy never trains at all; only the critic
does, and it is fitted directly at stage 4.

    python -m tools.valueselfplay --weights runs/nn/sp44-b6.npz \\
        --out /local/data/vng205/val4 --games 3000 --workers 60
    python -m learn.valuetrain --data /local/data/vng205/val4 \\
        --out runs/nn/critic4.npz --layers 8 --channels 64
    ... --init-critic runs/nn/critic4.npz

`--layers/--channels` are NOT optional on that middle step: `valuetrain` defaults
to a 4-layer trunk and a 4-layer critic cannot load into an 8-layer policy.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import features, rules                       # noqa: E402
from bot.memory import TemporalMemory                 # noqa: E402
from bot.policy.net import Net                        # noqa: E402
from sim import engine, mapgen                        # noqa: E402
from tools.cfprobe import _act                      # noqa: E402

# Disjoint from every other block: 1.0M train, 1.4M stage, 1.5M comp, 1.6M gate,
# 2.0M pool, 3.0M cfprobe. A critic pretrained on the very boards the run then
# trains on would read as a healthy evar it had partly memorised.
VS_SEED0 = 4_000_000
SHARD = 60_000          # matches learn/valuedata.py, so shards interleave


_CTX: dict = {}


def _init(weights: str, max_turns: int) -> None:
    _CTX.update(net=Net(weights), max_turns=max_turns)


def _game(job):
    """One self-play game -> (x, y) for BOTH seats, strided.

    Both seats are the same frozen network and both sample, exactly as
    `learn/selfplay.py:_rollout` mode 0 does, so the state distribution is the
    one the run will see. Sampling rather than argmax matters: a critic fitted
    only to greedy play has never seen the states exploration reaches.
    """
    seed, dmin, dmax, stride, drop_last = job
    net, max_turns = _CTX["net"], _CTX["max_turns"]
    st = engine.from_grid(mapgen.generate(seed, dmin, dmax))
    rng = np.random.default_rng(seed)
    # Direct value labels {-1, 0, +1}, including draws. `learn.valuetrain`
    # optimises a bounded tanh value with Huber loss in the same units used by
    # PPO, so standalone critics transfer without a hidden logit conversion.

    frames: list = [[], []]
    memories = [TemporalMemory(*st.armies.shape) for _ in range(2)]
    for _ in range(1, max_turns + 1):
        acts = [None, None]
        for s in (0, 1):
            obs = engine.observe(st, s)
            memories[s].update(obs)
            idx, _, _ = _act(net, obs, rng, memories[s])
            frames[s].append(features.encode(obs, memories[s]).astype(np.float16))
            acts[s] = features.index_to_action(idx)
        if engine.step(st, acts[0], acts[1]):
            break

    xs, ys, game_ids, seats, turns = [], [], [], [], []
    for s in (0, 1):
        # Drop the tail: the last ticks of a decided game are trivially
        # predictable and would inflate every evar reading downstream without
        # teaching the critic anything about the positions that matter.
        keep = frames[s][:max(len(frames[s]) - drop_last, 0)][::stride]
        if not keep:
            return None
        xs.append(np.stack(keep))
        target = 0.0 if st.winner < 0 else (1.0 if s == st.winner else -1.0)
        ys.append(np.full(len(keep), target, np.float32))
        game_ids.append(np.full(len(keep), seed, np.int64))
        seats.append(np.full(len(keep), s, np.int8))
        n0 = max(len(frames[s]) - drop_last, 0)
        turns.append(np.arange(0, n0, stride, dtype=np.int32)[:len(keep)])
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(game_ids),
            np.concatenate(seats), np.concatenate(turns))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="frozen policy to play with")
    ap.add_argument("--out", required=True, help="shard directory for valuetrain")
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--dmin", type=int, default=17, help="stage 4 lower bound")
    ap.add_argument("--dmax", type=int, default=24, help="stage 4 upper bound")
    ap.add_argument("--competition-distance", action="store_true",
                    help="use the exact competition distribution (17+ with no "
                         "upper bound), overriding --dmin/--dmax")
    ap.add_argument("--stride", type=int, default=4, help="keep every Nth tick")
    ap.add_argument("--drop-last", type=int, default=25)
    ap.add_argument("--max-turns", type=int, default=rules.TURN_LIMIT)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed0", type=int, default=VS_SEED0)
    args = ap.parse_args()

    if args.competition_distance:
        args.dmin, args.dmax = rules.MIN_GENERALS_DISTANCE, None

    # Before the pool: `_init` runs in every worker, so a bad path otherwise
    # prints 60 tracebacks and dies in a BrokenProcessPool naming neither.
    if not os.path.exists(args.weights):
        raise SystemExit(f"--weights {args.weights}: no such file. /local/data is "
                         f"per-node -- the checkpoint may be on another one.")
    Net(args.weights)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(args.seed0 + i, args.dmin, args.dmax, args.stride, args.drop_last)
            for i in range(args.games)]

    xs, ys, game_ids, seats, turns = [], [], [], [], []
    shard = kept = buffered = games = 0

    def flush():
        nonlocal xs, ys, game_ids, seats, turns, shard, kept, buffered
        if not xs:
            return
        np.savez_compressed(out / f"shard_{shard:04d}.npz",
                            x=np.concatenate(xs), y=np.concatenate(ys),
                            game=np.concatenate(game_ids),
                            seat=np.concatenate(seats), t=np.concatenate(turns))
        kept += buffered
        shard += 1
        xs, ys, game_ids, seats, turns, buffered = [], [], [], [], [], 0

    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context("spawn"),
                             initializer=_init,
                             initargs=(args.weights, args.max_turns)) as pool:
        for res in pool.map(_game, jobs, chunksize=1):
            games += 1
            if res is not None:
                x, y, game, seat, turn = res
                xs.append(x)
                ys.append(y)
                game_ids.append(game); seats.append(seat); turns.append(turn)
                buffered += len(y)
                if buffered >= SHARD:
                    flush()
            if games % 100 == 0:
                print(f"  {games}/{len(jobs)} games, {kept + buffered} positions",
                      flush=True)
    flush()

    from tools.manifest import artifact, file_set
    shards = sorted(out.glob("shard_*.npz"))
    (out / "meta.json").write_text(json.dumps(
        {"positions": kept, "games": games, "stride": args.stride,
         "drop_last": args.drop_last, "channels": features.C,
         "pad": features.PAD, "weights": artifact(args.weights),
         "dmin": args.dmin, "dmax": args.dmax, "seed0": args.seed0,
         "shards": file_set(shards, out)}, indent=2))
    print(f"\n{kept} positions from {games} games -> {out}")
    print(f"next: python -m learn.valuetrain --data {out} --out CRITIC.npz "
          f"--layers 8 --channels 64")
    print("  read its `net` vs `control` line: a critic that does not beat the "
          "board-blind control on held-out data will not beat `sc` in the run "
          "either, and there is no point launching on it.")


def _labels(winner: int, n: int) -> np.ndarray:
    """Direct value targets in seat-major order; ``winner=-1`` is a draw."""
    return np.concatenate([
        np.full(n, 0.0 if winner < 0 else (1.0 if s == winner else -1.0),
                np.float32) for s in (0, 1)])


def selfcheck() -> None:
    """Pin direct {-1,0,+1} labels, pairing, and retained draws."""
    assert list(_labels(0, 2)) == [1.0, 1.0, -1.0, -1.0]
    assert list(_labels(1, 2)) == [-1.0, -1.0, 1.0, 1.0]
    assert list(_labels(-1, 2)) == [0.0, 0.0, 0.0, 0.0]
    for w in (0, 1):
        y = _labels(w, 3)
        assert set(np.unique(y)) <= {-1.0, 1.0}, np.unique(y)
        assert y.sum() == 0                      # zero-sum across seats

    # A stub game through the real path. Nothing can be decided in 12 turns, so
    # this pins that a draw remains in the data with value 0 for both seats.
    import tempfile
    from bot.policy.net import DEFAULT_CHANNELS, trunk_keys
    from learn.netoracle import publish, policy_keys
    rng = np.random.default_rng(0)
    arch = {"layers": 2, "channels": DEFAULT_CHANNELS, "residual": False,
            "context": False}
    p, prev = {}, features.C
    for name in trunk_keys(arch["layers"], arch["residual"]):
        p[f"{name}_w"] = (rng.normal(size=(arch["channels"], prev, 3, 3)) * 0.1).astype("f4")
        p[f"{name}_b"] = np.zeros(arch["channels"], np.float32)
        prev = arch["channels"]
    p["head_w"] = (rng.normal(size=(features.PER_CELL, prev, 3, 3)) * 0.1).astype("f4")
    p["head_b"] = np.zeros(features.PER_CELL, np.float32)
    p["pass_w"] = np.zeros(prev, np.float32)
    p["pass_b"] = np.float32(0.0)
    assert set(p) == policy_keys(arch)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ck.npz"
        publish(p, path)
        _init(str(path), 12)
        draw = _game((VS_SEED0, 17, 24, 2, 0))
        assert draw is not None and np.all(draw[1] == 0), "a draw must target value 0"

        # And the shape/pairing on a decided game, forced by handing the packer a
        # winner directly rather than trying to win one in 12 turns.
        _init(str(path), 12)
        n = 5
        x = np.zeros((2 * n, features.C, features.PAD, features.PAD), np.float16)
        y = _labels(0, n)
        assert x.shape[1:] == (features.C, features.PAD, features.PAD)
        assert len(x) == len(y) and not np.array_equal(y[:n], y[n:])
        _CTX.clear()
    print("valueselfplay selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        main()
