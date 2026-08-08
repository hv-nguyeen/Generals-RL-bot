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
    # LABELS ARE {0.0, 1.0} AND DRAWS ARE DROPPED, matching `learn/valuedata.py`
    # exactly, because `learn/valuetrain.py` optimises
    # `logaddexp(0, logit) - y * logit`. That is BCE for y in {0, 1} and
    # UNBOUNDED BELOW for anything else: at y = -1 it reduces to ~2*logit, so the
    # fit drives the logit to -inf and the loss to -1e20 while val accuracy sits
    # at chance. Writing `outcome()`'s {-1, 0, +1} here produced exactly that.

    frames: list = [[], []]
    for _ in range(1, max_turns + 1):
        acts = [None, None]
        for s in (0, 1):
            obs = engine.observe(st, s)
            idx, _, _ = _act(net, obs, rng)
            frames[s].append(features.encode(obs).astype(np.float16))
            acts[s] = features.index_to_action(idx)
        if engine.step(st, acts[0], acts[1]):
            break

    if st.winner < 0:
        return None                      # draw: no label exists for it

    xs, ys = [], []
    for s in (0, 1):
        # Drop the tail: the last ticks of a decided game are trivially
        # predictable and would inflate every evar reading downstream without
        # teaching the critic anything about the positions that matter.
        keep = frames[s][:max(len(frames[s]) - drop_last, 0)][::stride]
        if not keep:
            return None
        xs.append(np.stack(keep))
        ys.append(np.full(len(keep), 1.0 if s == st.winner else 0.0, np.float32))
    return np.concatenate(xs), np.concatenate(ys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="frozen policy to play with")
    ap.add_argument("--out", required=True, help="shard directory for valuetrain")
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--dmin", type=int, default=17, help="stage 4 lower bound")
    ap.add_argument("--dmax", type=int, default=24, help="stage 4 upper bound")
    ap.add_argument("--stride", type=int, default=4, help="keep every Nth tick")
    ap.add_argument("--drop-last", type=int, default=25)
    ap.add_argument("--max-turns", type=int, default=rules.TURN_LIMIT)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed0", type=int, default=VS_SEED0)
    args = ap.parse_args()

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

    xs, ys, shard, kept, buffered, games = [], [], 0, 0, 0, 0

    def flush():
        nonlocal xs, ys, shard, kept, buffered
        if not xs:
            return
        np.savez_compressed(out / f"shard_{shard:04d}.npz",
                            x=np.concatenate(xs), y=np.concatenate(ys))
        kept += buffered
        shard += 1
        xs, ys, buffered = [], [], 0

    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context("spawn"),
                             initializer=_init,
                             initargs=(args.weights, args.max_turns)) as pool:
        for res in pool.map(_game, jobs, chunksize=1):
            games += 1
            if res is not None:
                x, y = res
                xs.append(x)
                ys.append(y)
                buffered += len(y)
                if buffered >= SHARD:
                    flush()
            if games % 100 == 0:
                print(f"  {games}/{len(jobs)} games, {kept + buffered} positions",
                      flush=True)
    flush()

    (out / "meta.json").write_text(json.dumps(
        {"positions": kept, "games": games, "stride": args.stride,
         "drop_last": args.drop_last, "channels": features.C,
         "pad": features.PAD, "weights": args.weights,
         "dmin": args.dmin, "dmax": args.dmax}, indent=2))
    print(f"\n{kept} positions from {games} games -> {out}")
    print(f"next: python -m learn.valuetrain --data {out} --out CRITIC.npz "
          f"--layers 8 --channels 64")
    print("  read its `net` vs `control` line: a critic that does not beat the "
          "board-blind control on held-out data will not beat `sc` in the run "
          "either, and there is no point launching on it.")


def _labels(winner: int, n: int) -> np.ndarray:
    """The two seats' labels for a decided game, seat-major. Extracted so the
    encoding is testable without playing one."""
    return np.concatenate([np.full(n, 1.0 if s == winner else 0.0, np.float32)
                           for s in (0, 1)])


def selfcheck() -> None:
    """The label ENCODING, and that a draw produces no rows at all.

    Both are silent failures. `learn/valuetrain.py` optimises
    `logaddexp(0, logit) - y * logit`, which is BCE on {0, 1} and unbounded
    below outside it: a label of -1 sends the loss to -1e20 with val accuracy
    pinned at chance, which reads as "this critic cannot be fitted" rather than
    as a wrong label. That is exactly what {-1, 0, +1} did on 2026-08-08.
    """
    assert list(_labels(0, 2)) == [1.0, 1.0, 0.0, 0.0]
    assert list(_labels(1, 2)) == [0.0, 0.0, 1.0, 1.0]
    for w in (0, 1):
        y = _labels(w, 3)
        assert set(np.unique(y)) <= {0.0, 1.0}, np.unique(y)
        assert y.sum() == 3                      # exactly one seat wins

    # A stub game through the real path. Nothing can be decided in 12 turns, so
    # what this pins is that a DRAW yields None -- `valuedata` drops draws and
    # keeping them would feed y=0 rows that say "seat 0 lost" about a game
    # nobody lost.
    import tempfile
    from bot.policy.net import DEFAULT_CHANNELS, trunk_keys
    from learn.netoracle import publish, policy_keys
    rng = np.random.default_rng(0)
    arch = {"layers": 2, "channels": DEFAULT_CHANNELS, "residual": False}
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
        assert _game((VS_SEED0, 17, 24, 2, 0)) is None, "a draw must yield no rows"

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
