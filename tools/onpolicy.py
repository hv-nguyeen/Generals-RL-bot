"""Label the NET's own positions with what the heuristic would do there.

Castles are worth +111 Elo [76.5, 148.5] to the heuristic (`configs/v16.json`
against the same config with `castle_enabled=False`, 262W-138L over 400 games).
The trained net builds about a tenth of a castle a game against the heuristic's
~0.9, and cannot learn otherwise: a build is legal on 19.5% of turns, the net
puts 0.0009 probability on one, and the payoff lands ~70 turns later diluted
among hundreds of actions. Terminal-only reward cannot assign credit through
that, so the prior has to come from the initialisation.

`tools/mixbuilds.py` tried that and FAILED — 3% of its training labels were
builds and the resulting clone still put 0.0009 on a build. The reason is a
distribution mismatch, not a shortage of examples. Those labels came from
heuristic-vs-heuristic games; the clone is 97% ladder replays, plays like ladder
players, and at inference sits in states the heuristic never visits. It learned
"build in THESE 3711 positions", memorised through 16 repetitions, and those
positions do not come up.

So generate the states from the policy and the labels from the heuristic. This is
one DAgger step: roll the net out, ask the heuristic at every turn what IT would
play, and keep the frames where the answer is a build. The states are then
exactly the ones the net reaches.

    python -m tools.onpolicy --net /local/data/vng205/clone20.npz \\
        --games 2000 --out /local/data/vng205/onpolicy
    python -m tools.mixbuilds --base /local/data/vng205/bc20 \\
        --builds /local/data/vng205/onpolicy --out /local/data/vng205/bc20mix2
    python -m learn.train --data /local/data/vng205/bc20mix2 --layers 8 --channels 32 \\
        --out /local/data/vng205/clone20-build2.npz

Then re-run the p(build) check. **0.0009 is the failure case and the number to
beat**; above ~0.02 a build is sampled a few times a game, which is where credit
assignment starts working. If it fails again the mismatch was not the cause and
the next suspect is the observation — castle cost depends on distance to every
own structure, which is a whole-board quantity the 20 channels may not carry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bot import features, rules
from sim import engine, mapgen

SHARD = 40_000

_NET = _TEACHER = None


def _init(net_path: str, teacher: str) -> None:
    global _NET, _TEACHER
    _NET, _TEACHER = net_path, teacher


def _one_game(job: tuple):
    """Roll the net out; keep the frames where the teacher would build.

    Both seats are the net, so the states are drawn from self-play — the same
    distribution the RL rollouts visit. The teacher never acts; it is queried for
    a label and its answer is discarded unless it is a build.
    """
    seed, max_turns = job
    from arena import agents
    from bot.policy.net import Net

    net = Net(_NET)
    rng = np.random.default_rng(seed)
    grid = mapgen.generate(seed)
    st = engine.from_grid(grid)
    # One teacher per seat: a Controller carries belief state across turns, so a
    # shared instance would be fed two interleaved games and remember neither.
    teach = [agents.make(_TEACHER, i, *grid.shape) for i in (0, 1)]

    xs, ys = [], []
    for _ in range(max_turns):
        obs = [engine.observe(st, 0), engine.observe(st, 1)]
        acts = []
        for i in (0, 1):
            want = teach[i].act(obs[i])          # the label, not the move
            idx = features.action_to_index(want)
            if idx is not None and (idx % features.PER_CELL) == features.PER_CELL - 1:
                xs.append(features.encode(obs[i]).astype(np.float16))
                ys.append(idx)
            m = features.legal_mask(obs[i])
            if not m.any():
                acts.append(rules.PASS_ACTION)
                continue
            # Sampled, not argmax: argmax play is deterministic given the seed and
            # would visit one trajectory per board, which is a far narrower slice
            # of the policy's own distribution than the RL rollouts see.
            z = np.where(m, net.logits(obs[i]), -np.inf)
            z -= z.max()
            p = np.exp(z) * m
            p /= p.sum()
            acts.append(features.index_to_action(int(rng.choice(len(p), p=p))))
        if engine.step(st, acts[0], acts[1]):
            break
    if not xs:
        return None
    return np.stack(xs), np.asarray(ys, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, help="the policy whose states we want")
    ap.add_argument("--teacher", default="ours:configs/v16.json")
    ap.add_argument("--out", default="/local/data/vng205/onpolicy")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed0", type=int, default=4_000_000,
                    help="disjoint from the arena, curriculum and distil blocks")
    ap.add_argument("--max-turns", type=int, default=rules.TURN_LIMIT)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        _init(args.net, args.teacher)
        r = _one_game((args.seed0, 200))
        if r is None:
            print("selfcheck: no builds in 200 turns of one game (possible; "
                  "the teacher builds ~0.9 per FULL game)")
            return
        x, y = r
        assert x.shape[1] == features.C, x.shape
        assert ((y % features.PER_CELL) == features.PER_CELL - 1).all(), "non-build leaked in"
        print(f"onpolicy selfcheck OK ({len(y)} build labels from one 200-turn game)")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("shard_*.npz"):
        stale.unlink()

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(args.seed0 + i, args.max_turns) for i in range(args.games)]
    xs, ys, buffered, shard, total = [], [], 0, 0, 0

    def flush():
        nonlocal xs, ys, buffered, shard, total
        if not xs:
            return
        np.savez_compressed(out / f"shard_{shard:04d}.npz",
                            x=np.concatenate(xs), y=np.concatenate(ys))
        total += buffered
        shard += 1
        xs, ys, buffered = [], [], 0

    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context("spawn"),
                             initializer=_init,
                             initargs=(args.net, args.teacher)) as pool:
        for n, r in enumerate(pool.map(_one_game, jobs, chunksize=1), 1):
            if r is not None:
                x, y = r
                xs.append(x)
                ys.append(y)
                buffered += len(y)
                if buffered >= SHARD:
                    flush()
            if n % 100 == 0:
                print(f"  {n}/{args.games} games, {total + buffered} build labels",
                      flush=True)
    flush()

    (out / "meta.json").write_text(json.dumps(
        {"n_actions": features.N_ACTIONS, "channels": features.C,
         "net": args.net, "teacher": args.teacher, "games": args.games,
         "labels": total}, indent=2) + "\n")

    print(f"\n{total} build labels from {args.games} games of {args.net} states "
          f"labelled by {args.teacher} -> {out}")
    print(f"  {shard} shards. These sit on states the POLICY reaches, which is "
          f"what mixbuilds' heuristic-vs-heuristic frames did not.")


if __name__ == "__main__":
    main()
