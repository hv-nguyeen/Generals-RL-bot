"""Set the policy's castle-building prior by editing one number.

Castles are worth +111 Elo [76.5, 148.5] to the heuristic (`configs/v16.json`
against the same config with `castle_enabled=False`, 262W-138L over 400 games).
The trained net assigns mean probability **0.0009** to a build in the 19.2% of
turns where one is legal — about a tenth of a castle per game against the
heuristic's ~0.9 — and cannot learn otherwise, because the payoff lands ~70 turns
later and terminal-only reward will not carry credit that far through an action
it never samples.

TWO THINGS THAT DID NOT WORK, so nobody repeats them:

`tools/mixbuilds.py` spliced heuristic-vs-heuristic build frames into the clone's
training set until they were 3% of all labels. The retrained clone still read
0.0009. That is not a failure to learn — softmax cross-entropy fits the
CONDITIONAL P(build | state), and on the ladder-like states this clone actually
visits its training data says P(build) = 34/203924 = 0.00017. The 3% marginal
lived entirely on 3,711 frames that got 16 repetitions x 20 epochs = 320
exposures each into a 73k-parameter net, which memorised them; they never recur
at inference. It also spliced into the held-out shards, so validation top-1 was
incapable of noticing.

Retraining with more or better-distributed labels is the expensive version of the
same idea, and it is not needed for what the clone is FOR. The clone is an RL
initialisation. PPO is supposed to learn WHEN to build; it can only do that if
builds get sampled at all. A flat prior over legal build cells is enough for
that, and the legal mask already gates affordability.

`bot/policy/net.py`'s head is a single 3x3 conv emitting `PER_CELL` channels, so
slot 8's bias is ONE SCALAR shared by every cell. Adding to it raises the build
prior uniformly, is independent of the input, and is therefore immune by
construction to the distribution shift that defeated the data approach.

    python -m tools.buildprior --net clone20.npz --out clone20-bp.npz --target 0.03

Then confirm on states the policy actually reaches:

    mean p(build | legal) 0.0009  ->  ~0.03

WATCH IT SURVIVE TRAINING. Curriculum stages 0-3 run short games on boards with
no safe rear, where building is genuinely wrong, so the prior can be eroded
before the curriculum reaches boards where it pays. `learn/selfplay.py`'s `bld`
field is the instrument; it should rise off 0.00 by stage 4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bot import features
from bot.policy.net import Net
from sim import engine, mapgen


def probe(net: Net, seeds, opponent: str, max_turns: int):
    """Mean P(build) over the build-legal states this policy reaches.

    The policy drives both seats, so the states are its own — measuring on
    heuristic-vs-heuristic states is what made the data approach look fine right
    up until it was tested where it mattered.
    """
    from arena import agents
    legal, steps, pm = 0, 0, []
    for seed in seeds:
        grid = mapgen.generate(seed)
        st = engine.from_grid(grid)
        opp = agents.make(opponent, 1, *grid.shape)
        for _ in range(max_turns):
            o = engine.observe(st, 0)
            m = features.legal_mask(o)
            steps += 1
            if not m.any():
                break
            cells = m[:features.PAD * features.PAD * features.PER_CELL]
            cells = cells.reshape(-1, features.PER_CELL)
            lg = net.logits(o)
            if cells[:, features.BUILD_OFFSET].any():
                legal += 1
                z = np.where(m, lg, -np.inf)
                z = z - z.max()
                p = np.exp(z)
                p /= p.sum()
                pc = p[:features.PAD * features.PAD * features.PER_CELL]
                pm.append(float(pc.reshape(-1, features.PER_CELL)[:, features.BUILD_OFFSET].sum()))
            a = features.index_to_action(int(np.argmax(np.where(m, lg, -np.inf))))
            if engine.step(st, a, opp.act(engine.observe(st, 1))):
                break
    return (float(np.mean(pm)) if pm else 0.0), legal, steps


def with_bias(path: str, delta: float, tmp: Path) -> str:
    z = dict(np.load(path))
    b = z["head_b"].copy()
    b[features.BUILD_OFFSET] += delta
    z["head_b"] = b
    np.savez(tmp, **z)
    return str(tmp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net")
    ap.add_argument("--out")
    ap.add_argument("--target", type=float, default=0.03,
                    help="mean P(build) in build-legal states. 0.02-0.05 gets a "
                         "build sampled a few times a game, which is where "
                         "credit assignment starts working")
    ap.add_argument("--opponent", default="ours:configs/v16.json")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=5_000_000,
                    help="disjoint from the training, distil and arena blocks")
    ap.add_argument("--max-turns", type=int, default=500)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if not args.selfcheck and not (args.net and args.out):
        ap.error("--net and --out are required unless --selfcheck")

    if args.selfcheck:
        # The bias is one scalar and slot 8 is where builds live -- assert both,
        # since the whole method rests on them.
        assert features.BUILD_OFFSET == features.PER_CELL - 1, features.BUILD_OFFSET
        import tempfile
        d = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(0)
        fake = {"head_b": rng.normal(size=(features.PER_CELL,)).astype(np.float32)}
        np.savez(d / "a.npz", **fake)
        p = with_bias(str(d / "a.npz"), 2.5, d / "b.npz")
        got = np.load(p)["head_b"]
        assert np.isclose(got[features.BUILD_OFFSET],
                          fake["head_b"][features.BUILD_OFFSET] + 2.5)
        assert np.allclose(np.delete(got, features.BUILD_OFFSET),
                           np.delete(fake["head_b"], features.BUILD_OFFSET)), \
            "only the build slot may move"
        print("buildprior selfcheck OK (slot 8, one scalar, nothing else touched)")
        return

    seeds = range(args.seed0, args.seed0 + args.games)
    import tempfile
    tmp = Path(tempfile.mkdtemp())

    base, legal, steps = probe(Net(args.net), seeds, args.opponent, args.max_turns)
    print(f"before: p(build|legal) {base:.4f}   legal on {legal}/{steps} turns "
          f"({100 * legal / max(steps, 1):.1f}%)")
    if base >= args.target:
        print(f"already at or above --target {args.target}; nothing to do")
        return

    # p(build) is monotone in the bias and spans (0, 1), so bisect. Solving it
    # in closed form would need the logit gap to every legal move in every state.
    lo, hi = 0.0, 12.0
    best = None
    for i in range(9):
        mid = (lo + hi) / 2
        cand = with_bias(args.net, mid, tmp / f"c{i}.npz")
        got, _, _ = probe(Net(cand), seeds, args.opponent, args.max_turns)
        print(f"  bias {mid:+.2f} -> p(build|legal) {got:.4f}")
        if got < args.target:
            lo = mid
        else:
            hi = mid
            best = (mid, got)
    if best is None:
        raise SystemExit(f"even bias +{hi:.1f} does not reach {args.target}; the "
                         f"build slot may be suppressed by the head weights, not "
                         f"the bias")

    delta, got = best
    z = dict(np.load(args.net))
    b = z["head_b"].copy()
    b[features.BUILD_OFFSET] += delta
    z["head_b"] = b
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **z)
    print(f"\nhead_b[{features.BUILD_OFFSET}] += {delta:.3f} -> {args.out}")
    print(f"  p(build|legal) {base:.4f} -> {got:.4f}")
    print("\nNothing else changed. Now watch `bld` in the training log: it should")
    print("rise off 0.00 by stage 4. Stages 0-3 are short games with no safe rear,")
    print("where building really is wrong, so the prior can erode before it pays.")


if __name__ == "__main__":
    main()
