"""Measure `delta` -- the causal effect of ONE forced castle build -- before
anything is trained on it.

`delta = z(forced build) - z(control)` from the SAME position, both branches
played to terminal by the same stochastic policy under common random numbers.
It is the quantity the counterfactual-build aux loss multiplies:

    L_pref = -mean(delta * log_pi(build | s_fork))

and `delta` is deliberately NOT centred, so `E[delta] <= 0` makes that term push
build probability DOWN. The design's whole purpose is to push it up. So a
non-positive mean here is not weak signal, it is a sign flip, and this probe is
a hard gate in front of a three-week run. See
`docs/design-counterfactual-build.md`.

Nothing in this repo has measured `delta`. Every existing castle number routes
through the critic: `tools/vprobe` reads the critic's own belief, `bldA` in
`learn/selfplay.py` is the critic-derived GAE advantage at real build actions,
and the 2026-08-07 `tools/buildbias` verdict is an inference from what PPO did
next. A forced unilateral deviation played to terminal is the one estimate that
does not depend on the critic being right, which is the point: the design's
premise is that the critic is wrong about castles.

TWO STRATA, NEVER POOLED. Builds are legal on ~89 turn-seats a game at stage 4
against ~1.2 actual builds, so a uniform draw lands on a moment a competent
agent would build at ~1.4%. Sampling only where the champion would build turns
the estimand into "imitate v16's castle timing", a behaviour-cloning target
smuggled into an RL run, and strips the negatives that teach where NOT to build
(`docs/ml-log.md:46`, -36 Elo at ~15 castles/game). So: one candidate reservoir
per stratum, a fair coin between them, and separate means. Stratum A is the
predicate `bot/config.py`'s castle block already encodes on the only
castle-profitable agent in this repo.

    python -m tools.cfprobe --weights runs/nn/sp44.best.npz --games 700

Costs about four minutes on 32 cores. The run it gates costs three weeks.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import features, rules                      # noqa: E402
from bot.board import bfs_field                      # noqa: E402
from bot.policy.net import Net                       # noqa: E402
from sim import engine, mapgen                       # noqa: E402

# Disjoint from every seed block in `learn/selfplay.py` (1.0M train, 1.4M stage,
# 1.5M comp, 1.6M gate, 2.0M pool), so a probe board is never a training board
# and this number cannot be read off a position the policy was fitted on.
CF_PROBE_SEED0 = 3_000_000

_CTX: dict = {}


def _init(weights: str, max_turns: int) -> None:
    _CTX.update(net=Net(weights), max_turns=max_turns)


def outcome(winner: int, seat: int) -> float:
    """Terminal reward for `seat`. Mirrors `learn.selfplay.outcome` exactly.

    Duplicated rather than imported because importing `learn.selfplay` pulls in
    JAX, and this probe spawns 32 workers that have no use for it.
    """
    return 0.0 if winner < 0 else (1.0 if winner == seat else -1.0)


def _act(net: Net, obs, rng):
    """(index, mask, logits) from the masked softmax, sampled. Mirrors
    `learn.selfplay._act`, but also returns the logits: the build cell is chosen
    from the same forward the action came from rather than a second one."""
    mask = features.legal_mask(obs)
    raw = net.logits(obs)
    lg = np.where(mask, raw, -np.inf)
    lg -= lg.max()
    p = np.exp(lg)
    p /= p.sum()
    return int(rng.choice(len(p), p=p)), mask, raw


def build_cells(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """(k, 2) array of cells where a build is legal, from an already-built mask."""
    per, pad = features.PER_CELL, features.PAD
    grid = mask[:pad * pad * per].reshape(pad, pad, per)
    return np.argwhere(grid[:h, :w, features.BUILD_OFFSET])


def stratum_a(obs, r: int, c: int, turn: int, max_turns: int, cost: int) -> bool:
    """Is this a build a competent castle player would consider?

    Every clause is `bot/config.py`'s castle block, which is tuned on v16 -- the
    only agent in this repo measured to gain from castles (+111 Elo,
    `docs/ml-log.md:68`). Distances are BFS steps over passable ground, matching
    `analysis.dist_enemy_terr`, not Chebyshev: a castle two tiles away around a
    mountain wall is not two tiles away.
    """
    if turn < 45:                                   # castle_min_turn
        return False
    if max_turns - turn < 2 * rules.BASE_COST:      # no time to pay itself back
        return False
    if cost != rules.BASE_COST:                     # castle_max_cost: no surcharge
        return False
    if obs.my_land < 12:                            # castle_min_land
        return False
    enemy = obs.owner_grid == rules.OWNER_OPP
    if enemy.any():
        passable = obs.type_grid != rules.T_MOUNTAIN
        if int(bfs_field(passable, enemy)[r, c]) < 7:   # castle_safe_dist
            return False
    return True


def _continue(st, net: Net, seat: int, turn0: int, max_turns: int,
              forced, rng):
    """Play `st` to terminal from turn `turn0`. Returns (z for `seat`, state).

    `forced` overrides `seat`'s action on turn `turn0` only. The override is
    applied AFTER the sample, not instead of it, so both branches consume
    identical draws on the fork turn and the pair stays coupled at t=0 -- which
    is the whole of what common random numbers can buy here. The branches
    occupy different states from t=1 onward and drift apart on their own.
    """
    for turn in range(turn0, max_turns + 1):
        acts = [None, None]
        for s in (0, 1):
            idx, _, _ = _act(net, engine.observe(st, s), rng)
            acts[s] = features.index_to_action(idx)
            if turn == turn0 and s == seat and forced is not None:
                acts[s] = forced
        if engine.step(st, acts[0], acts[1]):
            break
    return outcome(st.winner, seat), st


def _probe(job):
    """One parent game; a fork for EVERY stratum that had a candidate.

    The trainer takes one fork per game -- it is paying for the continuations out
    of its iteration time and a coin flip between the two strata keeps that cost
    flat. This is a measurement, so it takes both when both exist: stratum A is
    the scarce one and the one the gate is computed on, and throwing half of it
    away to a coin would need four times the games to reach the same SE.

    Returns a list, empty if no build was ever legal in the game.
    """
    seed, dmin, dmax = job
    net, max_turns = _CTX["net"], _CTX["max_turns"]
    grid = mapgen.generate(seed, dmin, dmax)
    st = engine.from_grid(grid)

    # The game's action stream and the reservoir's coin flips are SEPARATE
    # streams. Sharing one would make the parent game's trajectory a function of
    # how many build candidates it happened to see, so two probe builds with
    # different stratum predicates would no longer be playing the same games.
    rng = np.random.default_rng(seed)
    pick = np.random.default_rng([seed, 1])

    seen = [0, 0]            # candidates per stratum, for reservoir sampling
    res: list = [None, None]
    turns = 0
    for turn in range(1, max_turns + 1):
        turns = turn
        snap = None
        acts = [None, None]
        for s in (0, 1):
            obs = engine.observe(st, s)
            idx, mask, raw = _act(net, obs, rng)
            acts[s] = features.index_to_action(idx)
            cells = build_cells(mask, obs.H, obs.W)
            if not len(cells):
                continue
            # One snapshot per turn, shared by both seats: `engine.step` has not
            # run yet, so seat 1 sees exactly the board seat 0 saw.
            if snap is None:
                snap = st.copy()
            per, pad = features.PER_CELL, features.PAD
            flat = (cells[:, 0] * pad + cells[:, 1]) * per + features.BUILD_OFFSET
            r, c = (int(v) for v in cells[int(np.argmax(raw[flat]))])
            mine = obs.owner_grid == rules.OWNER_ME
            structs = mine & ((obs.type_grid == rules.T_GENERAL)
                              | (obs.type_grid == rules.T_CASTLE))
            cost = int(rules.build_cost_grid(structs)[r, c])
            k = 0 if stratum_a(obs, r, c, turn, max_turns, cost) else 1
            seen[k] += 1
            if pick.random() < 1.0 / seen[k]:
                res[k] = (snap, turn, s, r, c, int(obs.army_grid[r, c]))
        if engine.step(st, acts[0], acts[1]):
            break

    out = []
    for k in (0, 1):
        if res[k] is None:
            continue
        snap, turn0, seat, r, c, stack = res[k]
        forced = (rules.BUILD, r, c, 0, 0)
        # Both branches from ONE seed: same noise source, same consumption order
        # at the fork. Keyed on the fork turn so a different fork on the same
        # board is a different stream rather than a replay of this one.
        zb, stb = _continue(snap.copy(), net, seat, turn0, max_turns, forced,
                            np.random.default_rng([seed, turn0]))
        zc, _ = _continue(snap.copy(), net, seat, turn0, max_turns, None,
                          np.random.default_rng([seed, turn0]))
        out.append({
            "delta": zb - zc, "z_build": zb, "z_ctrl": zc,
            "stratum": k, "fork_turn": turn0, "game_turns": turns,
            "seen_a": seen[0], "seen_b": seen[1],
            # The two costs that are correct competition rules, not repo bugs,
            # and so are real prices of the intervention rather than things to
            # patch. `sim/engine.py` rewrites BUILD into a PASS, so the build
            # turn forfeits its move; and a captured castle keeps producing, for
            # its new owner.
            "burn": rules.BASE_COST / max(stack, 1),
            "captured": bool(stb.own[1 - seat][r, c]),
        })
    return out


def summarise(rows: list[dict], gate: float, sigma: float) -> bool:
    """Print the per-stratum table and return whether stratum A clears the gate."""
    print(f"\n{len(rows)} forks\n")
    print(f"{'stratum':>9}  {'n':>5}  {'mean':>7}  {'sd':>5}  {'SE':>6}  "
          f"{'-2':>4} {'0':>4} {'+2':>4}  {'fork':>5}  {'burn':>5}  {'capt':>5}")
    passed = False
    for k, name in ((0, "A payable"), (1, "B other")):
        sub = [r for r in rows if r["stratum"] == k]
        if not sub:
            print(f"{name:>9}  {0:>5}   -- no forks in this stratum --")
            continue
        d = np.array([r["delta"] for r in sub], dtype=np.float64)
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
        m = float(d.mean())
        print(f"{name:>9}  {len(d):>5}  {m:>+7.3f}  {d.std():>5.3f}  {se:>6.3f}  "
              f"{int((d < 0).sum()):>4} {int((d == 0).sum()):>4} "
              f"{int((d > 0).sum()):>4}  "
              f"{np.mean([r['fork_turn'] for r in sub]):>5.0f}  "
              f"{np.mean([r['burn'] for r in sub]):>5.2f}  "
              f"{np.mean([r['captured'] for r in sub]):>5.2f}")
        if k == 0:
            passed = m - sigma * se > gate
            if len(d) < 900:
                print(f"{'':>9}  UNDERPOWERED: {len(d)} stratum-A forks, need "
                      f"~900 for SE < {gate / sigma:.3f}. Raise --games.")

    a = [r for r in rows if r["stratum"] == 0]
    print(f"\nlegal build turn-seats per game: "
          f"A {np.mean([r['seen_a'] for r in rows]):.1f}  "
          f"B {np.mean([r['seen_b'] for r in rows]):.1f}   "
          f"mean game {np.mean([r['game_turns'] for r in rows]):.0f} turns")
    print("\n" + "=" * 72)
    if passed:
        print(f"PASS  stratum A mean is above +{gate} at {sigma:.0f} sigma.")
        print("      The aux term's gradient points the way the design claims.")
        print("      Proceed: --cf-frac 0.25 --cf-pref-weight 0.05")
    else:
        print(f"FAIL  stratum A does not clear +{gate} at {sigma:.0f} sigma"
              f"{'' if a else ' (no stratum-A forks at all)'}.")
        print("      `delta` is not centred, so a non-positive mean makes")
        print("      L_pref push build probability DOWN. Do not launch the run;")
        print("      record the number in docs/ml-log.md instead.")
    print("=" * 72)
    return passed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True,
                    help="policy whose delta is being measured, e.g. sp44.best.npz")
    ap.add_argument("--games", type=int, default=3000,
                    help="parent games. The gate needs SE < 0.033 on stratum A, "
                         "and delta has sd ~1.0, so it needs n_A >= 900. A fires "
                         "in only a fraction of games, so 3000 is the floor that "
                         "can reach the gate rather than report 'inconclusive'. "
                         "Read `A n` in the output: if it is under 900 the "
                         "verdict is not powered, whatever it says")
    ap.add_argument("--dmin", type=int, default=17, help="stage 4 lower bound")
    ap.add_argument("--dmax", type=int, default=24, help="stage 4 upper bound")
    ap.add_argument("--max-turns", type=int, default=rules.TURN_LIMIT)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--gate", type=float, default=0.10,
                    help="stratum-A mean in raw z units required to proceed")
    ap.add_argument("--sigma", type=float, default=3.0,
                    help="how many SE below the mean must still clear --gate")
    ap.add_argument("--seed0", type=int, default=CF_PROBE_SEED0)
    ap.add_argument("--save", default="", help="optional .npz of the raw rows")
    args = ap.parse_args()

    jobs = [(args.seed0 + i, args.dmin, args.dmax) for i in range(args.games)]
    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context("spawn"),
                             initializer=_init,
                             initargs=(args.weights, args.max_turns)) as pool:
        for i, r in enumerate(pool.map(_probe, jobs, chunksize=1), 1):
            rows.extend(r)
            if i % 50 == 0:
                print(f"  {i}/{args.games} games, {len(rows)} forks, "
                      f"{time.time() - t0:.0f}s", flush=True)

    if not rows:
        raise SystemExit("no forks: a build was never legal. Check --dmin/--dmax "
                         "and that --weights is the policy you meant.")
    if args.save:
        np.savez_compressed(args.save, **{
            k: np.array([r[k] for r in rows]) for k in rows[0]})
        print(f"wrote {args.save}")
    print(f"\n{args.weights}  dist {args.dmin}-{args.dmax}  "
          f"{time.time() - t0:.0f}s")
    raise SystemExit(0 if summarise(rows, args.gate, args.sigma) else 1)


def selfcheck() -> None:
    """Pins the two things that would silently produce a wrong number.

    Not a unit-test suite: `_probe` needs a trained net and a few core-seconds a
    game, so what is testable in microseconds is the indexing and the reward
    sign, and those are exactly the two that fail quietly.
    """
    # 1. build_cells must agree with the encoder's own action indexing. Off by
    #    one here and the probe forces a MOVE and reports it as a castle.
    pad, per = features.PAD, features.PER_CELL
    m = np.zeros(features.N_ACTIONS, dtype=bool)
    for r, c in ((0, 0), (3, 7), (17, 20)):
        m[((r * pad + c) * per) + features.BUILD_OFFSET] = True
    got = {(int(a), int(b)) for a, b in build_cells(m, 21, 21)}
    assert got == {(0, 0), (3, 7), (17, 20)}, got
    for r, c in got:
        kind, rr, cc, _, _ = features.index_to_action(
            ((r * pad + c) * per) + features.BUILD_OFFSET)
        assert (kind, rr, cc) == (rules.BUILD, r, c), (kind, rr, cc, r, c)

    # 2. The reward sign, both seats. delta is a DIFFERENCE of these, so a flip
    #    on one seat inverts the gate's verdict rather than adding noise.
    assert (outcome(0, 0), outcome(0, 1)) == (1.0, -1.0)
    assert (outcome(1, 0), outcome(1, 1)) == (-1.0, 1.0)
    assert (outcome(-1, 0), outcome(-1, 1)) == (0.0, 0.0)

    # 3. A fork is only meaningful if the snapshot is independent of the parent.
    st = engine.from_grid(mapgen.generate(1, 17, 24))
    snap = st.copy()
    st.armies[0, 0] += 99
    assert snap.armies[0, 0] != st.armies[0, 0], "State.copy() is not deep"
    print("cfprobe selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        main()
