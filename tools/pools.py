"""Prebuilt curriculum map pools: one .npz per stage, numpy only, no JAX.

    python -m tools.pools --out runs/pools --workers 32
    python -m tools.pools --selfcheck

WHY THIS EXISTS AT ALL, since `sim.mapgen.generate` is already 3.2 ms a board:
a vectorised rollout cannot call it. `generate` is a python loop over BFS
fields; the GPU path needs N boards as ONE `(N, 21, 21) int32` array, resident,
with the same shape at every curriculum stage so that swapping stages is an
argument change and not an XLA retrace. Building that on the host, once, and
caching it to disk is the whole job.

WHY NOT the starter kit's own `GeneralsEnv` pool. Three of the five documented
blockers in `learn/rlenv.py`'s neighbourhood are properties of that pool rather
than of the problem:

  * every environment's `pool_idx` starts at 0 (`generals/core/game.py:139`) and
    increments identically (`env.py:339`), so 512 parallel envs replay ONE
    environment's worth of boards;
  * `mode="competition"` sets `min_generals_distance=17` from its preset and
    never touches `max`, so `GeneralsEnv(mode="competition",
    max_generals_distance=6)` is min=17/max=6, which empties the candidate set
    and drops into a `farthest` fallback (`grid.py:356-358`) with no error --
    a curriculum built the obvious way gets silently wrong maps;
  * both distances are `static_argnames` (`grid.py:142`), so K bands are K sets
    of compiled kernels.

None of that can happen here. We index the pool ourselves so `pool_idx` is
never read; `_MODE_PRESETS` is never constructed; and every stage is the same
`(N, 21, 21)` array. The distance is MEASURED per board by
`mapgen.generals_distance`, not requested and hoped for, which is the same
discipline `learn/selfplay.selfcheck` already applies -- `generate`'s bracket is
a request and its fallback chain will seat a general outside it rather than
fail.

Boards are byte-identical to what `mapgen.generate` produces for the pool's own
seeds. NOTE they are NOT the boards `--backend cpu` plays: that path draws from
`SP_TRAIN_SEED0` and the pool from `SP_POOL_SEED0`, which are disjoint. A CPU/GPU
A/B therefore shares the distribution, not the maps, and any quality difference
between backends is confounded with the map sample.

File format, per stage:
    grid  (N, 21, 21) int32   -2 mountain, 0 plain, 1 general A, 2 general B,
                              padded bottom/right with -2 exactly as the
                              engine's own generator pads
    h, w  (N,)        int32   the real board inside that pad
    dist  (N,)        int32   MEASURED BFS distance between the generals
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bot import features
from learn.selfplay import SP_POOL_SEED0, STAGES
from sim import mapgen

PAD = features.PAD
DEFAULT_SIZE = 32768        # 58 MB of grids a stage; ~9 replays of each board
                            # over a 1200 x 256 run, and 160 MB as device state
IN_RANGE_MIN = 0.95         # same bar selfplay.selfcheck holds `generate` to


def pool_seed(stage: int, i: int, size: int = DEFAULT_SIZE) -> int:
    """Board seed for one entry. Stages get disjoint runs so a stage's boards
    are never another stage's boards."""
    return SP_POOL_SEED0 + stage * size + i


def pool_seed_span(size: int = DEFAULT_SIZE) -> int:
    """One past the highest seed any pool would use. `selfplay.selfcheck`
    compares it against the other four blocks' origins."""
    return pool_seed(len(STAGES) - 1, size - 1, size) + 1


def _one(args) -> tuple[np.ndarray, int, int, int]:
    stage, i, size = args
    dmin, dmax, _ = STAGES[stage]
    grid = mapgen.generate(pool_seed(stage, i, size), dmin, dmax)
    h, w = grid.shape
    padded = np.full((PAD, PAD), -2, dtype=np.int32)
    padded[:h, :w] = grid
    return padded, h, w, mapgen.generals_distance(grid)


def build(stage: int, size: int = DEFAULT_SIZE, workers: int = 1) -> dict:
    """One stage's pool as plain arrays. ~105 s single-core at the default size."""
    jobs = [(stage, i, size) for i in range(size)]
    runner = (map(_one, jobs) if workers <= 1 else
              ProcessPoolExecutor(max_workers=workers).map(_one, jobs, chunksize=64))
    grid = np.empty((size, PAD, PAD), dtype=np.int32)
    h = np.empty(size, dtype=np.int32)
    w = np.empty(size, dtype=np.int32)
    dist = np.empty(size, dtype=np.int32)
    for i, (g, hh, ww, d) in enumerate(runner):
        grid[i], h[i], w[i], dist[i] = g, hh, ww, d
    return {"grid": grid, "h": h, "w": w, "dist": dist}


def in_range(stage: int, dist: np.ndarray) -> float:
    dmin, dmax, _ = STAGES[stage]
    hi = dmax if dmax is not None else 10 ** 9
    return float(((dist >= dmin) & (dist <= hi)).mean())


def path_for(out: Path | str, stage: int) -> Path:
    return Path(out) / f"stage{stage}.npz"


def load(out: Path | str, stage: int) -> dict:
    """Read one stage back. Raises rather than regenerating: a pool that is
    silently rebuilt with different code is a pool nobody can reproduce."""
    p = path_for(out, stage)
    if not p.exists():
        raise SystemExit(f"no {p} — run `python -m tools.pools --out {out}` first")
    z = np.load(p)
    return {k: z[k] for k in ("grid", "h", "w", "dist")}


KEYS = ("grid", "h", "w", "dist")


def mix(out: Path | str, stage: int, replay: float, seed: int = 0,
        tail: float = 0.0) -> dict:
    """`stage`'s pool with a `replay` share swapped for already-cleared stages.

    `tail` is the same idea upward: that share swapped for boards from the FINAL
    stage whose distance exceeds this stage's `dmax`. Stage 4 is (17, 24) and the
    real ladder is 17+ unbounded -- measured over 600 harvested games, 69.0% fall
    in 17-24, 23.7% in 25-30 and 7.3% in 31-40, so stage 4 never sees 31% of the
    distribution it is evaluated on. `tail=0.31` reconstructs the real histogram;
    anything less is a stage-4-weighted compromise.

    Note what the two measured dose points are before reaching for a large one:
    `tail=0` produced sp16.best at +35.7 Elo, the only checkpoint to beat the
    champion, and stage 5 -- which IS this distribution -- measured about -18.
    A middling `tail` helping requires the response to be non-monotonic.

    Promotion is otherwise a hard switch: enter stage 4 and every board is 11-17
    from then on, stage 3 never appearing again. After a FORCED promotion that is
    the damaging case -- the policy is handed positions it loses whatever it
    plays, terminal reward carries no gradient there, and it loses the distances
    it had. sp8 fell from comp-eval 0.820 through both of its forced promotions.

    The pool KEEPS ITS SIZE, so `npool` and the `it * games` board-index walk in
    `selfplay.play_train` are unchanged and a mixed run stays comparable to a
    plain one. The current stage still supplies `1 - replay` of every batch and
    the promotion gate is untouched, so the curriculum does not slow down.

    Costs no wall clock: the vectorised rollout runs to `--max-turns` under a
    done mask either way, so a short board frees no time and a long one adds none.
    """
    cur = load(out, stage)
    n = len(cur["dist"])
    dmax = STAGES[stage][1]
    k = int(round(replay * n)) if stage > 0 and replay > 0.0 else 0
    # The last stage is unbounded, so it has no "beyond dmax" to draw from and
    # the flag is a no-op there rather than an error.
    kt = int(round(tail * n)) if dmax is not None and tail > 0.0 else 0
    if k == 0 and kt == 0:
        return cur

    rng = np.random.default_rng(seed)
    keep = rng.permutation(n)[:n - k - kt]
    parts = {key: [cur[key][keep]] for key in KEYS}
    drawn = rng.integers(0, stage, size=k) if k else np.empty(0, int)
    for s in range(stage):
        m = int((drawn == s).sum())
        if not m:
            continue
        old = load(out, s)
        j = rng.permutation(len(old["dist"]))[:m]
        for key in KEYS:
            parts[key].append(old[key][j])
    if kt:
        far = load(out, len(STAGES) - 1)
        cand = np.flatnonzero(far["dist"] > dmax)
        # Fail loudly rather than sampling with replacement: a pool that
        # silently repeats 200 boards 30 times is not the distribution anyone
        # asked for, and it would look identical in every downstream number.
        assert kt <= len(cand), (
            f"tail {tail:.2f} needs {kt} boards above dist {dmax} but the final "
            f"stage's pool has {len(cand)}; the most this pool supports is "
            f"{len(cand) / n:.2f}")
        j = rng.permutation(cand)[:kt]
        for key in KEYS:
            parts[key].append(far[key][j])
    mixed = {key: np.concatenate(v) for key, v in parts.items()}
    assert len(mixed["dist"]) == n, "mix must preserve pool size"
    # SHUFFLE, and it is not cosmetic. `selfplay.play_train` walks the pool
    # sequentially -- idx = (it * games + arange) % npool -- so a concatenated
    # block is not a 25% mixture, it is 75% of the run at the current stage
    # followed by 25% entirely at replay distances. Measured: with 32768 boards
    # and 256 games, iteration 96 flipped to dist 3.7 under a stage-3 header and
    # stayed there.
    order = rng.permutation(n)
    mixed = {key: val[order] for key, val in mixed.items()}
    # Cheap guard against that ever coming back: any slice of the pool must
    # still be mostly the current stage. Runs once per stage change.
    head = in_range(stage, mixed["dist"][:max(n // 8, 1)])
    # Threshold relative to what the flags asked for. Replay and tail boards are
    # BOTH out of band, so a fixed 0.5 is a bug once they sum past a half:
    # replay 0.25 with tail 0.31 leaves 0.44 in band and would fail on a
    # perfectly shuffled pool.
    want_in = 1.0 - (replay if stage else 0.0) - (tail if dmax is not None else 0.0)
    assert head > 0.5 * want_in, (
        f"first eighth of the mixed pool is {head:.2f} in band for stage "
        f"{stage}, against {want_in:.2f} asked for -- the mixed-in boards are "
        f"clustered, and play_train walks this array in order")
    # The head guard cannot see a clustered TAIL: tail boards are above dmax, so
    # a run of them lowers `in_range` in one eighth and raises it in another, and
    # a single slice still passes.
    #
    # Tested for PRESENCE per eighth, not for share -- a share test needs a
    # tolerance holding at both n=256 (binomial sd 0.077 per eighth) and n=32768
    # (sd 0.007), and tuning one is how a guard starts failing for reasons that
    # are not the bug.
    if kt:
        for i in range(8):
            sl = mixed["dist"][i * n // 8:(i + 1) * n // 8]
            above = int((sl > dmax).sum())
            # An eighth that is ENTIRELY tail is impossible under a shuffle
            # whenever kt < n, so this side is always safe.
            assert above < len(sl), (
                f"eighth {i} of the mixed pool is entirely above dist {dmax} "
                f"-- the tail boards are concatenated, not interleaved, and "
                f"play_train walks this array in order")
            # The empty side is only safe once an empty eighth is essentially
            # impossible by chance. At kt=8 a correct shuffle leaves some eighth
            # empty 99.8% of the time; at kt>=512 (64 expected per eighth) it is
            # ~1e-30. Below that, skip rather than emit a false "concatenated".
            if kt >= 512:
                assert above > 0, (
                    f"eighth {i} of the mixed pool has no board above dist "
                    f"{dmax} out of {kt} -- the tail boards are concatenated, "
                    f"not interleaved, and play_train walks this array in order")
    return mixed


def write(out: Path | str, stages=None, size: int = DEFAULT_SIZE,
          workers: int = 1) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for stage in (range(len(STAGES)) if stages is None else stages):
        t0 = time.time()
        pool = build(stage, size, workers)
        frac = in_range(stage, pool["dist"])
        dmin, dmax, _ = STAGES[stage]
        # The measured distance is the only proof a stage reached the generator.
        # Boards outside the band are KEPT, not filtered: every stage must stay
        # the same shape or the device pool retraces on promotion, and at 100%
        # in range (which is what every stage measures today) the question is
        # moot anyway.
        assert frac >= IN_RANGE_MIN, (
            f"stage {stage} ({dmin}-{dmax}): only {frac:.2f} of boards in range")
        np.savez_compressed(path_for(out, stage), **pool)
        print(f"  stage {stage} {dmin}-{dmax}: {size} boards, {frac:.0%} in range, "
              f"mean dist {pool['dist'].mean():.1f}, "
              f"sides {pool['h'].min()}-{pool['h'].max()}, {time.time() - t0:.0f}s",
              flush=True)


def selfcheck() -> None:
    """Everything provable without a GPU, a game or a full pool. ~0.3 s."""
    size = 16
    for stage in (0, len(STAGES) - 1):
        pool = build(stage, size)
        dmin, dmax, _ = STAGES[stage]
        assert pool["grid"].shape == (size, PAD, PAD), pool["grid"].shape
        for i in range(size):
            g, h, w = pool["grid"][i], int(pool["h"][i]), int(pool["w"][i])
            # the padded copy IS the generator's board, and the pad is mountain
            want = mapgen.generate(pool_seed(stage, i, size), dmin, dmax)
            assert np.array_equal(g[:h, :w], want), (stage, i)
            assert (g[h:, :] == -2).all() and (g[:, w:] == -2).all(), (stage, i)
            assert int(pool["dist"][i]) == mapgen.generals_distance(want), (stage, i)
            assert (g == 1).sum() == 1 and (g == 2).sum() == 1, (stage, i)
        frac = in_range(stage, pool["dist"])
        assert frac >= IN_RANGE_MIN, (stage, frac)
        print(f"  pool stage {stage} ({dmin}-{dmax}): {frac:.0%} in range, "
              f"mean dist {pool['dist'].mean():.1f}")

    # stage seed runs are disjoint, which is what stops two stages sharing boards
    seen = set()
    for stage in range(len(STAGES)):
        s = {pool_seed(stage, i, DEFAULT_SIZE) for i in (0, DEFAULT_SIZE - 1)}
        assert not (s & seen), stage
        seen |= s
    assert pool_seed_span() == SP_POOL_SEED0 + len(STAGES) * DEFAULT_SIZE

    _selfcheck_tail()
    print("pools selfcheck OK")


def _selfcheck_tail() -> None:
    """`mix(tail=)` keeps the size, hits the share, and stays interleaved.

    The last of those is the one that matters. `play_train` walks the pool in
    order, so a concatenated tail block is not a mixture -- it is most of the run
    in band followed by a stretch entirely beyond it. That exact bug shipped once
    for `replay` and took a run to notice.
    """
    import tempfile

    stage = len(STAGES) - 2                      # the last BOUNDED stage
    dmax = STAGES[stage][1]
    n, frac = 256, 0.25
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        def fake(dists):
            m = len(dists)
            return {"grid": np.zeros((m, PAD, PAD), np.int32),
                    "h": np.full(m, PAD, np.int32), "w": np.full(m, PAD, np.int32),
                    "dist": np.asarray(dists, np.int32)}
        np.savez(out / f"stage{stage}.npz", **fake([dmax - 2] * n))
        np.savez(out / f"stage{len(STAGES) - 1}.npz",
                 **fake([dmax - 2] * (n // 2) + list(range(dmax + 1, dmax + 1 + n // 2))))

        mixed = mix(out, stage, replay=0.0, tail=frac)
        assert len(mixed["dist"]) == n, "tail mix must preserve pool size"
        got = int((mixed["dist"] > dmax).sum())
        assert got == round(frac * n), (got, round(frac * n))
        for i in range(8):
            sl = mixed["dist"][i * n // 8:(i + 1) * n // 8]
            assert 0 < (sl > dmax).sum() < len(sl), (
                f"eighth {i} is all-or-nothing above dist {dmax}: the tail is "
                f"concatenated, not interleaved")

        # tail=0 must be byte-identical to no mixing at all, because every run
        # in flight depends on that default changing nothing.
        assert np.array_equal(mix(out, stage, 0.0, tail=0.0)["dist"],
                              load(out, stage)["dist"])
    print(f"  mix(tail={frac}) at stage {stage}: {got}/{n} above {dmax}, interleaved")


# Measured over 600 harvested games from the ten strongest ladder players,
# 2026-08-06. `analysis.official` has the replays; the buckets are BFS distance
# between the generals on the opening frame.
LADDER_BUCKETS = ((17, 24, 0.690), (25, 30, 0.237), (31, 40, 0.073))
LADDER_MEAN, LADDER_MEDIAN = 22.7, 22


def mix_report(out: Path | str, stage: int, tail: float, replay: float = 0.0) -> None:
    """The distance histogram a run WOULD train on, against the real one.

    Runs the production `mix`, guards included, so it fails in exactly the cases
    a real run would -- including the head guard, which is why `replay` is a
    parameter rather than pinned to zero.
    """
    dmin, dmax, _ = STAGES[stage]
    if dmax is None:
        raise SystemExit(f"stage {stage} is unbounded, so --dist-tail is a no-op "
                         f"there and there is nothing to report. Use a bounded "
                         f"stage (0..{len(STAGES) - 2}).")
    host = mix(out, stage, replay, tail=tail)
    d = host["dist"]
    n = len(d)
    print(f"stage {stage} ({dmin}-{dmax}), tail {tail:.2f}, replay {replay:.2f}, "
          f"{n} boards")
    print(f"  mean {d.mean():.1f} (ladder {LADDER_MEAN})   "
          f"median {int(np.median(d))} (ladder {LADDER_MEDIAN})   "
          f"min {d.min()}  max {d.max()}")
    # The in-band buckets share `1 - tail - replay` between them in the same
    # proportions the ladder gives them; the out-of-band ones share `tail` in the
    # ladder's proportions. Written this way so the report is right at every
    # stage, not only where dmax happens to end a bucket -- at stage 3 (dmax 17)
    # no bucket edge lines up at all.
    inb = [(lo, hi, r) for lo, hi, r in LADDER_BUCKETS if hi <= dmax]
    outb = [(lo, hi, r) for lo, hi, r in LADDER_BUCKETS if hi > dmax]
    si = sum(r for _, _, r in inb) or 1.0
    so = sum(r for _, _, r in outb) or 1.0
    for lo, hi, real in LADDER_BUCKETS:
        got = float(((d >= lo) & (d <= hi)).mean())
        want = (real / si) * (1.0 - tail - replay) if hi <= dmax else (real / so) * tail
        flag = "" if abs(got - want) < 0.015 else "   <-- off"
        print(f"  {lo:>2}-{hi:<2} {got:6.1%}   want {want:6.1%}   "
              f"ladder {real:5.1%}{flag}")
    if dmin < LADDER_BUCKETS[0][0]:
        print(f"  NOTE this stage trains below dist {LADDER_BUCKETS[0][0]}, the "
              f"competition minimum, so the ladder column does not apply and "
              f"the in-band rows are not comparable to it. Only stage "
              f"{len(STAGES) - 2} is.")
    elif any(lo < dmax < hi for lo, hi, _ in LADDER_BUCKETS):
        print(f"  NOTE dist {dmax} splits a bucket, so the two rows either side "
              f"of it are not directly comparable to the ladder column")
    assert d.min() >= dmin or replay, f"a board below the stage minimum: {d.min()}"
    print("  per-eighth tail share: " + " ".join(
        f"{float((d[i * n // 8:(i + 1) * n // 8] > dmax).mean()):.2f}"
        for i in range(8)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/pools")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--stage", type=int, default=None, help="one stage, else all")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--tail", type=float, default=0.0,
                    help="with --mix-report, the --dist-tail to inspect")
    ap.add_argument("--replay", type=float, default=0.0,
                    help="with --mix-report, the --stage-replay to inspect")
    ap.add_argument("--mix-report", action="store_true",
                    help="print the distance histogram a run would actually "
                         "train on, against the real ladder's, before spending "
                         "a GPU night on it")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    if args.mix_report:
        mix_report(args.out, args.stage if args.stage is not None else len(STAGES) - 2,
                   args.tail, args.replay)
        return
    write(args.out, None if args.stage is None else [args.stage],
          args.size, args.workers)


if __name__ == "__main__":
    main()
