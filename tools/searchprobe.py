"""Measure whether policy-guided engine search has a usable teaching signal.

This is a falsification probe, not a deployed search policy.  It samples states
from frozen-policy mirror self-play, expands the policy's top-k legal actions,
samples the simultaneous opponent response, and plays every candidate to a
terminal result under paired continuation seeds.  An optional matched critic is
read at a fixed branch depth and compared with those terminal action rankings.

The hidden simulator state used for a row is the one that actually generated
the player's observation.  That makes each row an on-policy sample from the
policy-induced hidden-state distribution, but it is still only ONE
determinization of that information state.  Consequently this tool writes
action-return samples, not hard "best move" labels, and stamps its summary
``single_actual_hidden_state``.  A positive result is an upper bound that
justifies building observation-conditioned Q/reanalysis; it is not evidence
that perfect-information search is safe at deployment.

Example on a compute node::

    python -m tools.searchprobe \
      --weights "$HOME/top3-v2-shared/weights/selfplay-champion-gen1.npz" \
      --critic "$HOME/top3-v2-shared/weights/selfplay-champion-gen1.critic.npz" \
      --out /local/data/vng205/search-probe-gen1 \
      --games 128 --positions-per-game 2 --topk 6 --rollouts 8 --workers 32

Read ``summary.json`` before training anything.  The load-bearing fields are
``robust_change_frac``, ``robust_lcb_gain``,
``oracle_best_at_topk_boundary_frac`` and ``critic_pair_accuracy``.  The rows in
``rows.npz`` retain all rollout samples so a later audit can change the
confidence rule without regenerating games.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bot import features, rules
from bot.memory import TemporalMemory
from bot.policy.net import Net, ValueNet
from sim import engine, mapgen


SP_SEED0 = 5_000_000
SCOPE = "single_actual_hidden_state"
_CTX: dict = {}


def _softmax(logits: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masked = np.where(mask, np.asarray(logits, np.float64), -np.inf)
    masked -= np.max(masked)
    p = np.exp(masked)
    p /= p.sum()
    return masked, p


def _act(net: Net, obs, memory: TemporalMemory, rng: np.random.Generator,
         tta: bool = False) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    memory.update(obs)
    mask = features.legal_mask(obs)
    logits = net.logits(obs, tta=tta, full=tta, memory=memory)
    _, p = _softmax(logits, mask)
    return int(rng.choice(len(p), p=p)), mask, logits, p


def _outcome(winner: int, seat: int) -> float:
    return 0.0 if winner < 0 else (1.0 if winner == seat else -1.0)


def _topk(mask: np.ndarray, logits: np.ndarray, k: int) -> np.ndarray:
    legal = np.flatnonzero(mask)
    order = legal[np.argsort(np.asarray(logits)[legal], kind="stable")[::-1]]
    return order[:min(k, len(order))].astype(np.int32)


def robust_slot(returns: np.ndarray, confidence: float = 2.0,
                min_gap: float = 0.0) -> tuple[int, np.ndarray, np.ndarray]:
    """Conservative paired choice relative to slot 0 (policy argmax).

    Returns ``(slot, mean_delta, lower_bound)``.  The same continuation seed is
    used across action slots, so uncertainty is computed on paired differences,
    not by adding independent standard errors.
    """
    z = np.asarray(returns, np.float64)
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] < 1:
        raise ValueError(f"returns must be (actions, rollouts), got {z.shape}")
    delta = z - z[0]
    mean = delta.mean(axis=1)
    if z.shape[1] > 1:
        se = delta.std(axis=1, ddof=1) / math.sqrt(z.shape[1])
    else:
        se = np.full(z.shape[0], np.inf)
        se[0] = 0.0
    # Avoid IEEE ``0 * inf -> nan`` when a caller deliberately disables the
    # uncertainty penalty for a one-rollout smoke test.
    lower = mean if confidence == 0 else mean - float(confidence) * se
    eligible = np.flatnonzero(lower > float(min_gap))
    eligible = eligible[eligible != 0]
    if not len(eligible):
        return 0, mean, lower
    # Maximise the lower bound, then the point estimate, then prefer the policy
    # ordering (lower slot) for a deterministic and conservative tie-break.
    slot = max((int(i) for i in eligible),
               key=lambda i: (float(lower[i]), float(mean[i]), -i))
    return slot, mean, lower


def pair_accuracy(target: np.ndarray, estimate: np.ndarray,
                  min_gap: float = 0.05) -> tuple[int, int]:
    """Concordant action pairs, excluding terminal-return near ties."""
    y, v = np.asarray(target), np.asarray(estimate)
    good = total = 0
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            d = float(y[i] - y[j])
            if abs(d) < min_gap:
                continue
            total += 1
            good += int(np.sign(d) == np.sign(float(v[i] - v[j])))
    return good, total


def _value(critic: ValueNet | None, st: engine.State, seat: int,
           memory: TemporalMemory) -> float:
    if critic is None:
        return float("nan")
    obs = engine.observe(st, seat)
    memory.update(obs)
    return critic.value_encoded(features.encode(obs, memory))


def _branch(snapshot, seat: int, first_idx: int, foe_idx: int, seed: int,
            critic_depth: int) -> tuple[float, float]:
    """One exact-engine continuation; returns terminal z and leaf critic V."""
    st, memories = snapshot[0].copy(), [m.copy() for m in snapshot[1]]
    net: Net = _CTX["net"]
    critic: ValueNet | None = _CTX["critic"]
    tta = _CTX["tta"]
    rng = np.random.default_rng(seed)
    acts = [None, None]
    acts[seat] = features.index_to_action(int(first_idx))
    acts[1 - seat] = features.index_to_action(int(foe_idx))
    ended = engine.step(st, acts[0], acts[1])
    depth = 1
    leaf = (_outcome(st.winner, seat) if ended
            else _value(critic, st, seat, memories[seat])
            if critic_depth == 1 else float("nan"))

    while not ended and st.time < _CTX["max_turns"]:
        acts = [None, None]
        for s in (0, 1):
            obs = engine.observe(st, s)
            idx, _, _, _ = _act(net, obs, memories[s], rng, tta)
            acts[s] = features.index_to_action(idx)
        ended = engine.step(st, acts[0], acts[1])
        depth += 1
        if leaf != leaf and (ended or depth == critic_depth):
            leaf = (_outcome(st.winner, seat) if ended
                    else _value(critic, st, seat, memories[seat]))

    z = _outcome(st.winner, seat) if ended else 0.0
    if leaf != leaf and critic is not None:
        # A branch can hit max_turns before the requested depth only when the
        # sampled parent was already near the draw boundary.  The terminal
        # convention is then the only calibrated target.
        leaf = z
    return z, leaf


def _offer(reservoir: list, seen: int, capacity: int, item, rng) -> int:
    seen += 1
    if len(reservoir) < capacity:
        reservoir.append(item)
    else:
        j = int(rng.integers(seen))
        if j < capacity:
            reservoir[j] = item
    return seen


def _position(seed: int, snap, pos_index: int) -> dict:
    st, memories, seat = snap
    net: Net = _CTX["net"]
    obs = engine.observe(st, seat)
    memory = memories[seat]
    memory.update(obs)
    mask = features.legal_mask(obs)
    logits = net.logits(obs, tta=_CTX["tta"], full=_CTX["tta"], memory=memory)
    _, prior = _softmax(logits, mask)
    candidates = _topk(mask, logits, _CTX["topk"])
    k, r = len(candidates), _CTX["rollouts"]
    returns = np.empty((k, r), np.float32)
    leaves = np.full((k, r), np.nan, np.float32)

    for ri in range(r):
        # The opponent moves simultaneously from the same pre-action
        # observation, so sample its response once and reuse it for every own
        # candidate.  Subsequent rollout RNGs are also paired across candidates.
        response_rng = np.random.default_rng([seed, int(st.time), pos_index, ri, 17])
        foe_memory = memories[1 - seat].copy()
        foe_obs = engine.observe(st, 1 - seat)
        foe_idx, _, _, _ = _act(net, foe_obs, foe_memory, response_rng, _CTX["tta"])
        branch_seed = int(np.random.SeedSequence(
            [seed, int(st.time), pos_index, ri, 29]).generate_state(1)[0])
        for ai, idx in enumerate(candidates):
            returns[ai, ri], leaves[ai, ri] = _branch(
                (st, memories), seat, int(idx), foe_idx, branch_seed,
                _CTX["critic_depth"])

    chosen, delta, lower = robust_slot(
        returns, _CTX["confidence"], _CTX["min_gap"])
    return {
        "x": features.encode(obs, memory).astype(np.float16),
        "seed": np.int64(seed), "turn": np.int32(st.time),
        "seat": np.int8(seat), "actions": candidates,
        # Distillation needs the complete legal support to keep the candidate
        # policy anchored outside top-k. Reconstructing it from a compressed
        # tensor later is an unnecessary second implementation of the rules.
        "legal": mask.astype(bool),
        "prior_logp": np.log(np.maximum(prior[candidates], 1e-30)).astype(np.float32),
        "returns": returns, "leaves": leaves,
        "selected": np.int16(chosen),
        "delta": delta.astype(np.float32), "lower": lower.astype(np.float32),
    }


def _init(weights: str, critic: str | None, args: dict) -> None:
    _CTX.clear()
    _CTX.update(args)
    _CTX["net"] = Net(weights)
    _CTX["critic"] = ValueNet(critic) if critic else None


def _game(seed: int) -> list[dict]:
    net: Net = _CTX["net"]
    grid = mapgen.generate(seed)
    st = engine.from_grid(grid)
    memories = [TemporalMemory(*grid.shape) for _ in range(2)]
    play_rng = np.random.default_rng([seed, 3])
    reservoir: list = []
    seen = 0

    while st.time < _CTX["max_turns"]:
        acts = [None, None]
        for seat in (0, 1):
            obs = engine.observe(st, seat)
            idx, _, _, _ = _act(net, obs, memories[seat], play_rng, _CTX["tta"])
            acts[seat] = features.index_to_action(idx)
        if st.time >= _CTX["min_turn"]:
            for seat in (0, 1):
                snap = (st.copy(), [m.copy() for m in memories], seat)
                seen = _offer(reservoir, seen, _CTX["positions_per_game"],
                              snap, play_rng)
        if engine.step(st, acts[0], acts[1]):
            break

    return [_position(seed, snap, i) for i, snap in enumerate(reservoir)]


def _pad(rows: list[dict], key: str, shape: tuple, fill, dtype) -> np.ndarray:
    out = np.full((len(rows), *shape), fill, dtype=dtype)
    for i, row in enumerate(rows):
        value = np.asarray(row[key])
        slices = (i, *[slice(0, n) for n in value.shape])
        out[slices] = value
    return out


def _cluster_normal(values: list[float], seeds: list[int]) -> tuple[float, list[float]]:
    ids = sorted(set(seeds))
    cluster = np.asarray([
        np.mean([v for v, s in zip(values, seeds) if s == seed]) for seed in ids
    ], np.float64)
    mean = float(cluster.mean())
    if len(cluster) < 2:
        return mean, [float("-inf"), float("inf")]
    se = float(cluster.std(ddof=1) / math.sqrt(len(cluster)))
    return mean, [mean - 1.96 * se, mean + 1.96 * se]


def _crossfit(rows: list[dict]) -> dict:
    """Choose on half the rollouts and evaluate on the untouched half."""
    n = min(int(r["returns"].shape[1]) for r in rows)
    if n < 4:
        return {}
    even = np.arange(n) % 2 == 0
    odd = ~even
    gain = {"robust": [], "greedy": []}
    change = {"robust": [], "greedy": []}
    seeds: list[int] = []
    for row in rows:
        z = np.asarray(row["returns"][:, :n], np.float64)
        for select, evaluate in ((even, odd), (odd, even)):
            slots = {"robust": robust_slot(z[:, select])[0],
                     "greedy": int(np.argmax(z[:, select].mean(axis=1)))}
            seeds.append(int(row["seed"]))
            for name, slot in slots.items():
                gain[name].append(float((z[slot, evaluate] - z[0, evaluate]).mean()))
                change[name].append(float(slot != 0))
    out = {"crossfit_rollouts_per_half": int(even.sum())}
    for name in ("robust", "greedy"):
        mean, ci = _cluster_normal(gain[name], seeds)
        out.update({f"crossfit_{name}_change_frac": float(np.mean(change[name])),
                    f"crossfit_{name}_gain": mean,
                    f"crossfit_{name}_gain_ci95_normal": ci})
    return out


def _summary(rows: list[dict], critic: bool, topk: int) -> dict:
    base = np.asarray([r["returns"][0].mean() for r in rows])
    selected = np.asarray([r["returns"][int(r["selected"])].mean() for r in rows])
    oracle = np.asarray([r["returns"].mean(axis=1).max() for r in rows])
    changed = np.asarray([int(r["selected"]) != 0 for r in rows])
    selected_lcb = np.asarray([r["lower"][int(r["selected"])] for r in rows])
    best_slots = np.asarray([
        int(np.argmax(r["returns"].mean(axis=1))) for r in rows], np.int32)
    full = np.asarray([len(r["actions"]) == topk for r in rows])
    at_boundary = np.asarray([
        best_slots[i] == len(r["actions"]) - 1 for i, r in enumerate(rows)])
    summary = {
        "schema_version": 2,
        "oracle_scope": SCOPE,
        "belief_safe": False,
        "states": len(rows),
        "robust_changes": int(changed.sum()),
        "robust_change_frac": float(changed.mean()),
        "base_return": float(base.mean()),
        "oracle_return_same_samples": float(oracle.mean()),
        "oracle_gain_optimistic": float((oracle - base).mean()),
        "oracle_best_rank_mean": float((best_slots + 1).mean()),
        "oracle_best_at_topk_boundary_frac": (
            float(at_boundary[full].mean()) if full.any() else None),
        "robust_return_same_samples": float(selected.mean()),
        "robust_gain_same_samples": float((selected - base).mean()),
        "robust_lcb_gain": float(selected_lcb.mean()),
    }
    if all("seed" in row for row in rows):
        summary.update(_crossfit(rows))
    if critic:
        good = total = 0
        for row in rows:
            g, n = pair_accuracy(row["returns"].mean(axis=1),
                                 np.nanmean(row["leaves"], axis=1))
            good += g; total += n
        summary.update(critic_pairs=total, critic_concordant=good,
                       critic_pair_accuracy=(float(good / total) if total else None))
    return summary


def selfcheck() -> None:
    z = np.asarray([[0, 0, 0, 0], [1, 1, 1, 1], [1, -1, 1, -1]], np.float32)
    slot, mean, lower = robust_slot(z)
    assert slot == 1 and mean[1] == lower[1] == 1.0
    # One rollout cannot establish a lower confidence bound.
    assert robust_slot(z[:, :1])[0] == 0
    assert pair_accuracy([0.8, 0.1, -0.5], [0.2, 0.0, -0.1]) == (3, 3)
    assert pair_accuracy([0.8, 0.1], [-0.2, 0.2]) == (0, 1)
    print("searchprobe selfcheck ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="frozen proposal/rollout policy")
    ap.add_argument("--critic", default=None,
                    help="optional standalone or phi__ matched PPO critic")
    ap.add_argument("--out", default="runs/searchprobe")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--positions-per-game", type=int, default=2)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--critic-depth", type=int, default=32,
                    help="transition depth at which to audit the critic")
    ap.add_argument("--confidence", type=float, default=2.0,
                    help="paired-SE multiplier for accepting a changed action")
    ap.add_argument("--min-gap", type=float, default=0.0,
                    help="required lower confidence bound over policy argmax")
    ap.add_argument("--min-turn", type=int, default=80)
    ap.add_argument("--max-turns", type=int, default=rules.TURN_LIMIT)
    ap.add_argument("--seed0", type=int, default=SP_SEED0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--tta", action="store_true",
                    help="use full dihedral averaging for proposal and rollouts")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    if not args.weights or not Path(args.weights).is_file():
        raise SystemExit("--weights must name an existing policy checkpoint")
    if args.critic and not Path(args.critic).is_file():
        raise SystemExit(f"--critic {args.critic}: no such file")
    if min(args.games, args.positions_per_game, args.topk, args.rollouts,
           args.critic_depth, args.workers) < 1:
        raise SystemExit("games/positions/topk/rollouts/critic-depth/workers must be positive")
    if not 0 <= args.min_turn < args.max_turns:
        raise SystemExit("need 0 <= --min-turn < --max-turns")
    if args.confidence < 0:
        raise SystemExit("--confidence must be non-negative")

    # Fail once in the parent, rather than once per spawned worker.
    Net(args.weights)
    if args.critic:
        ValueNet(args.critic)
    for name in ("OPENBLAS", "OMP", "MKL", "NUMEXPR"):
        os.environ[f"{name}_NUM_THREADS"] = "1"

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"REFUSING: output already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    worker_args = {k: getattr(args, k) for k in (
        "positions_per_game", "topk", "rollouts", "critic_depth",
        "confidence", "min_gap", "min_turn", "max_turns", "tta")}
    seeds = list(range(args.seed0, args.seed0 + args.games))
    rows: list[dict] = []
    started = time.time()
    with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context("spawn"),
            initializer=_init, initargs=(args.weights, args.critic, worker_args)) as pool:
        for i, game_rows in enumerate(pool.map(_game, seeds, chunksize=1), 1):
            rows.extend(game_rows)
            if i % 10 == 0 or i == len(seeds):
                print(f"  {i}/{len(seeds)} games, {len(rows)} states", flush=True)
    if not rows:
        raise SystemExit("probe produced no states; lower --min-turn")

    k, r = args.topk, args.rollouts
    arrays = {
        "x": np.stack([row["x"] for row in rows]),
        "seed": np.asarray([row["seed"] for row in rows], np.int64),
        "turn": np.asarray([row["turn"] for row in rows], np.int32),
        "seat": np.asarray([row["seat"] for row in rows], np.int8),
        "actions": _pad(rows, "actions", (k,), -1, np.int32),
        "legal": np.stack([row["legal"] for row in rows]),
        "prior_logp": _pad(rows, "prior_logp", (k,), -np.inf, np.float32),
        "returns": _pad(rows, "returns", (k, r), np.nan, np.float32),
        "leaves": _pad(rows, "leaves", (k, r), np.nan, np.float32),
        "selected": np.asarray([row["selected"] for row in rows], np.int16),
        "delta": _pad(rows, "delta", (k,), np.nan, np.float32),
        "lower": _pad(rows, "lower", (k,), -np.inf, np.float32),
    }
    rows_path = out / "rows.npz"
    np.savez_compressed(rows_path, **arrays)
    summary = _summary(rows, bool(args.critic), args.topk)
    summary.update({
        "weights": str(Path(args.weights).resolve()),
        "critic": str(Path(args.critic).resolve()) if args.critic else None,
        "games": args.games, "positions_per_game": args.positions_per_game,
        "topk": args.topk, "rollouts": args.rollouts,
        "critic_depth": args.critic_depth, "confidence": args.confidence,
        "min_gap": args.min_gap, "seed0": args.seed0,
        "elapsed_seconds": round(time.time() - started, 3),
        "warning": ("Each row uses one actual hidden state. Do not deploy or "
                    "hard-distil its per-row argmax without an observation-"
                    "conditioned aggregation or belief-sensitivity test."),
    })
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {rows_path} and {summary_path}")

    from tools import manifest
    artifacts = [args.weights, rows_path, summary_path]
    if args.critic:
        artifacts.append(args.critic)
    manifest.write(out / "manifest.json", command=sys.argv, artifacts=artifacts,
                   extra={"kind": "search-falsification-probe",
                          "oracle_scope": SCOPE, "args": vars(args)})


if __name__ == "__main__":
    main()
