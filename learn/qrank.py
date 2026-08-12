"""Fit an observation-conditioned action ranker on searchprobe rollouts.

This is the aggregation step that ``tools.searchprobe`` deliberately does not
perform.  A searchprobe row contains returns from one sampled hidden state.  We
never turn that row's argmax into a label.  Instead, a small action head sees
only the encoded observation/history and is trained on paired return
differences across many independently generated games.  The champion trunk is
frozen, so the pilot asks one narrow question: can its existing representation
predict which nearby action has higher on-policy continuation value?

Complete game seeds are split once, with a fixed split seed, into train,
selection, and sealed test partitions.  Epoch and KL temperature are selected
without touching the test games.  The test gate is the cluster-bootstrap lower
confidence bound of return gain over the champion's policy argmax.  A failed
gate still writes an auditable *ranker* checkpoint, but ``learn.qdistil``
refuses to consume it and no deployable policy is produced.

Example::

    python -m learn.qrank \
      --data /local/data/vng205/qdata/rows.npz \
      --policy /home/vng205/top3-v2-shared/weights/selfplay-champion-gen1.npz \
      --out /local/data/vng205/qrank/qrank.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from bot import features
from bot.policy.net import arch_of
from learn import train as bc
from learn.netoracle import policy_keys
from tools import manifest


QRANK_SCHEMA = 1
SPLIT_SEED = 20260812
BETA_GRID = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
BOOTSTRAPS = 20_000


def _row_files(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            p = p / "rows.npz"
        if not p.is_file():
            raise SystemExit(f"searchprobe rows not found: {p}")
        out.append(p.resolve())
    if len(set(out)) != len(out):
        raise SystemExit("the same searchprobe rows file was supplied more than once")
    return out


def _verify_probe_source(rows: Path, policy_sha256: str) -> None:
    """Require the probe manifest to contain the exact policy being ranked."""
    summary_path, manifest_path = rows.parent / "summary.json", rows.parent / "manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise SystemExit(
            f"{rows.parent} needs summary.json and manifest.json; provenance is a gate")
    summary = json.loads(summary_path.read_text())
    if summary.get("oracle_scope") != "single_actual_hidden_state":
        raise SystemExit(f"{summary_path}: unexpected oracle scope")
    record = json.loads(manifest_path.read_text())
    hashes = {str(a.get("sha256")) for a in record.get("artifacts", [])}
    if policy_sha256 not in hashes:
        raise SystemExit(
            f"{rows}: its manifest does not contain the supplied policy SHA-256 "
            f"{policy_sha256}; refusing a mismatched teacher")


def load_policy(path: str | Path) -> tuple[dict, dict]:
    z = np.load(path)
    arch = arch_of(z)
    keys = policy_keys(arch)
    missing = keys - set(z.files)
    if missing:
        raise SystemExit(f"{path}: policy is missing {sorted(missing)}")
    params = {k: np.asarray(z[k], np.float32) for k in keys}
    if params["conv0_w"].shape[1] != features.C:
        raise SystemExit(
            f"{path}: stem takes {params['conv0_w'].shape[1]} channels, Q data uses "
            f"the current {features.C}-channel observation; migrate the policy first")
    return params, arch


def load_dataset(paths: list[str], policy: str | Path,
                 require_legal: bool = False) -> tuple[dict, list[Path]]:
    files = _row_files(paths)
    policy_hash = manifest.sha256(policy)
    blocks, used_seeds = [], set()
    required = {"x", "seed", "actions", "prior_logp", "returns"}
    if require_legal:
        required.add("legal")
    for path in files:
        _verify_probe_source(path, policy_hash)
        z = np.load(path)
        missing = required - set(z.files)
        if missing:
            extra = (" Rebuild the probe with this source tree."
                     if "legal" in missing else "")
            raise SystemExit(f"{path}: missing {sorted(missing)}.{extra}")
        block = {k: np.asarray(z[k]) for k in required}
        n = len(block["seed"])
        if any(len(v) != n for v in block.values()):
            raise SystemExit(f"{path}: row arrays have inconsistent lengths")
        if block["x"].ndim != 4 or block["x"].shape[1:] != (
                features.C, features.PAD, features.PAD):
            raise SystemExit(f"{path}: x has incompatible shape {block['x'].shape}")
        if block["actions"].ndim != 2 or block["returns"].ndim != 3:
            raise SystemExit(f"{path}: actions/returns need shapes (N,K)/(N,K,R)")
        if block["returns"].shape[:2] != block["actions"].shape:
            raise SystemExit(f"{path}: action and return slots disagree")
        if block["prior_logp"].shape != block["actions"].shape:
            raise SystemExit(f"{path}: prior_logp and actions disagree")
        valid = block["actions"] >= 0
        if not valid[:, 0].all():
            raise SystemExit(f"{path}: every row needs policy argmax in slot zero")
        if ((block["actions"][valid] >= features.N_ACTIONS).any()
                or not np.isfinite(block["prior_logp"][valid]).all()
                or not np.isfinite(block["returns"][valid]).all()):
            raise SystemExit(f"{path}: valid action slots contain invalid values")
        seeds = set(map(int, np.unique(block["seed"])))
        overlap = used_seeds & seeds
        if overlap:
            raise SystemExit(
                f"probe seed blocks overlap (first duplicate {min(overlap)}); "
                "duplicated games are not additional evidence")
        used_seeds |= seeds
        blocks.append(block)

    keys = blocks[0]
    for block in blocks[1:]:
        for key in ("actions", "prior_logp", "returns"):
            if block[key].shape[1:] != blocks[0][key].shape[1:]:
                raise SystemExit(f"probe files disagree on {key} shape")
    data = {k: np.concatenate([b[k] for b in blocks]) for k in keys}
    data["x"] = features.ensure_channels(data["x"]).astype(np.float16, copy=False)
    data["seed"] = data["seed"].astype(np.int64, copy=False)
    data["actions"] = data["actions"].astype(np.int32, copy=False)
    data["prior_logp"] = data["prior_logp"].astype(np.float32, copy=False)
    data["returns"] = data["returns"].astype(np.float32, copy=False)
    if "legal" in data:
        if data["legal"].shape != (len(data["seed"]), features.N_ACTIONS):
            raise SystemExit(f"legal mask has incompatible shape {data['legal'].shape}")
        data["legal"] = data["legal"].astype(bool, copy=False)
    return data, files


def split_games(seeds: np.ndarray, select_frac: float, test_frac: float,
                split_seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed complete-game train/selection/test split, returned as row masks."""
    if not (0.05 <= select_frac <= 0.4 and 0.05 <= test_frac <= 0.4
            and select_frac + test_frac <= 0.6):
        raise ValueError("select/test fractions must each be in [0.05,0.4] and sum <= 0.6")
    games = np.asarray(sorted(set(map(int, seeds))), np.int64)
    if len(games) < 30:
        raise ValueError(f"Q ranking needs at least 30 independent games, found {len(games)}")
    order = np.random.default_rng(split_seed).permutation(games)
    ntest = max(2, int(round(len(games) * test_frac)))
    nselect = max(2, int(round(len(games) * select_frac)))
    test, select = set(map(int, order[:ntest])), set(map(int, order[ntest:ntest + nselect]))
    train = set(map(int, order[ntest + nselect:]))
    if not train or train & select or train & test or select & test:
        raise AssertionError("game split is empty or overlapping")
    return (np.isin(seeds, list(train)), np.isin(seeds, list(select)),
            np.isin(seeds, list(test)))


def head_forward(head: dict, h):
    """Action logits/values from an already-computed policy feature map."""
    import jax.numpy as jnp

    move = bc._conv(h, head["head_w"], head["head_b"])
    flat = jnp.transpose(move, (0, 2, 3, 1)).reshape(h.shape[0], -1)
    passed = h.mean(axis=(2, 3)) @ head["pass_w"] + head["pass_b"]
    return jnp.concatenate([flat, passed[:, None]], axis=1)


def action_head(params: dict) -> dict:
    return {k: params[k] for k in ("head_w", "head_b", "pass_w", "pass_b")}


def trunk_params(params: dict) -> dict:
    return {k: v for k, v in params.items()
            if k not in {"head_w", "head_b", "pass_w", "pass_b"}}


def init_q_head(policy: dict, xp=np) -> dict:
    return {"head_w": xp.zeros_like(policy["head_w"]),
            "head_b": xp.zeros_like(policy["head_b"]),
            "pass_w": xp.zeros_like(policy["pass_w"]),
            "pass_b": xp.zeros_like(policy["pass_b"])}


def _cluster_values(values: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    return np.asarray([np.mean(values[seeds == seed])
                       for seed in sorted(set(map(int, seeds)))], np.float64)


def cluster_interval(values: np.ndarray, seeds: np.ndarray, bootstraps: int = 0,
                     rng_seed: int = SPLIT_SEED) -> tuple[float, float, float]:
    cluster = _cluster_values(np.asarray(values, np.float64), np.asarray(seeds))
    mean = float(cluster.mean())
    if len(cluster) < 2:
        return mean, float("-inf"), float("inf")
    if bootstraps:
        rng = np.random.default_rng(rng_seed)
        draws = rng.integers(0, len(cluster), size=(bootstraps, len(cluster)))
        sampled = cluster[draws].mean(axis=1)
        lo, hi = np.quantile(sampled, [0.025, 0.975])
    else:
        se = cluster.std(ddof=1) / math.sqrt(len(cluster))
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
    return mean, float(lo), float(hi)


def pair_accuracy(target: np.ndarray, estimate: np.ndarray, valid: np.ndarray,
                  min_gap: float = 0.05) -> tuple[int, int]:
    good = total = 0
    for row in range(len(target)):
        slots = np.flatnonzero(valid[row])
        for ii, i in enumerate(slots):
            for j in slots[ii + 1:]:
                delta = float(target[row, i] - target[row, j])
                if abs(delta) < min_gap:
                    continue
                total += 1
                good += int(np.sign(delta) == np.sign(float(estimate[row, i]
                                                         - estimate[row, j])))
    return good, total


def candidate_metrics(q: np.ndarray, data: dict, rows: np.ndarray, beta: float,
                      bootstraps: int = 0) -> dict:
    actions = data["actions"][rows]
    valid = actions >= 0
    prior = data["prior_logp"][rows]
    target = np.where(valid, data["returns"][rows].mean(axis=2), 0.0)
    score = np.where(valid, prior + (q[rows] - q[rows, :1]) / float(beta), -np.inf)
    selected = np.argmax(score, axis=1)
    rr = data["returns"][rows]
    gain = rr[np.arange(len(rows)), selected].mean(axis=1) - rr[:, 0].mean(axis=1)
    seeds = data["seed"][rows]
    mean, lo, hi = cluster_interval(gain, seeds, bootstraps)
    good, pairs = pair_accuracy(target, q[rows], valid)
    return {"beta": float(beta), "games": len(set(map(int, seeds))),
            "states": int(len(rows)), "change_frac": float(np.mean(selected != 0)),
            "gain": mean, "gain_ci95": [lo, hi],
            "pair_accuracy": float(good / pairs) if pairs else None,
            "pair_concordant": int(good), "pairs": int(pairs)}


def _predict(qhead: dict, fixed: dict, x: np.ndarray, actions: np.ndarray,
             batch: int = 512) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    @jax.jit
    def one(xb, ab):
        h = bc.trunk(fixed, xb)
        logits = head_forward(qhead, h)
        safe = jnp.maximum(ab, 0)
        got = jnp.take_along_axis(logits, safe, axis=1)
        return jnp.where(ab >= 0, got, 0.0)

    out = []
    for i in range(0, len(x), batch):
        out.append(np.asarray(one(jnp.asarray(x[i:i + batch].astype(np.float32)),
                                  jnp.asarray(actions[i:i + batch]))))
    return np.concatenate(out)


def _save_ranker(path: Path, fixed: dict, qhead: dict, source_sha: str,
                 beta: float) -> None:
    arrays = {f"trunk__{k}": np.asarray(v) for k, v in fixed.items()}
    arrays.update({f"q__{k}": np.asarray(v) for k, v in qhead.items()})
    np.savez_compressed(path, **arrays, qrank_schema=np.int16(QRANK_SCHEMA),
                        source_policy_sha256=np.array(source_sha),
                        beta=np.float32(beta))


def load_ranker(path: str | Path) -> tuple[dict, dict, str, float]:
    z = np.load(path)
    if int(z["qrank_schema"]) != QRANK_SCHEMA:
        raise SystemExit(f"{path}: unsupported Q-ranker schema")
    fixed = {k[7:]: np.asarray(z[k], np.float32)
             for k in z.files if k.startswith("trunk__")}
    qhead = {k[3:]: np.asarray(z[k], np.float32)
             for k in z.files if k.startswith("q__")}
    if set(qhead) != {"head_w", "head_b", "pass_w", "pass_b"}:
        raise SystemExit(f"{path}: incomplete Q head")
    arch_of(fixed)
    if qhead["head_w"].shape[0] != features.PER_CELL:
        raise SystemExit(f"{path}: Q head uses a different action schema")
    return fixed, qhead, str(z["source_policy_sha256"]), float(z["beta"])


def selfcheck() -> None:
    seeds = np.repeat(np.arange(40), 2)
    train, select, test = split_games(seeds, 0.2, 0.2)
    assert train.any() and select.any() and test.any()
    assert not np.any(train & select) and not np.any(train & test)
    assert np.array_equal(split_games(seeds, 0.2, 0.2)[0], train)

    # Eight deterministic game clusters: slot 1 is better and a ranker that
    # identifies it must have a strictly positive bootstrap lower bound.
    n = 16
    data = {"actions": np.tile(np.array([[3, 7]], np.int32), (n, 1)),
            "prior_logp": np.tile(np.array([[-0.1, -0.2]], np.float32), (n, 1)),
            "returns": np.tile(np.array([[[0, 0], [1, 1]]], np.float32), (n, 1, 1)),
            "seed": np.repeat(np.arange(8), 2)}
    q = np.tile(np.array([[0.0, 1.0]], np.float32), (n, 1))
    met = candidate_metrics(q, data, np.arange(n), beta=0.2, bootstraps=1000)
    assert met["change_frac"] == 1.0 and met["gain"] == 1.0
    assert met["gain_ci95"][0] == 1.0 and met["pair_accuracy"] == 1.0
    print("qrank selfcheck OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", default=None,
                    help="searchprobe rows.npz or its directory; repeat for seed blocks")
    ap.add_argument("--policy", help="exact frozen policy used by every probe")
    ap.add_argument("--out", default="/local/data/vng205/qrank/qrank.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--select-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--min-test-gain", type=float, default=0.01,
                    help="minimum sealed-test mean return gain")
    ap.add_argument("--min-change", type=float, default=0.01)
    ap.add_argument("--max-change", type=float, default=0.25)
    ap.add_argument("--min-pair-accuracy", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=0,
                    help="optimizer/order seed; the game split is fixed")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if not args.data or not args.policy:
        raise SystemExit("--data and --policy are required")
    if min(args.epochs, args.patience, args.batch) < 1 or args.lr <= 0:
        raise SystemExit("epochs/patience/batch and lr must be positive")
    if not 0 <= args.min_change < args.max_change <= 1:
        raise SystemExit("need 0 <= --min-change < --max-change <= 1")
    out = Path(args.out)
    occupied = [p for p in (out, out.with_suffix(".json"),
                             out.with_suffix(".manifest.json")) if p.exists()]
    if occupied:
        raise SystemExit(f"REFUSING: output path already exists: {occupied[0]}")

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_default_matmul_precision", "highest")
    print("devices:", jax.devices())
    policy, arch = load_policy(args.policy)
    data, row_files = load_dataset(args.data, args.policy)
    train_mask, select_mask, test_mask = split_games(
        data["seed"], args.select_frac, args.test_frac)
    train_rows, select_rows, test_rows = map(np.flatnonzero,
                                             (train_mask, select_mask, test_mask))
    print(f"{len(set(map(int, data['seed'])))} games/{len(data['seed'])} states: "
          f"train {len(set(map(int, data['seed'][train_rows])))}, "
          f"select {len(set(map(int, data['seed'][select_rows])))}, "
          f"sealed test {len(set(map(int, data['seed'][test_rows])))}")

    fixed = {k: jnp.asarray(v) for k, v in trunk_params(policy).items()}
    qhead = {k: jnp.asarray(v) for k, v in init_q_head(policy).items()}
    m = {k: jnp.zeros_like(v) for k, v in qhead.items()}
    v = {k: jnp.zeros_like(x) for k, x in qhead.items()}
    target = np.where(data["actions"] >= 0, data["returns"].mean(axis=2),
                      0.0).astype(np.float32)
    counts = {int(seed): int(np.sum(data["seed"][train_rows] == seed))
              for seed in np.unique(data["seed"][train_rows])}
    # Only train rows consume these weights. Held-out games deliberately have
    # no training count, but keeping one aligned array makes minibatch gathers
    # simple without ever assigning them gradient weight.
    row_weight = np.asarray([1.0 / counts.get(int(s), 1)
                             for s in data["seed"]], np.float32)

    def objective(head, xb, ab, yb, wb):
        h = bc.trunk(fixed, xb)
        all_q = head_forward(head, h)
        safe = jnp.maximum(ab, 0)
        pred = jnp.take_along_axis(all_q, safe, axis=1)
        valid = ab >= 0
        error = (pred[:, :, None] - pred[:, None, :]
                 - (yb[:, :, None] - yb[:, None, :]))
        pair = (valid[:, :, None] & valid[:, None, :]
                & jnp.triu(jnp.ones(error.shape[1:], dtype=bool), 1)[None])
        ae = jnp.abs(error)
        huber = jnp.where(ae < 0.5, 0.5 * ae ** 2, 0.5 * (ae - 0.25))
        weight = pair * wb[:, None, None]
        return jnp.sum(weight * huber) / jnp.maximum(jnp.sum(weight), 1e-8)

    @jax.jit
    def step(head, m, v, t, lr, xb, ab, yb, wb):
        loss, grad = jax.value_and_grad(objective)(head, xb, ab, yb, wb)
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = {k: b1 * m[k] + (1 - b1) * grad[k] for k in head}
        v = {k: b2 * v[k] + (1 - b2) * grad[k] ** 2 for k in head}
        head = {k: head[k] - lr * (m[k] / (1 - b1 ** t))
                / (jnp.sqrt(v[k] / (1 - b2 ** t)) + eps) for k in head}
        return head, m, v, loss

    rng = np.random.default_rng(args.seed)
    best_score, best_head, best_selection, stale, updates = (
        -np.inf, None, None, 0, 0)
    for epoch in range(args.epochs):
        started, losses = time.time(), []
        order = rng.permutation(train_rows)
        for i in range(0, len(order) - args.batch + 1, args.batch):
            rows = order[i:i + args.batch]
            updates += 1
            qhead, m, v, loss = step(
                qhead, m, v, updates, args.lr,
                jnp.asarray(data["x"][rows].astype(np.float32)),
                jnp.asarray(data["actions"][rows]), jnp.asarray(target[rows]),
                jnp.asarray(row_weight[rows]))
            losses.append(float(loss))
        if not losses:
            raise SystemExit("no complete training batch; lower --batch or add data")
        # Do not even forward the sealed-test observations during model/epoch
        # selection. Only selection rows are materialized into predictions.
        q = np.zeros_like(data["prior_logp"], dtype=np.float32)
        q[select_rows] = _predict(qhead, fixed, data["x"][select_rows],
                                  data["actions"][select_rows])
        choices = [candidate_metrics(q, data, select_rows, beta) for beta in BETA_GRID]
        chosen = max(choices, key=lambda x: (x["gain_ci95"][0], x["gain"], x["beta"]))
        score = float(chosen["gain_ci95"][0])
        marker = ""
        if score > best_score + 1e-6:
            best_score, best_head, best_selection, stale, marker = (
                score, {k: np.asarray(v) for k, v in qhead.items()}, chosen, 0,
                "  <- kept")
        else:
            stale += 1
        print(f"epoch {epoch:2d} loss {np.mean(losses):.4f}  "
              f"select beta {chosen['beta']:.2g} change {chosen['change_frac']:.3f}  "
              f"gain {chosen['gain']:+.4f} normal95 "
              f"[{chosen['gain_ci95'][0]:+.4f},{chosen['gain_ci95'][1]:+.4f}]  "
              f"pair {chosen['pair_accuracy']:.3f}  "
              f"{time.time() - started:.0f}s{marker}", flush=True)
        if stale >= args.patience:
            print(f"early stop: {args.patience} epochs without selection improvement")
            break

    if best_head is None or best_selection is None:
        raise AssertionError("training produced no selected checkpoint")
    selection_passed = bool(best_selection["gain_ci95"][0] > 0
                            and args.min_change <= best_selection["change_frac"]
                            <= args.max_change
                            and (best_selection["pair_accuracy"] or 0)
                            >= args.min_pair_accuracy)
    test_metrics = None
    if selection_passed:
        # First and only touch of the sealed test games.
        q = np.zeros_like(data["prior_logp"], dtype=np.float32)
        q[test_rows] = _predict(
            {k: jnp.asarray(v) for k, v in best_head.items()}, fixed,
            data["x"][test_rows], data["actions"][test_rows])
        test_metrics = candidate_metrics(
            q, data, test_rows, best_selection["beta"], BOOTSTRAPS)
        gate_passed = bool(test_metrics["gain"] >= args.min_test_gain
                           and test_metrics["gain_ci95"][0] > 0
                           and args.min_change <= test_metrics["change_frac"]
                           <= args.max_change
                           and (test_metrics["pair_accuracy"] or 0)
                           >= args.min_pair_accuracy)
    else:
        gate_passed = False

    out.parent.mkdir(parents=True, exist_ok=True)
    source_sha = manifest.sha256(args.policy)
    _save_ranker(out, {k: np.asarray(v) for k, v in fixed.items()}, best_head,
                 source_sha, best_selection["beta"])
    game_list = lambda mask: sorted(set(map(int, data["seed"][mask])))  # noqa: E731
    meta = {"schema_version": QRANK_SCHEMA, "source_policy": str(Path(args.policy).resolve()),
            "source_policy_sha256": source_sha, "data": list(map(str, row_files)),
            "split_seed": SPLIT_SEED, "train_seeds": game_list(train_mask),
            "selection_seeds": game_list(select_mask), "test_seeds": game_list(test_mask),
            "selection_metrics": best_selection, "selection_passed": selection_passed,
            "test_metrics": test_metrics, "gate_passed": gate_passed,
            "gate": {"min_test_gain": args.min_test_gain,
                     "min_change": args.min_change, "max_change": args.max_change,
                     "min_pair_accuracy": args.min_pair_accuracy}, **arch}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    manifest.write(out.with_suffix(".manifest.json"), command=sys.argv,
                   artifacts=[args.policy, *row_files, out, out.with_suffix(".json")],
                   extra={"kind": "observation-conditioned-q-ranker",
                          "gate_passed": gate_passed, "args": vars(args)})
    print(f"\nwrote auditable ranker {out}")
    print("selection:", json.dumps(best_selection, indent=2))
    if test_metrics is not None:
        print("sealed test:", json.dumps(test_metrics, indent=2))
    if not gate_passed:
        print("GATE FAILED: do not distil this ranker into a policy.")
        raise SystemExit(1)
    print("Q-RANKER GATE PASSED: soft head-only distillation is authorized.")


if __name__ == "__main__":
    main()
