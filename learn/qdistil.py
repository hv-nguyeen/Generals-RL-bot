"""Distil a gated Q ranker into a conservative champion policy update.

The ranker never replaces the deployment policy.  It supplies bounded
advantages for the champion's top-k actions.  Those advantages tilt the full
legal champion distribution,

``target(a|h) proportional to champion(a|h) * exp(clipped(A(h,a))/beta)``,

and only the four action-head tensors are updated.  The trunk remains exactly
the accepted champion's trunk.  A second KL to the champion limits projection
error outside the probed candidates.  The sealed Q-ranker test games are never
used for distillation or epoch selection.

This command refuses a ranker whose held-out gate failed and writes a deployable
policy only when its own projection/KL gate passes.  Gameplay improvement is
still decided later by fresh paired arena games, never by this loss.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from bot import features
from bot.policy.net import Net
from learn import qrank
from tools import manifest


QDISTIL_SCHEMA = 1


def _log_softmax_np(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def soft_target_numpy(old_logits: np.ndarray, legal: np.ndarray,
                      actions: np.ndarray, q: np.ndarray, beta: float,
                      advantage_clip: float) -> np.ndarray:
    """Reference target construction used by the self-check."""
    old_logits = np.asarray(old_logits, np.float64)
    legal, actions, q = np.asarray(legal, bool), np.asarray(actions), np.asarray(q)
    bonus = np.zeros_like(old_logits)
    for row in range(len(old_logits)):
        for slot, action in enumerate(actions[row]):
            if action >= 0:
                adv = np.clip(q[row, slot] - q[row, 0],
                              -advantage_clip, advantage_clip)
                bonus[row, int(action)] += adv / beta
    return np.exp(_log_softmax_np(np.where(legal, old_logits + bonus, -1e9)))


def _meta_for(ranker: Path) -> dict:
    sidecar = ranker.with_suffix(".json")
    run_manifest = ranker.with_suffix(".manifest.json")
    if not sidecar.is_file() or not run_manifest.is_file():
        raise SystemExit(
            f"{ranker}: missing ranker gate sidecar or manifest")
    record = json.loads(run_manifest.read_text())
    recorded = {(str(a.get("sha256")), int(a.get("bytes", -1)))
                for a in record.get("artifacts", [])}
    for artifact in (ranker, sidecar):
        identity = (manifest.sha256(artifact), artifact.stat().st_size)
        if identity not in recorded:
            raise SystemExit(
                f"{artifact}: current file does not match the ranker manifest")
    meta = json.loads(sidecar.read_text())
    if meta.get("gate_passed") is not True:
        raise SystemExit(f"{ranker}: Q-ranker gate did not pass; refusing distillation")
    return meta


def _evaluate(student: dict, fixed: dict, original: dict, qhead: dict,
              data: dict, rows: np.ndarray, beta: float, advantage_clip: float,
              batch: int = 256) -> dict:
    import jax
    import jax.numpy as jnp

    @jax.jit
    def one(head, xb, ab, legal):
        h = qrank.bc.trunk(fixed, xb)
        old = qrank.head_forward(original, h)
        new = qrank.head_forward(head, h)
        qall = qrank.head_forward(qhead, h)
        safe = jnp.maximum(ab, 0)
        qc = jnp.take_along_axis(qall, safe, axis=1)
        valid = ab >= 0
        adv = jnp.clip(qc - qc[:, :1], -advantage_clip, advantage_clip)
        bonus = jnp.zeros_like(old)
        bonus = bonus.at[jnp.arange(len(xb))[:, None], safe].add(
            jnp.where(valid, adv / beta, 0.0))
        masked_old = jnp.where(legal, old, -1e9)
        old_lp = jax.nn.log_softmax(masked_old)
        target_lp = jax.nn.log_softmax(jnp.where(legal, old + bonus, -1e9))
        new_lp = jax.nn.log_softmax(jnp.where(legal, new, -1e9))
        tp, op = jnp.exp(target_lp), jnp.exp(old_lp)
        target_kl = jnp.sum(tp * (target_lp - new_lp), axis=1)
        anchor_kl = jnp.sum(op * (old_lp - new_lp), axis=1)
        source_target = jnp.sum(tp * (target_lp - old_lp), axis=1)
        old_top = jnp.argmax(masked_old, axis=1)
        target_top = jnp.argmax(jnp.where(legal, old + bonus, -1e9), axis=1)
        new_top = jnp.argmax(jnp.where(legal, new, -1e9), axis=1)
        oldc = jnp.take_along_axis(old, safe, axis=1)
        prior_gap = jnp.where(valid,
                              jnp.abs((oldc - oldc[:, :1])
                                      - (jnp.asarray(0.0))), 0.0)
        return (target_kl, anchor_kl, source_target,
                old_top, target_top, new_top, oldc, prior_gap)

    pieces = [[] for _ in range(8)]
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        got = one(student, np.asarray(data["x"][r], np.float32),
                  data["actions"][r], data["legal"][r])
        for dst, value in zip(pieces, got):
            dst.append(np.asarray(value))
    target_kl, anchor_kl, source_target, old_top, target_top, new_top, oldc, _ = (
        np.concatenate(x) for x in pieces)
    # searchprobe stores log probabilities; differences must equal the source
    # checkpoint's raw-logit differences. This catches a wrong checkpoint or an
    # encoder/temporal-memory mismatch before it becomes a policy update.
    prior = data["prior_logp"][rows]
    valid = data["actions"][rows] >= 0
    discrepancy = np.where(valid,
                           np.abs((oldc - oldc[:, :1])
                                  - (prior - prior[:, :1])), 0.0)
    changed = target_top != old_top
    return {"states": int(len(rows)),
            "source_target_kl": float(source_target.mean()),
            "target_kl": float(target_kl.mean()),
            "anchor_kl": float(anchor_kl.mean()),
            "target_change_frac": float(changed.mean()),
            "student_change_frac": float((new_top != old_top).mean()),
            "student_target_match": float((new_top == target_top).mean()),
            "changed_target_match": (float((new_top[changed] == target_top[changed]).mean())
                                     if changed.any() else 0.0),
            "prior_logit_max_error": float(discrepancy.max())}


def selfcheck() -> None:
    old = np.array([[2.0, 1.0, -3.0], [0.0, 0.0, 9.0]])
    legal = np.array([[1, 1, 0], [1, 1, 0]], bool)
    actions = np.array([[0, 1], [0, 1]])
    q = np.array([[0.0, 1.0], [0.0, -1.0]])
    target = soft_target_numpy(old, legal, actions, q, beta=0.2,
                               advantage_clip=1.0)
    assert np.allclose(target.sum(axis=1), 1.0)
    assert not target[:, 2].any(), "illegal actions must have zero target mass"
    assert target[0, 1] > target[0, 0], "positive Q advantage must overcome the prior"
    assert target[1, 0] > target[1, 1], "negative Q advantage must preserve the baseline"
    print("qdistil selfcheck OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", default=None,
                    help="the exact searchprobe data used by the ranker")
    ap.add_argument("--policy", help="accepted source policy")
    ap.add_argument("--ranker", help="gated learn.qrank checkpoint")
    ap.add_argument("--out", default="/local/data/vng205/qdistil/candidate.npz")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--anchor-weight", type=float, default=0.5)
    ap.add_argument("--advantage-clip", type=float, default=0.5)
    ap.add_argument("--max-kl", type=float, default=0.02)
    ap.add_argument("--target-ratio", type=float, default=0.75,
                    help="student target KL must be this fraction of source target KL")
    ap.add_argument("--min-change", type=float, default=0.005)
    ap.add_argument("--max-change", type=float, default=0.25)
    ap.add_argument("--min-changed-match", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if not args.data or not args.policy or not args.ranker:
        raise SystemExit("--data, --policy, and --ranker are required")
    if min(args.epochs, args.patience, args.batch) < 1 or args.lr <= 0:
        raise SystemExit("epochs/patience/batch and lr must be positive")
    if args.anchor_weight < 0 or args.advantage_clip <= 0 or args.max_kl <= 0:
        raise SystemExit("anchor-weight must be nonnegative; clip/max-kl positive")
    if not 0 < args.target_ratio < 1:
        raise SystemExit("--target-ratio must be in (0,1)")
    out = Path(args.out)
    occupied = [p for p in (out, out.with_suffix(".json"),
                             out.with_suffix(".manifest.json")) if p.exists()]
    if occupied:
        raise SystemExit(f"REFUSING: output path already exists: {occupied[0]}")

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_default_matmul_precision", "highest")
    print("devices:", jax.devices())
    ranker_path = Path(args.ranker).resolve()
    rank_meta = _meta_for(ranker_path)
    policy, arch = qrank.load_policy(args.policy)
    policy_sha = manifest.sha256(args.policy)
    fixed_np, qhead_np, source_sha, beta = qrank.load_ranker(ranker_path)
    if source_sha != policy_sha or rank_meta.get("source_policy_sha256") != policy_sha:
        raise SystemExit("ranker and --policy SHA-256 do not match")
    expected_fixed = qrank.trunk_params(policy)
    if set(fixed_np) != set(expected_fixed) or any(
            not np.array_equal(fixed_np[k], expected_fixed[k]) for k in fixed_np):
        raise SystemExit("ranker trunk is not byte-identical to the supplied policy trunk")

    data, row_files = qrank.load_dataset(args.data, args.policy, require_legal=True)
    if set(map(str, row_files)) != set(rank_meta.get("data", [])):
        raise SystemExit("--data is not the exact dataset recorded by the gated ranker")
    train_seeds = set(map(int, rank_meta["train_seeds"]))
    select_seeds = set(map(int, rank_meta["selection_seeds"]))
    test_seeds = set(map(int, rank_meta["test_seeds"]))
    if train_seeds & select_seeds or train_seeds & test_seeds or select_seeds & test_seeds:
        raise SystemExit("ranker sidecar contains overlapping game splits")
    train_rows = np.flatnonzero(np.isin(data["seed"], list(train_seeds)))
    select_rows = np.flatnonzero(np.isin(data["seed"], list(select_seeds)))
    if not len(train_rows) or not len(select_rows):
        raise SystemExit("ranker train/selection games are absent from --data")
    if np.isin(data["seed"][np.r_[train_rows, select_rows]], list(test_seeds)).any():
        raise AssertionError("sealed test games entered distillation")
    valid = data["actions"] >= 0
    safe = np.maximum(data["actions"], 0)
    if not data["legal"][np.arange(len(valid))[:, None], safe][valid].all():
        raise SystemExit("searchprobe candidate list contains an action outside its legal mask")

    fixed = {k: jnp.asarray(v) for k, v in fixed_np.items()}
    original = {k: jnp.asarray(v) for k, v in qrank.action_head(policy).items()}
    qhead = {k: jnp.asarray(v) for k, v in qhead_np.items()}
    student = {k: jnp.asarray(v) for k, v in qrank.action_head(policy).items()}
    m = {k: jnp.zeros_like(v) for k, v in student.items()}
    v = {k: jnp.zeros_like(x) for k, x in student.items()}
    counts = {int(seed): int(np.sum(data["seed"][train_rows] == seed))
              for seed in np.unique(data["seed"][train_rows])}
    row_weight = np.asarray([1.0 / counts.get(int(s), 1) for s in data["seed"]], np.float32)

    def objective(head, xb, ab, legal, wb):
        h = qrank.bc.trunk(fixed, xb)
        old = qrank.head_forward(original, h)
        new = qrank.head_forward(head, h)
        qall = qrank.head_forward(qhead, h)
        safe_actions = jnp.maximum(ab, 0)
        qc = jnp.take_along_axis(qall, safe_actions, axis=1)
        candidate = ab >= 0
        adv = jnp.clip(qc - qc[:, :1], -args.advantage_clip, args.advantage_clip)
        bonus = jnp.zeros_like(old)
        bonus = bonus.at[jnp.arange(len(xb))[:, None], safe_actions].add(
            jnp.where(candidate, adv / beta, 0.0))
        old_lp = jax.nn.log_softmax(jnp.where(legal, old, -1e9))
        target_lp = jax.nn.log_softmax(jnp.where(legal, old + bonus, -1e9))
        new_lp = jax.nn.log_softmax(jnp.where(legal, new, -1e9))
        tp, op = jnp.exp(target_lp), jnp.exp(old_lp)
        target_kl = jnp.sum(tp * (target_lp - new_lp), axis=1)
        anchor_kl = jnp.sum(op * (old_lp - new_lp), axis=1)
        loss = target_kl + args.anchor_weight * anchor_kl
        return jnp.sum(wb * loss) / jnp.maximum(jnp.sum(wb), 1e-8)

    @jax.jit
    def step(head, m, v, t, xb, ab, legal, wb):
        loss, grad = jax.value_and_grad(objective)(head, xb, ab, legal, wb)
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = {k: b1 * m[k] + (1 - b1) * grad[k] for k in head}
        v = {k: b2 * v[k] + (1 - b2) * grad[k] ** 2 for k in head}
        head = {k: head[k] - args.lr * (m[k] / (1 - b1 ** t))
                / (jnp.sqrt(v[k] / (1 - b2 ** t)) + eps) for k in head}
        return head, m, v, loss

    initial = _evaluate(student, fixed, original, qhead, data, select_rows,
                        beta, args.advantage_clip)
    if initial["prior_logit_max_error"] > 0.02:
        raise SystemExit(
            f"source checkpoint/data forward mismatch: max candidate logit error "
            f"{initial['prior_logit_max_error']:.4f} > 0.02")
    print(f"ranker beta {beta:g}; source->target KL "
          f"{initial['source_target_kl']:.5f}; target changes "
          f"{initial['target_change_frac']:.3f}")

    rng = np.random.default_rng(args.seed)
    best_score, best_head, best_metrics, stale, updates = (
        float("inf"), None, None, 0, 0)
    for epoch in range(args.epochs):
        started, losses = time.time(), []
        order = rng.permutation(train_rows)
        for i in range(0, len(order) - args.batch + 1, args.batch):
            rows = order[i:i + args.batch]
            updates += 1
            student, m, v, loss = step(
                student, m, v, updates,
                jnp.asarray(data["x"][rows].astype(np.float32)),
                jnp.asarray(data["actions"][rows]),
                jnp.asarray(data["legal"][rows]), jnp.asarray(row_weight[rows]))
            losses.append(float(loss))
        if not losses:
            raise SystemExit("no complete training batch; lower --batch or add data")
        metrics = _evaluate(student, fixed, original, qhead, data, select_rows,
                            beta, args.advantage_clip)
        score = metrics["target_kl"] + args.anchor_weight * metrics["anchor_kl"]
        marker = ""
        if score < best_score - 1e-7:
            best_score, best_head, best_metrics, stale, marker = (
                score, {k: np.asarray(v) for k, v in student.items()}, metrics, 0,
                "  <- kept")
        else:
            stale += 1
        print(f"epoch {epoch:2d} loss {np.mean(losses):.5f}  "
              f"targetKL {metrics['target_kl']:.5f}  "
              f"anchorKL {metrics['anchor_kl']:.5f}  "
              f"change {metrics['student_change_frac']:.3f}  "
              f"changed-match {metrics['changed_target_match']:.3f}  "
              f"{time.time() - started:.0f}s{marker}", flush=True)
        if stale >= args.patience:
            print(f"early stop: {args.patience} epochs without projection improvement")
            break

    if best_head is None or best_metrics is None:
        raise AssertionError("distillation produced no selected checkpoint")
    gate_passed = bool(best_metrics["source_target_kl"] > 1e-6
                       and best_metrics["target_kl"]
                       <= args.target_ratio * best_metrics["source_target_kl"]
                       and best_metrics["anchor_kl"] <= args.max_kl
                       and args.min_change <= best_metrics["student_change_frac"]
                       <= args.max_change
                       and best_metrics["changed_target_match"]
                       >= args.min_changed_match
                       and best_metrics["prior_logit_max_error"] <= 0.02)

    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": QDISTIL_SCHEMA, "source_policy": str(Path(args.policy).resolve()),
              "source_policy_sha256": policy_sha, "ranker": str(ranker_path),
              "ranker_sha256": manifest.sha256(ranker_path), "beta": beta,
              "advantage_clip": args.advantage_clip, "anchor_weight": args.anchor_weight,
              "validation_metrics": best_metrics, "gate_passed": gate_passed,
              "sealed_q_test_seeds_used": False, **arch}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    artifacts: list[str | Path] = [args.policy, ranker_path, *row_files]
    if gate_passed:
        candidate = {**{k: np.asarray(v) for k, v in fixed.items()}, **best_head}
        qrank.bc.save(candidate, out)
        Net(str(out))
        artifacts.append(out)
    artifacts.append(out.with_suffix(".json"))
    manifest.write(out.with_suffix(".manifest.json"), command=sys.argv,
                   artifacts=artifacts,
                   extra={"kind": "qrank-soft-policy-distillation",
                          "gate_passed": gate_passed, "args": vars(args)})
    if not gate_passed:
        print("DISTILLATION GATE FAILED: no deployable policy was written.")
        print(json.dumps(best_metrics, indent=2))
        raise SystemExit(1)
    print(f"DISTILLATION GATE PASSED: wrote candidate policy {out}")
    print(json.dumps(best_metrics, indent=2))
    print("This authorizes a fresh paired arena pilot, not promotion.")


if __name__ == "__main__":
    main()
