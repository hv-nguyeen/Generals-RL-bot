"""Promotion-grade, multi-opponent evaluation with immutable manifests.

The candidate must beat the current shipped checkpoint on fresh paired boards,
retain floors against behavioural baselines, survive a decapitation-style
opponent, stay within the move budget, and produce no agent faults.

    python -m tools.evaluate --candidate runs/new.npz --reference runs/champ.npz \
        --out runs/eval/new-vs-champ --workers 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from arena import rating, runner
from tools import manifest

SUITE_SCHEMA_VERSION = 5
EVAL_RESULT_SCHEMA_VERSION = 6


def approval_status(suite_passed: bool, operational_passed: bool,
                    smoke: bool) -> tuple[bool, bool]:
    """(promotion approved, process success); smoke can never approve."""
    return (bool(suite_passed and not smoke),
            bool(operational_passed if smoke else suite_passed))


def load_suite(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != SUITE_SCHEMA_VERSION:
        raise ValueError(f"unsupported evaluation schema in {path}")
    required = {"seed0", "confirmation_seed0", "attempt_stride", "max_attempts",
                "elo_margin", "max_turns", "max_move_ms", "buckets", "families",
                "required_style_roles"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"evaluation suite is missing {sorted(missing)}")
    if int(data["max_attempts"]) < 1 or int(data["attempt_stride"]) < 1:
        raise ValueError("max_attempts and attempt_stride must be positive")
    if int(data["seed0"]) == int(data["confirmation_seed0"]):
        raise ValueError("development and confirmation seed blocks must differ")
    names = [b["name"] for b in data["buckets"]]
    if len(names) != len(set(names)):
        raise ValueError("evaluation bucket names must be unique")
    for b in data["buckets"]:
        if "requires_sprt" in b:
            raise ValueError(f"bucket {b['name']} uses removed requires_sprt; "
                             "use requires_paired_superiority")
        games = int(b.get("games", 0))
        if games < 2 or games % 2:
            raise ValueError(f"bucket {b['name']} needs an even games >= 2")
        family = b.get("family")
        if family is not None and family not in data["families"]:
            raise ValueError(f"bucket {b['name']} names unknown family {family}")
    style_roles = [b.get("style_role") for b in data["buckets"]
                   if b.get("role") == "signal"]
    required_roles = list(data["required_style_roles"])
    if sorted(style_roles) != sorted(required_roles) or len(style_roles) != len(set(style_roles)):
        raise ValueError("signal buckets must cover each required_style_role exactly once: "
                         f"required {required_roles}, found {style_roles}")
    for name, family in data["families"].items():
        if family.get("method") != "bonferroni":
            raise ValueError(f"family {name}: only bonferroni is supported")
        if not 0.0 < float(family.get("alpha", 0.0)) < 1.0:
            raise ValueError(f"family {name}: alpha must be in (0, 1)")
    return data


def _render(spec: str, candidate: Path, reference: Path) -> str:
    return spec.format(candidate=str(candidate.resolve()), reference=str(reference.resolve()))


def _bucket(name: str, spec: str, rows: list[dict], rule: dict,
            elo_margin: float, alpha: float) -> dict:
    w, d, loss = runner.tally(rows)
    s = rating.paired_summary(rows, alpha)
    test = rating.paired_test(rows, 0.0, elo_margin, alpha)
    a_faults = sum(r["faults"][r["a_seat"]] for r in rows)
    a_max_ms = max((r["max_ms"][r["a_seat"]] for r in rows), default=0.0)
    a_mean_ms = float(np.mean([r["mean_ms"][r["a_seat"]] for r in rows]))
    lower_score = rating.elo_to_score(s["elo_lo"])
    passed = (lower_score >= float(rule.get("min_lower_score", 0.0))
              and a_faults == 0)
    if rule.get("requires_paired_superiority"):
        passed = passed and test["verdict"].startswith("accept H1")
    return {"name": name, "opponent": spec, **s, "lower_score": lower_score,
            "paired_test": test, "a_faults": a_faults, "a_max_ms": a_max_ms,
            "a_mean_ms": a_mean_ms, "passed": bool(passed), "rule": rule,
            "games": len(rows), "family": rule.get("family")}


def bucket_alphas(suite: dict, buckets: list[dict],
                  repeated_blocks: int = 1, attempt_correct: bool = False) -> list[float]:
    """Per-bucket alpha after style, confirmation, and attempt correction."""
    counts = {name: sum(b.get("family") == name for b in buckets)
              for name in suite["families"]}
    out = []
    for b in buckets:
        family = b.get("family")
        if family is None:
            alpha = float(b.get("alpha", 0.05))
        else:
            alpha = float(suite["families"][family]["alpha"]) / counts[family]
        alpha /= max(int(repeated_blocks), 1)
        if attempt_correct:
            alpha /= int(suite["max_attempts"])
        out.append(alpha)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def reserve_attempt(registry_path: Path, suite: dict, candidate: Path,
                    reference: Path, label: str = "") -> tuple[int, str]:
    """Consume a unique seed block before games start.

    Failed or interrupted evaluations remain consumed. That is intentional: a
    flaky run must not be repeated on the same boards until it happens to pass.
    """
    registry = ({"schema_version": 1, "attempts": []}
                if not registry_path.exists()
                else json.loads(registry_path.read_text()))
    if registry.get("schema_version") != 1 or not isinstance(registry.get("attempts"), list):
        raise ValueError(f"invalid attempt registry {registry_path}")
    candidate_hash, reference_hash = _sha256(candidate), _sha256(reference)
    duplicate = [a for a in registry["attempts"]
                 if a.get("candidate_sha256") == candidate_hash
                 and a.get("reference_sha256") == reference_hash]
    if duplicate:
        raise SystemExit(f"this candidate/reference pair already consumed promotion "
                         f"attempt {duplicate[-1]['attempt_id']}")
    index = len(registry["attempts"])
    if index >= int(suite["max_attempts"]):
        raise SystemExit(f"promotion registry exhausted {suite['max_attempts']} "
                         "family-corrected attempts; version a new suite")
    attempt_id = f"{index + 1:03d}-{candidate_hash[:12]}"
    registry["attempts"].append({
        "attempt_id": attempt_id, "index": index, "status": "running",
        "label": label, "candidate": str(candidate.resolve()),
        "candidate_sha256": candidate_hash,
        "reference": str(reference.resolve()),
        "reference_sha256": reference_hash,
        "reserved_unix": time.time(),
    })
    _write_json_atomic(registry_path, registry)
    return index, attempt_id


def finish_attempt(registry_path: Path, attempt_id: str, summary: dict,
                   summary_path: Path) -> None:
    registry = json.loads(registry_path.read_text())
    matches = [a for a in registry["attempts"] if a["attempt_id"] == attempt_id]
    if len(matches) != 1:
        raise ValueError(f"attempt {attempt_id} missing or duplicated in registry")
    matches[0].update(status="passed" if summary["passed"] else "failed",
                      finished_unix=time.time(), passed=bool(summary["passed"]),
                      summary=str(summary_path.resolve()))
    _write_json_atomic(registry_path, registry)


def planning_score_to_clear(games: int, alpha: float, floor: float,
                            pair_variance: float = 0.25) -> float:
    """Expected score needed for an empirical-Bernstein lower bound to clear.

    This is a planning approximation, not a gate calculation: the actual bound
    uses the observed variance of the board-pair scores.
    """
    boards = games // 2
    if boards < 2 or not 0.0 < alpha < 1.0:
        raise ValueError("planning estimate needs >=4 games and alpha in (0, 1)")
    log_term = math.log(3.0 / alpha)
    radius = (math.sqrt(2.0 * pair_variance * log_term / boards)
              + 3.0 * log_term / boards)
    return floor + radius


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--suite", type=Path, default=Path("evaluation/top3-v2.json"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--games", type=int, default=None,
                    help="override games per bucket (smoke tests only)")
    ap.add_argument("--extra-opponent", action="append", default=[],
                    help="additional agent spec; requires a non-negative lower CI")
    ap.add_argument("--registry", type=Path, default=Path("evaluation/attempts.json"),
                    help="promotion-attempt registry; full evaluations consume "
                         "one unique development/confirmation seed block")
    ap.add_argument("--attempt-label", default="")
    args = ap.parse_args()
    if not args.candidate.is_file() or not args.reference.is_file():
        raise SystemExit("candidate and reference must be existing checkpoint files")
    suite = load_suite(args.suite)
    smoke = args.games is not None
    if args.games is not None and (args.games < 2 or args.games % 2):
        raise SystemExit("games per bucket must be an even number >= 2")

    attempt_index, attempt_id = 0, None
    if not smoke:
        attempt_index, attempt_id = reserve_attempt(
            args.registry, suite, args.candidate, args.reference,
            args.attempt_label)

    buckets = list(suite["buckets"])
    buckets.extend({"name": f"extra-{i}", "opponent": spec,
                    "min_lower_score": 0.5, "family": "style",
                    "games": max(int(b["games"]) for b in suite["buckets"])}
                   for i, spec in enumerate(args.extra_opponent))
    alphas = bucket_alphas(suite, buckets, repeated_blocks=2,
                           attempt_correct=not smoke)
    candidate_spec = _render(suite.get("candidate_spec", "ship:{candidate}"),
                             args.candidate, args.reference)
    rendered = []
    stride = max(max(int(b["games"]) for b in buckets) // 2 + 1, 10_000)
    for b in buckets:
        rendered.append(_render(b["opponent"], args.candidate, args.reference))
    attempt_offset = attempt_index * int(suite["attempt_stride"])
    block_seed0 = {
        "development": int(suite["seed0"]) + attempt_offset,
        "confirmation": int(suite["confirmation_seed0"]) + attempt_offset,
    }
    matches, match_meta = [], []
    for block, base_seed in block_seed0.items():
        for i, (b, spec) in enumerate(zip(buckets, rendered)):
            games = int(args.games if args.games is not None else b["games"])
            matches.append((candidate_spec, spec, games, base_seed + i * stride))
            match_meta.append((block, b, spec))

    started = time.time()
    rows = runner.run_many(matches, workers=args.workers,
                           max_turns=int(suite["max_turns"]))
    reports_by_block = {"development": [], "confirmation": []}
    for result, match, (block, b, spec) in zip(rows, matches, match_meta):
        alpha = alphas[buckets.index(b)]
        report = _bucket(b["name"], spec, result, b,
                         float(suite["elo_margin"]), alpha)
        report.update(seed0=match[3], block=block, alpha=alpha)
        reports_by_block[block].append(report)
    reports = [r for block in reports_by_block.values() for r in block]
    max_ms = max((x["a_max_ms"] for x in reports), default=0.0)
    runtime_ok = max_ms <= float(suite["max_move_ms"])
    suite_passed = runtime_ok and all(x["passed"] for x in reports)
    operational_passed = runtime_ok and all(x["a_faults"] == 0 for x in reports)
    # A reduced --games run is useful for construction/runtime smoke only. It
    # must never mint a promotion-approved summary, even after a lucky sweep.
    passed, exit_ok = approval_status(suite_passed, operational_passed, smoke)
    # A conservative aggregate for comparisons between passing candidates: the
    # weighted geometric mean of score lower bounds punishes one weak matchup.
    weights = np.tile(
        np.asarray([float(b.get("weight", 1.0)) for b in buckets]), 2)
    lows = np.asarray([max(1e-6, x["lower_score"]) for x in reports])
    robust = float(math.exp(float(np.sum(weights * np.log(lows)) / weights.sum())))
    summary = {"schema_version": EVAL_RESULT_SCHEMA_VERSION, "passed": passed,
               "smoke": smoke, "approval_eligible": not smoke,
               "attempt_id": attempt_id, "attempt_index": attempt_index,
               "suite_thresholds_passed": suite_passed,
               "operational_passed": operational_passed,
               "runtime_passed": runtime_ok, "max_move_ms": max_ms,
               "robust_lower_score": robust,
               "bucket_games": {b["name"]: int(
                   args.games if args.games is not None else b["games"]) for b in buckets},
               "seed_blocks": block_seed0,
               "family_corrections": suite["families"],
               "max_attempts": int(suite["max_attempts"]),
               "wall_seconds": time.time() - started,
               "buckets": reports_by_block["development"],
               "confirmation_buckets": reports_by_block["confirmation"]}

    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    results_path = args.out / "results.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    with results_path.open("w") as f:
        for (block, b, _), result in zip(match_meta, rows):
            for row in result:
                f.write(json.dumps({"block": block, "bucket": b["name"], **row}) + "\n")
    manifest.write(args.out / "manifest.json", command=sys.argv,
                   artifacts=[args.candidate, args.reference, args.suite,
                              summary_path, results_path],
                   extra={"kind": "promotion-evaluation", "summary": summary})
    if attempt_id is not None:
        finish_attempt(args.registry, attempt_id, summary, summary_path)

    for r in reports:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"{mark:4} {r['block'][:4]} {r['name']:<14} "
              f"{r['wins']}W {r['draws']}D {r['losses']}L "
              f"elo {r['elo']:+.1f} [{r['elo_lo']:+.1f}, {r['elo_hi']:+.1f}] "
              f"max {r['a_max_ms']:.1f} ms faults {r['a_faults']}")
    if smoke:
        print("SMOKE ONLY — NOT PROMOTABLE; "
              f"operational {'PASS' if operational_passed else 'FAIL'}")
    else:
        print(f"{'PASS' if passed else 'FAIL'} promotion; robust lower score "
              f"{robust:.3f}; max move {max_ms:.1f}/{suite['max_move_ms']} ms")
    raise SystemExit(0 if exit_ok else 1)


if __name__ == "__main__":
    main()
