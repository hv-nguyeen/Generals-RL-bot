"""Promotion-grade, multi-opponent evaluation with immutable manifests.

The candidate must beat the current shipped checkpoint on fresh paired boards,
retain floors against behavioural baselines, survive a decapitation-style
opponent, stay within the move budget, and produce no agent faults.

    python -m tools.evaluate --candidate runs/new.npz --reference runs/champ.npz \
        --out runs/eval/new-vs-champ --workers 12
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from arena import rating, runner
from tools import manifest

EVAL_SCHEMA_VERSION = 2


def load_suite(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != EVAL_SCHEMA_VERSION:
        raise ValueError(f"unsupported evaluation schema in {path}")
    required = {"games_per_bucket", "seed0", "elo_margin", "max_turns",
                "max_move_ms", "buckets"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"evaluation suite is missing {sorted(missing)}")
    names = [b["name"] for b in data["buckets"]]
    if len(names) != len(set(names)):
        raise ValueError("evaluation bucket names must be unique")
    return data


def _render(spec: str, candidate: Path, reference: Path) -> str:
    return spec.format(candidate=str(candidate.resolve()), reference=str(reference.resolve()))


def _bucket(name: str, spec: str, rows: list[dict], rule: dict, elo_margin: float) -> dict:
    w, d, loss = runner.tally(rows)
    s = rating.summary(w, d, loss)
    test = rating.sprt(w, d, loss, 0.0, elo_margin)
    a_faults = sum(r["faults"][r["a_seat"]] for r in rows)
    a_max_ms = max((r["max_ms"][r["a_seat"]] for r in rows), default=0.0)
    a_mean_ms = float(np.mean([r["mean_ms"][r["a_seat"]] for r in rows]))
    lower_score = rating.elo_to_score(s["elo_lo"])
    passed = (lower_score >= float(rule.get("min_lower_score", 0.0))
              and a_faults == 0)
    if rule.get("requires_sprt"):
        passed = passed and test["verdict"].startswith("accept H1")
    return {"name": name, "opponent": spec, **s, "lower_score": lower_score,
            "sprt": test, "a_faults": a_faults, "a_max_ms": a_max_ms,
            "a_mean_ms": a_mean_ms, "passed": bool(passed), "rule": rule}


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
    args = ap.parse_args()
    if not args.candidate.is_file() or not args.reference.is_file():
        raise SystemExit("candidate and reference must be existing checkpoint files")
    suite = load_suite(args.suite)
    games = int(args.games or suite["games_per_bucket"])
    if games < 2 or games % 2:
        raise SystemExit("games per bucket must be an even number >= 2")

    buckets = list(suite["buckets"])
    buckets.extend({"name": f"extra-{i}", "opponent": spec,
                    "min_lower_score": 0.5} for i, spec in enumerate(args.extra_opponent))
    candidate_spec = _render(suite.get("candidate_spec", "ship:{candidate}"),
                             args.candidate, args.reference)
    matches, rendered = [], []
    stride = max(games // 2 + 1, 10_000)
    for i, b in enumerate(buckets):
        spec = _render(b["opponent"], args.candidate, args.reference)
        rendered.append(spec)
        matches.append((candidate_spec, spec, games, int(suite["seed0"]) + i * stride))

    started = time.time()
    rows = runner.run_many(matches, workers=args.workers,
                           max_turns=int(suite["max_turns"]))
    reports = [_bucket(b["name"], spec, result, b,
                       float(suite["elo_margin"]))
               for b, spec, result in zip(buckets, rendered, rows)]
    max_ms = max((x["a_max_ms"] for x in reports), default=0.0)
    runtime_ok = max_ms <= float(suite["max_move_ms"])
    passed = runtime_ok and all(x["passed"] for x in reports)
    # A conservative aggregate for comparisons between passing candidates: the
    # weighted geometric mean of score lower bounds punishes one weak matchup.
    weights = np.asarray([float(b.get("weight", 1.0)) for b in buckets])
    lows = np.asarray([max(1e-6, x["lower_score"]) for x in reports])
    robust = float(math.exp(float(np.sum(weights * np.log(lows)) / weights.sum())))
    summary = {"schema_version": EVAL_SCHEMA_VERSION, "passed": passed,
               "runtime_passed": runtime_ok, "max_move_ms": max_ms,
               "robust_lower_score": robust, "games_per_bucket": games,
               "wall_seconds": time.time() - started, "buckets": reports}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out / "results.jsonl").open("w") as f:
        for b, result in zip(buckets, rows):
            for row in result:
                f.write(json.dumps({"bucket": b["name"], **row}) + "\n")
    manifest.write(args.out / "manifest.json", command=sys.argv,
                   artifacts=[args.candidate, args.reference, args.suite],
                   extra={"kind": "promotion-evaluation", "summary": summary})

    for r in reports:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"{mark:4} {r['name']:<14} {r['wins']}W {r['draws']}D {r['losses']}L "
              f"elo {r['elo']:+.1f} [{r['elo_lo']:+.1f}, {r['elo_hi']:+.1f}] "
              f"max {r['a_max_ms']:.1f} ms faults {r['a_faults']}")
    print(f"{'PASS' if passed else 'FAIL'} promotion; robust lower score {robust:.3f}; "
          f"max move {max_ms:.1f}/{suite['max_move_ms']} ms")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
