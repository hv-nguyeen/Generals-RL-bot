"""Cross-entropy-method search over the config vector.

Heuristic bots live or die on their constants, and hand-tuning them is guesswork
with a 60-game standard error. This searches them instead. It is the piece of
the repo that wants a big machine: every candidate is independent, so throughput
scales with cores.

    python -m tools.tune --out runs/tune1 --iters 20 --pop 12 --games 40 \
        --opponents greedy,ours --workers 60

Design notes that matter for the result being real:

* **Common random numbers.** Every candidate in an iteration plays the *same*
  seeds, so candidates are compared on identical boards. This removes most of
  the between-candidate variance and is worth more than doubling the game count.
* **Colours swapped** inside `run_match`, so seat advantage cannot leak in.
* **Seeds rotate between iterations**, so the search cannot overfit one pool.
* **Resumable**: state is checkpointed each iteration, so a preempted job on a
  cluster picks up where it stopped.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from arena.runner import run_match, tally
from bot.config import Config

# Searching every knob at once wastes samples on things that barely matter.
# These are the ones with real leverage; pass --params to override.
DEFAULT_GROUPS = {
    "opening": ["first_expand_turn", "expand.cap_neutral", "expand.progress",
                "expand.army", "expand.from_general", "expand.frontier",
                "expand.toward_enemy", "expand.stack_break", "expand.reveal"],
    "castle": ["castle_min_turn", "castle_gather_min_army", "castle_safe_dist",
               "max_castles", "castle_min_land", "gather.progress", "gather.army"],
    "combat": ["defend_margin", "defend_horizon", "defend_hold", "attack_margin",
               "attack_margin_unsure", "attack_defense_frac", "probe_turn",
               "gather_start_turn", "gather_army_ratio", "expand_floor"],
}
def _int_params() -> set[str]:
    cfg = Config()
    names, _ = cfg.flatten()
    return {n for n in names if "." not in n and isinstance(getattr(cfg, n), int)}


INT_PARAMS = _int_params()


def evaluate(cfg: Config, opponents: list[str], games: int, seed0: int, workers: int,
             max_turns: int, tmpdir: Path, tag: str, maps: str | None = None) -> tuple[float, dict]:
    path = tmpdir / f"cand_{tag}.json"
    cfg.save(path)
    spec = f"ours:{path}"
    scores, detail = [], {}
    for opp in opponents:
        results = run_match(spec, opp, games, seed0, workers, max_turns, maps=maps)
        w, d, loss = tally(results)
        score = (w + 0.5 * d) / max(len(results), 1)
        scores.append(score)
        detail[opp] = {"w": w, "d": d, "l": loss, "score": round(score, 4)}
    return float(np.mean(scores)), detail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=None, help="starting config json")
    ap.add_argument("--out", default="runs/tune")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--games", type=int, default=40, help="games per opponent per candidate")
    ap.add_argument("--opponents", default="ours,hunter")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=900)
    ap.add_argument("--groups", default=",".join(DEFAULT_GROUPS),
                    help="which parameter groups to search")
    ap.add_argument("--params", default=None, help="explicit comma-separated parameter list")
    ap.add_argument("--sigma", type=float, default=0.45, help="initial relative spread")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maps", default=None, help="real-board pool from `analysis.official maps`")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmpdir = out / "candidates"
    tmpdir.mkdir(exist_ok=True)
    opponents = [s for s in args.opponents.split(",") if s]

    base = Config.load(args.base) if args.base else Config()
    all_names, all_vals = base.flatten()
    lookup = dict(zip(all_names, all_vals))

    if args.params:
        names = [p for p in args.params.split(",") if p]
    else:
        names = []
        for g in args.groups.split(","):
            names += DEFAULT_GROUPS.get(g.strip(), [])
    names = [n for n in dict.fromkeys(names) if n in lookup]
    if not names:
        raise SystemExit("no parameters selected")

    mean = np.array([lookup[n] for n in names], dtype=float)
    sigma = np.abs(mean) * args.sigma + 1.0
    rng = np.random.default_rng(args.seed)

    ckpt = out / "state.json"
    start_iter = 0
    if ckpt.exists():
        state = json.loads(ckpt.read_text())
        if state["names"] == names:
            mean = np.array(state["mean"])
            sigma = np.array(state["sigma"])
            start_iter = state["iter"]
            print(f"resuming from iteration {start_iter}")

    log = (out / "log.jsonl").open("a")
    base_score, base_detail = evaluate(base, opponents, args.games, args.seed * 1000,
                                       args.workers, args.max_turns, tmpdir, "base", args.maps)
    print(f"baseline score {base_score:.3f}  {base_detail}")
    best_score, best_cfg = base_score, base

    for it in range(start_iter, args.iters):
        t0 = time.time()
        seed0 = 10_000 + it * args.games   # rotate the pool, but share it within the iteration
        samples = rng.normal(mean, sigma, size=(args.pop, len(names)))
        scored = []
        for k, vec in enumerate(samples):
            vals = [round(v) if n in INT_PARAMS else float(v) for n, v in zip(names, vec)]
            cfg = base.with_vector(names, vals)
            score, detail = evaluate(cfg, opponents, args.games, seed0, args.workers,
                                     args.max_turns, tmpdir, f"{it}_{k}", args.maps)
            scored.append((score, vec, cfg, detail))
            print(f"  it{it:02d} cand{k:02d}  {score:.3f}  {detail}")

        scored.sort(key=lambda x: -x[0])
        elite = np.array([s[1] for s in scored[:args.elite]])
        mean = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), sigma * 0.3)   # keep some exploration

        top_score, _, top_cfg, top_detail = scored[0]
        if top_score > best_score:
            best_score, best_cfg = top_score, top_cfg
            best_cfg.save(out / "best.json")

        row = {"iter": it, "best_in_iter": top_score, "best_overall": best_score,
               "detail": top_detail, "mean": dict(zip(names, mean.round(3).tolist())),
               "secs": round(time.time() - t0, 1)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        ckpt.write_text(json.dumps({"names": names, "mean": mean.tolist(),
                                    "sigma": sigma.tolist(), "iter": it + 1,
                                    "best": best_score}, indent=2))
        print(f"it{it:02d} best {top_score:.3f}  overall {best_score:.3f}  "
              f"({row['secs']}s)")

    (out / "best.json").write_text(json.dumps(asdict(best_cfg), indent=2) + "\n")
    print(f"\nbest {best_score:.3f} (baseline {base_score:.3f}) -> {out / 'best.json'}")
    print("Confirm it for real with an SPRT run:")
    print(f"  python -m arena.runner --a ours:{out / 'best.json'} --b ours --games 400 "
          f"--workers {args.workers}")


if __name__ == "__main__":
    main()
