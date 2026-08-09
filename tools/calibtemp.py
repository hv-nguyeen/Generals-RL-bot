"""Fit the critic's temperature, because accuracy and evar punish different things.

`learn/valuetrain.py` selects on held-out explained variance and calibration.
`learn/selfplay.py` reads the critic through `tanh` and is also judged by
`evar`. A model
can win the first and lose the second by being overconfident: right 64% of the
time while saying 0.9, it scores 0.638 accuracy and NEGATIVE explained variance,
because each of the 36% it gets wrong costs (V - z)^2 close to 3.6.

That is what happened on 2026-08-08. `critic4.npz` passed valuetrain's gate at
0.638 against a 0.606 control, `tools/calibrate` read `mean P(win) 0.84`, and the
run it was loaded into opened at `evar -0.11` and fell to -0.38 -- worse than a
predictor that cannot see the board, from the very first iteration, before any
online training had touched it.

Schema-2 critics already use direct {-1,0,+1} targets and transfer without a
unit conversion. Legacy sigmoid critics are still supported and retain the
historical factor-of-two conversion.

So: sweep T on the selection games, then gate exactly once on sealed test games.
Scaling the linear branch and the residual head's final layer is exactly a
temperature -- no retraining, seconds to run.

    python -m tools.calibtemp --model runs/nn/critic4.npz \\
        --data /local/data/vng205/val4 --out runs/nn/critic4t.npz

Report `evar at T=1` against `best`. If T=1 is already best and still negative,
temperature is not the problem and the fit itself is too weak to use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path                              # noqa: E402
from bot import features                              # noqa: E402


def evar(z: np.ndarray, v: np.ndarray) -> float:
    """1 - Var(z - V)/Var(z): the SAME score `learn/selfplay.py` prints.

    Not accuracy, and not BCE. A critic is selected on this or it is selected on
    the wrong thing.
    """
    zv = float(z.var())
    return float(1.0 - ((z - v).var() / zv)) if zv > 1e-9 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="critic from learn.valuetrain")
    ap.add_argument("--data", action="append", required=True,
                    help="shard directory used for fitting; repeat in training order")
    ap.add_argument("--out", default="", help="write the rescaled critic here")
    ap.add_argument("--rows", type=int, default=40_000,
                    help="held-out rows sampled across ALL shards")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from learn import valuetrain as vt
    from bot.policy.net import arch_of, arch_record

    z0 = np.load(args.model)
    raw = {k: np.asarray(z0[k]) for k in z0.files}
    # `layers`/`channels`/`residual` are METADATA that `tools.grow` writes into
    # checkpoints. Left in, jit traces them and `arch_of` dies on int() of a
    # tracer. `load_critic` drops them the same way.
    p = {k: jnp.asarray(v) for k, v in raw.items() if k not in arch_record(raw)}
    print(f"critic {arch_of(raw)} from {args.model}")

    schema = int(raw.get("value_schema", 1))
    sidecar = Path(args.model).with_suffix(".json")
    model_meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    if schema >= 3:
        raise SystemExit(
            "schema-3 critics select temperature inside learn.valuetrain before "
            "the sealed test gate. Post-hoc calibration would reuse that test set; "
            "retrain instead of recalibrating this artifact.")
    select_keys = {(int(v["source"]), int(v["game"]))
                   for v in model_meta.get("selection_game_keys", [])}
    test_keys = {(int(v["source"]), int(v["game"]))
                 for v in model_meta.get("test_game_keys", [])}
    val_keys = {(int(v["source"]), int(v["game"]))
                for v in model_meta.get("validation_game_keys", [])}
    # Compatibility with schema-2 files produced before multi-source support.
    val_keys.update((0, int(v)) for v in model_meta.get("validation_game_ids", []))
    if schema >= 3 and (not select_keys or not test_keys):
        raise SystemExit("schema-3 calibration requires disjoint selection_game_keys "
                         "and test_game_keys in the model sidecar")
    if schema == 2 and not val_keys:
        raise SystemExit("schema-2 calibration requires validation_game_keys in the "
                         "model sidecar; retrain with the current valuetrain")
    if schema == 2:
        select_keys = test_keys = val_keys

    # Schema 2 uses the exact complete-game holdout that selected the model.
    # Legacy files predate game IDs and retain the old sampled fallback.
    rng = np.random.default_rng(args.seed)
    data_dirs = [Path(p) for p in args.data]
    expected_sources = model_meta.get("data_sources")
    actual_sources = [str(p.resolve()) for p in data_dirs]
    if schema >= 2 and expected_sources and expected_sources != actual_sources:
        raise SystemExit("--data sources or order differ from the critic sidecar: "
                         f"expected {expected_sources}, got {actual_sources}")
    shards = [(si, f) for si, d in enumerate(data_dirs)
              for f in sorted(d.glob("shard_*.npz"))]
    if not shards:
        raise SystemExit("no shard_*.npz in " + ", ".join(map(str, data_dirs)))
    per = max(args.rows // len(shards), 1)
    def collect(keys):
        xs, ys, gs = [], [], []
        for source, f in shards:
            d = np.load(f)
            if schema >= 2:
                wanted = [game for src, game in keys if src == source]
                candidates = np.flatnonzero(np.isin(d["game"], wanted))
            else:
                candidates = np.arange(len(d["y"]))
            if not len(candidates):
                continue
            m = rng.choice(candidates, size=min(per, len(candidates)), replace=False)
            xs.append(features.ensure_channels(d["x"][m])); ys.append(d["y"][m])
            if schema >= 2:
                gs.append(np.asarray([(source << 56) | int(g) for g in d["game"][m]],
                                     np.int64))
        if not xs:
            raise SystemExit("no rows matched the model split game keys")
        return np.concatenate(xs), np.concatenate(ys), (np.concatenate(gs) if gs else None)

    x, y, selected_games = collect(select_keys)
    print(f"{len(shards)} shards, {len(y)} selection rows, mean target {y.mean():+.3f}")

    logit = np.concatenate([
        np.asarray(jax.jit(vt.forward)(p, jnp.asarray(x[i:i + 4096], jnp.float32)))
        for i in range(0, len(x), 4096)]).astype(np.float64).ravel()
    zt = (y.astype(np.float64) if schema >= 2
          else 2.0 * y.astype(np.float64) - 1.0)

    print(f"\nlogit mean {logit.mean():+.3f} sd {logit.std():.3f}   "
          f"mean P(win) {float((1/(1+np.exp(-logit))).mean()):.3f}")
    print(f"\n{'T':>6} {'mean V':>8} {'sd V':>7} {'evar':>8}   "
          f"(raw direct-value temperature for schema 2)")
    best, best_t = -1e9, 1.0
    for t in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0):
        v = np.tanh(logit / t)
        e = evar(zt, v)
        mark = ""
        if e > best:
            best, best_t = e, t
            mark = "  <-"
        print(f"{t:>6.1f} {v.mean():>+8.3f} {v.std():>7.3f} {e:>+8.4f}{mark}")

    load_t = 1.0 if schema >= 2 else 2.0
    at_load = evar(zt, np.tanh(logit / load_t))
    print(f"\nbest T {best_t:.1f} -> evar {best:+.4f}   "
          f"(loader T={load_t:g}: {at_load:+.4f})")
    calibrated_metrics = None
    if schema >= 2:
        calibrated_metrics = vt.metrics(zt, np.tanh(logit / best_t), selected_games)
        print(f"calibrated game-balanced evar {calibrated_metrics['evar']:+.4f}, "
              f"ece {calibrated_metrics['ece']:.4f}")
    if best <= 0.0:
        print("\nNO TEMPERATURE HELPS. evar is <= 0 at every T, so the fit itself "
              "carries too little signal for this trainer -- more data or a "
              "different critic, not rescaling. Do not launch on it.")
        raise SystemExit(1)
    print("\nThis is an OFFLINE evar on the generator's own held-out rows. The "
          "run's `evar` is over a live buffer that includes draws and the last "
          "ticks of games, neither of which are here, so treat this as an upper "
          "bound and still read `diff` in the first iterations.")

    if args.out:
        # Scale every output branch. W1 stays fixed; the linear weights/bias and
        # residual W2/bias together form the complete scalar logit.
        scale = load_t / best_t
        out = vt.scale_value_output({k: np.asarray(z0[k]) for k in z0.files}, scale)
        np.savez(args.out, **out)
        if model_meta:
            model_meta["temperature"] = best_t
            model_meta["temperature_evar"] = best
            if calibrated_metrics is not None:
                model_meta["selection_metrics"] = calibrated_metrics
                gate_metrics = calibrated_metrics
                if schema >= 3:
                    test_x, test_y, test_games = collect(test_keys)
                    test_logit = np.concatenate([
                        np.asarray(jax.jit(vt.forward)(
                            p, jnp.asarray(test_x[i:i + 4096], jnp.float32)))
                        for i in range(0, len(test_x), 4096)]).astype(np.float64).ravel()
                    gate_metrics = vt.metrics(test_y.astype(np.float64),
                                              np.tanh(test_logit / best_t), test_games)
                model_meta["metrics"] = gate_metrics
                control = model_meta.get("control", {})
                model_meta["gate_passed"] = bool(
                    gate_metrics["target_var"] > 0.05
                    and gate_metrics["decided_frac"] > 0.1
                    and gate_metrics["evar"] - float(control.get("evar", 1e9)) >= 0.02
                    and gate_metrics["ece"] <= 0.15)
            Path(args.out).with_suffix(".json").write_text(
                json.dumps(model_meta, indent=2) + "\n")
        print(f"\nwrote {args.out} (head scaled by {scale:.4f}, T={best_t:.1f})")
    raise SystemExit(0)


def selfcheck() -> None:
    """The scaling identity, and that evar is the metric being maximised."""
    from learn import valuetrain as vt

    l = np.array([-3.0, -0.5, 0.0, 1.658, 3.0])
    assert np.allclose(2.0 / (1.0 + np.exp(-l)) - 1.0, np.tanh(l / 2.0))
    # a perfectly confident, perfectly correct critic scores 1; the mean scores 0
    z = np.array([1.0, -1.0, 1.0, -1.0])
    assert abs(evar(z, z) - 1.0) < 1e-9
    assert abs(evar(z, np.zeros_like(z))) < 1e-9
    # A CONSTANT offset does NOT hurt evar -- Var(z - c) == Var(z) -- so the
    # failure this tool addresses is not bias. Expanding,
    #     evar = (2*Cov(z, V) - Var(V)) / Var(z),
    # so evar goes negative exactly when Var(V) exceeds 2*Cov(z, V): a critic
    # that swings harder than its correlation earns. Shrinking V cuts Var(V)
    # quadratically and Cov only linearly, which is why an optimal T exists.
    for off in (0.7, -1.3):
        assert abs(evar(z, z + off) - evar(z, z)) < 1e-9, "offset must not move evar"
        assert abs(evar(z, np.zeros_like(z) + off) - evar(z, np.zeros_like(z))) < 1e-9
    conf = np.array([0.9, -0.9, -0.9, 0.9])          # confident, half of it wrong
    assert float(np.cov(z, conf, bias=True)[0, 1]) == 0.0
    assert evar(z, conf) < -0.5, evar(z, conf)
    # ...and shrinking that same prediction toward zero strictly helps
    assert evar(z, conf / 8.0) > evar(z, conf)
    # pre-multiplying by 2/T survives load_critic's halving to land on T
    T, w = 7.0, np.array([0.4, -0.2])
    assert np.allclose((w * (2.0 / T)) * 0.5, w / T)
    p = {"v_w": np.array([0.4, -0.2]), "v_b": np.array(0.3),
         "v_res1_w": np.eye(2), "v_res1_b": np.array([0.1, -0.2]),
         "v_res2_w": np.array([0.6, -0.4]), "v_res2_b": np.array(-0.1)}
    q = np.array([0.7, -0.5])
    raw = q @ p["v_w"] + p["v_b"] + np.maximum(
        q @ p["v_res1_w"] + p["v_res1_b"], 0) @ p["v_res2_w"] + p["v_res2_b"]
    scaled = vt.scale_value_output(p, 1.0 / T)
    got = q @ scaled["v_w"] + scaled["v_b"] + np.maximum(
        q @ scaled["v_res1_w"] + scaled["v_res1_b"], 0) @ scaled["v_res2_w"] \
        + scaled["v_res2_b"]
    assert np.allclose(got, raw / T), "residual-head temperature is not exact"
    print("calibtemp selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        main()
