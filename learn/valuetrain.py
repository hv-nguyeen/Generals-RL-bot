"""Fit a calibrated game-value model on complete-game train/select/test splits.

Same trunk as the policy so `bot/policy/net.py` can run it on one CPU core; only
the head differs. The trunk builder is `learn.train`'s, so `--layers`,
`--channels` and `--residual` mean the same thing here, and ValueNet discovers
whichever was used straight out of the npz.

    python -m learn.valuetrain --data /local/data/vng205/val --out /local/data/vng205/value.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from bot import features
from learn import train as bc


VALUE_REGIONS = 3
VALUE_POOL = 2 + VALUE_REGIONS * VALUE_REGIONS   # mean, max, 3x3 regional means
TEMPERATURES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)


def scale_value_output(params: dict, scale: float) -> dict:
    """Scale the complete scalar logit, including the residual output layer."""
    out = {k: np.asarray(v) for k, v in params.items()}
    for key in ("v_w", "v_b", "v_res2_w", "v_res2_b"):
        if key in out:
            out[key] = np.asarray(out[key], np.float32) * np.float32(scale)
    return out


def init_params(key, arch: dict | None = None, value_hidden: int = 64):
    """Same trunk as policy, with a linear value plus a zero-output residual MLP.

    The linear branch is retained so historical critics migrate exactly.  The
    residual branch starts with W2=0, hence adding it changes no prediction at
    initialisation while still giving W2 a gradient on the first update.
    """
    import jax
    import jax.numpy as jnp

    p = bc.init_params(key, arch)
    for k in ("head_w", "head_b", "pass_w", "pass_b"):
        del p[k]
    p["v_w"] = jax.random.normal(jax.random.fold_in(key, 7),
                                 (p["conv0_w"].shape[0] * VALUE_POOL,)) * 0.01
    p["v_b"] = jnp.zeros(())
    if value_hidden:
        if value_hidden < 1:
            raise ValueError("value_hidden must be zero or positive")
        width = int(p["v_w"].shape[0])
        p["v_res1_w"] = (jax.random.normal(jax.random.fold_in(key, 8),
                                            (width, value_hidden))
                            * np.sqrt(2.0 / width))
        p["v_res1_b"] = jnp.zeros((value_hidden,))
        p["v_res2_w"] = jnp.zeros((value_hidden,))
        p["v_res2_b"] = jnp.zeros(())
    return p


def pooled(h, valid):
    """Global and coarse spatial features without discarding board topology."""
    import jax.numpy as jnp

    valid = valid.astype(h.dtype)
    count = jnp.maximum(valid.sum(axis=(1, 2), keepdims=True), 1.0)
    mean = (h * valid[:, None]).sum(axis=(2, 3)) / count[:, 0]
    masked = jnp.where(valid[:, None] > 0, h, -1e9)
    maximum = masked.max(axis=(2, 3))
    parts = [mean, maximum]
    edges = (0, 7, 14, features.PAD)
    for ri in range(VALUE_REGIONS):
        for ci in range(VALUE_REGIONS):
            v = valid[:, edges[ri]:edges[ri + 1], edges[ci]:edges[ci + 1]]
            q = h[:, :, edges[ri]:edges[ri + 1], edges[ci]:edges[ci + 1]]
            den = jnp.maximum(v.sum(axis=(1, 2), keepdims=True), 1.0)
            parts.append((q * v[:, None]).sum(axis=(2, 3)) / den[:, 0])
    return jnp.concatenate(parts, axis=1)


def forward(p, x):
    h = bc.trunk(p, x)
    q = pooled(h, x[:, features.VALID])
    value = q @ p["v_w"] + p["v_b"]
    if "v_res1_w" in p:
        import jax
        value = (value
                 + jax.nn.relu(q @ p["v_res1_w"] + p["v_res1_b"])
                 @ p["v_res2_w"] + p["v_res2_b"])
    return value


def canonical_game_ids(shards):
    """Namespace numeric game ids by source directory.

    Official replay ids and deterministic self-play seeds are independently
    assigned namespaces.  Treating the raw integers as globally unique can
    silently merge unrelated games when data sources are combined.
    """
    source_dirs = []
    for f in shards:
        parent = Path(f).resolve().parent
        if parent not in source_dirs:
            source_dirs.append(parent)
    source_index = {p: i for i, p in enumerate(source_dirs)}
    raw = {f: np.asarray(np.load(f)["game"], np.int64) for f in shards}
    pairs = sorted({(source_index[Path(f).resolve().parent], int(g))
                    for f in shards for g in np.unique(raw[f])})
    encode = {pair: i for i, pair in enumerate(pairs)}
    ids = {f: np.asarray([encode[(source_index[Path(f).resolve().parent], int(g))]
                          for g in raw[f]], np.int64) for f in shards}
    reverse = {v: k for k, v in encode.items()}
    return ids, reverse, source_dirs


def _materialize(shards, game_ids, chosen):
    xs, ys, gs = [], [], []
    for f in shards:
        z = np.load(f)
        mask = np.isin(game_ids[f], list(chosen))
        if mask.any():
            xs.append(features.ensure_channels(z["x"][mask]))
            ys.append(z["y"][mask]); gs.append(game_ids[f][mask])
    if not xs:
        raise ValueError("value split selected no rows")
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys),
            np.concatenate(gs))


def split_three_way(shards, select_frac: float, test_frac: float, rng,
                    game_ids=None, game_refs=None):
    """Source-stratified complete-game train/select/test partition.

    Selection games choose the epoch (and later the temperature). Test games
    are untouched until the chosen checkpoint is gated. Each source contributes
    at least one game to every split, preventing a large generator from hiding
    failure on a smaller but strategically important source.
    """
    if not (0.0 < select_frac < 1.0 and 0.0 < test_frac < 1.0):
        raise ValueError("select/test fractions must be in (0, 1)")
    if game_ids is None:
        game_ids, game_refs, _ = canonical_game_ids(shards)
    if game_refs is None:
        _, game_refs, _ = canonical_game_ids(shards)
    by_source: dict[int, list[int]] = {}
    for gid, (source, _) in game_refs.items():
        by_source.setdefault(int(source), []).append(int(gid))
    select_games, test_games = set(), set()
    for source, source_games in sorted(by_source.items()):
        games = np.asarray(sorted(source_games), np.int64)
        if len(games) < 3:
            raise ValueError(f"value source {source} needs at least three complete "
                             f"games for train/select/test, found {len(games)}")
        order = rng.permutation(games)
        ntest = min(len(games) - 2, max(1, int(round(len(games) * test_frac))))
        nselect = min(len(games) - ntest - 1,
                      max(1, int(round(len(games) * select_frac))))
        select_games.update(int(g) for g in order[:nselect])
        test_games.update(int(g) for g in order[nselect:nselect + ntest])
    if select_games & test_games:
        raise AssertionError("selection/test game overlap")
    held = {f: np.isin(game_ids[f], list(select_games | test_games)) for f in shards}
    # Deliberately do not materialize test rows here. The caller receives only
    # their game IDs until model and temperature selection are finished.
    return (_materialize(shards, game_ids, select_games), held,
            select_games, test_games)


def _scalar_columns(x):
    idx = list(range(features.CLOCK, features.BASE_C)) + [
        features.DELTA_MY_ARMY, features.DELTA_OPP_ARMY,
        features.DELTA_LAND_ADV]
    return x[:, idx, 0, 0].astype(np.float64)


def _weights_by_game(game):
    ids, counts = np.unique(game, return_counts=True)
    inv = {int(g): 1.0 / int(n) for g, n in zip(ids, counts)}
    w = np.asarray([inv[int(g)] for g in game], np.float64)
    return w / w.sum()


def metrics(y, pred, game) -> dict:
    """Game-balanced value metrics; long games do not dominate the gate."""
    y, pred = np.asarray(y), np.asarray(pred)
    w = _weights_by_game(game)
    err = pred - y
    mse = float(np.sum(w * err ** 2))
    var = float(np.sum(w * (y - np.sum(w * y)) ** 2))
    decided = y != 0
    sign = (float(np.sum((w[decided] / w[decided].sum())
                         * (np.sign(pred[decided]) == y[decided])))
            if decided.any() else 0.0)
    # Expected calibration error on the direct value scale.
    ece = 0.0
    for bi, lo in enumerate(np.linspace(-1, 0.8, 10)):
        m = (pred >= lo) & ((pred <= lo + 0.2) if bi == 9 else (pred < lo + 0.2))
        if m.any():
            wm = w[m] / w[m].sum()
            ece += float(w[m].sum() * abs(np.sum(wm * pred[m]) - np.sum(wm * y[m])))
    # An all-draw/all-one-outcome holdout has no value question and cannot pass
    # by subtracting two enormous negative evars from an epsilon denominator.
    evar = 1.0 - mse / var if var > 1e-6 else -1.0
    return {"mae": float(np.sum(w * np.abs(err))), "mse": mse,
            "evar": evar, "target_var": var,
            "decided_frac": float(np.sum(w[decided])), "brier": mse / 4.0,
            "sign_acc": sign, "ece": ece}


def scalar_control(shards, held, xv, yv, gv, game_ids) -> tuple[np.ndarray, dict]:
    """Game-balanced scalar baseline fitted on train and scored on one split.

    The scalar-only control is deliberately denied both held-out partitions.
    Weighting each complete training game equally also prevents long games from
    dominating its fit while the critic is judged by game-balanced metrics.
    """
    d = (features.BASE_C - features.CLOCK) + 3 + 1
    a = np.zeros((d, d), np.float64)
    b = np.zeros(d, np.float64)
    counts = {}
    for f in shards:
        for g in game_ids[f][~held[f]]:
            counts[int(g)] = counts.get(int(g), 0) + 1
    for f in shards:
        z = np.load(f)
        keep = ~held[f]
        s = np.c_[_scalar_columns(features.ensure_channels(z["x"][keep])),
                  np.ones(int(keep.sum()))]
        weights = np.asarray([1.0 / counts[int(g)] for g in game_ids[f][keep]])
        a += s.T @ (weights[:, None] * s)
        b += s.T @ (weights * z["y"][keep].astype(np.float64))
    w = np.linalg.solve(a + 1e-6 * np.eye(d), b)
    sv = np.c_[_scalar_columns(xv), np.ones(len(yv))]
    pred = np.clip(sv @ w, -1.0, 1.0)
    return pred, metrics(yv, pred, gv)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", default=None,
                    help="value shard directory; repeat to mix sources")
    ap.add_argument("--out", default="/local/data/vng205/value.npz")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--residual", action="store_true", default=None)
    ap.add_argument("--context", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dense-context", action="store_true",
                    help="use CxC global/regional context mixers")
    ap.add_argument("--strategy-hidden", type=int, default=0,
                    help="global 3x3 spatial-strategy MLP width; 0 disables it")
    ap.add_argument("--value-hidden", type=int, default=64,
                    help="residual value-head width; 0 reproduces the linear head")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select-frac", "--val-frac", dest="select_frac", type=float,
                    default=0.02,
                    help="complete games used only for epoch/temperature selection")
    ap.add_argument("--test-frac", type=float, default=0.02,
                    help="complete games touched only once by the final gate")
    args = ap.parse_args()
    arch = bc.resolve_arch(None, args.layers, args.channels, args.residual,
                           args.context, args.dense_context, args.strategy_hidden)

    import jax
    import jax.numpy as jnp

    print("devices:", jax.devices())
    data_dirs = [Path(p) for p in (args.data or ["/local/data/vng205/val"])]
    shards = [f for d in data_dirs for f in sorted(d.glob("shard_*.npz"))]
    if not shards:
        raise SystemExit("no shards in " + ", ".join(map(str, data_dirs)))
    rng = np.random.default_rng(args.seed)
    train = shards
    for f in shards:
        z = np.load(f)
        missing = {"game", "seat", "t"} - set(z.files)
        if missing:
            raise SystemExit(f"{f} is legacy value data, missing {sorted(missing)}; "
                             "rebuild it with learn.valuedata/tools.valueselfplay")
        if not set(np.unique(z["y"])) <= {-1.0, 0.0, 1.0}:
            raise SystemExit(f"{f}: targets must be direct values -1/0/+1")
    game_ids, game_refs, source_dirs = canonical_game_ids(shards)
    selection, held, select_games, test_games = split_three_way(
        shards, args.select_frac, args.test_frac, rng, game_ids, game_refs)
    xselect, yselect, gselect = selection
    print(f"{len(shards)} shards from {len(source_dirs)} source(s), "
          f"{len(select_games)} selection games/{len(yselect)} rows, "
          f"{len(test_games)} sealed test games")

    _, select_control = scalar_control(
        shards, held, xselect, yselect, gselect, game_ids)
    print(f"selection scalar control: evar {select_control['evar']:.3f}  "
          f"mae {select_control['mae']:.3f}  ece {select_control['ece']:.3f}\n",
          flush=True)

    params = init_params(jax.random.PRNGKey(args.seed), arch, args.value_hidden)
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(x) for k, x in params.items()}

    def loss_fn(p, xb, yb, wb):
        pred = jnp.tanh(forward(p, xb))
        e = jnp.abs(pred - yb)
        huber = jnp.where(e < 0.5, 0.5 * e ** 2, 0.5 * (e - 0.25))
        return jnp.sum(wb * huber) / jnp.maximum(jnp.sum(wb), 1e-8)

    @jax.jit
    def step(p, m, v, t, xb, yb, wb):
        loss, g = jax.value_and_grad(loss_fn)(p, xb, yb, wb)
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = {k: b1 * m[k] + (1 - b1) * g[k] for k in p}
        v = {k: b2 * v[k] + (1 - b2) * g[k] ** 2 for k in p}
        p = {k: p[k] - args.lr * (m[k] / (1 - b1 ** t))
             / (jnp.sqrt(v[k] / (1 - b2 ** t)) + eps) for k in p}
        return p, m, v, loss

    game_counts = {}
    for f in train:
        z = np.load(f)
        for g in game_ids[f][~held[f]]:
            game_counts[int(g)] = game_counts.get(int(g), 0) + 1

    best, best_metrics, best_arrays, best_temperature, t = (
        -np.inf, None, None, None, 0)
    for epoch in range(args.epochs):
        started, run, nb = time.time(), 0.0, 0
        for si in rng.permutation(len(train)):
            z = np.load(train[si])
            keep = ~held[train[si]]           # never train on selection/test rows
            xs = features.ensure_channels(z["x"][keep])
            ys, gs = z["y"][keep], game_ids[train[si]][keep]
            order = rng.permutation(len(ys))
            for i in range(0, len(order) - args.batch + 1, args.batch):
                sel = order[i:i + args.batch]
                wb = np.asarray([1.0 / game_counts[int(g)] for g in gs[sel]],
                                np.float32)
                t += 1
                params, m, v, loss = step(params, m, v, t,
                                          jnp.asarray(xs[sel].astype(np.float32)),
                                          jnp.asarray(ys[sel]), jnp.asarray(wb))
                run += float(loss); nb += 1
            del xs, ys, gs
        logit = np.concatenate([
            np.asarray(forward(params, jnp.asarray(xselect[i:i + 1024])))
            for i in range(0, len(yselect), 1024)])
        choices = [(metrics(yselect, np.tanh(logit / temp), gselect), temp)
                   for temp in TEMPERATURES]
        met, temperature = max(
            choices, key=lambda q: q[0]["evar"] - 0.1 * q[0]["ece"])
        score = met["evar"] - 0.1 * met["ece"]
        mark = ""
        if score > best:
            best, best_metrics, best_temperature, mark = (
                score, met, temperature, "  <- kept")
            best_arrays = {k: np.asarray(v) for k, v in params.items()}
        print(f"epoch {epoch}  loss {run / max(nb, 1):.4f}  "
              f"evar {met['evar']:.3f}  mae {met['mae']:.3f}  "
              f"ece {met['ece']:.3f}  sign {met['sign_acc']:.3f}  "
              f"T {temperature:g}  "
              f"{time.time() - started:.0f}s{mark}", flush=True)

    if best_arrays is None:
        raise SystemExit("no complete training batch; lower --batch or add data")
    # First and only use of the sealed test partition: it cannot select an
    # epoch, temperature, architecture, or any other hyperparameter.
    best_arrays = scale_value_output(best_arrays, 1.0 / best_temperature)
    chosen = {k: jnp.asarray(v) for k, v in best_arrays.items()}
    xtest, ytest, gtest = _materialize(shards, game_ids, test_games)
    test_pred = np.concatenate([
        np.asarray(jnp.tanh(forward(chosen, jnp.asarray(xtest[i:i + 1024]))))
        for i in range(0, len(ytest), 1024)])
    test_metrics = metrics(ytest, test_pred, gtest)
    _, test_control = scalar_control(shards, held, xtest, ytest, gtest, game_ids)
    gate_passed = bool(test_metrics["target_var"] > 0.05
                       and test_metrics["decided_frac"] > 0.1
                       and test_metrics["evar"] - test_control["evar"] >= 0.02
                       and test_metrics["ece"] <= 0.15)
    np.savez_compressed(args.out, **best_arrays, **bc.arch_record(best_arrays),
                        value_schema=np.int16(3), value_pool=np.int16(VALUE_POOL),
                        value_hidden=np.int16(args.value_hidden))
    game_keys = lambda games: [                                           # noqa: E731
        {"source": game_refs[g][0], "game": game_refs[g][1]}
        for g in sorted(games)]
    selection_keys, test_keys = game_keys(select_games), game_keys(test_games)
    Path(args.out).with_suffix(".json").write_text(
        json.dumps({"schema_version": 3,
                    "selection_games": len(select_games),
                    "selection_game_keys": selection_keys,
                    "test_games": len(test_games), "test_game_keys": test_keys,
                    "data_sources": [str(p) for p in source_dirs],
                    "selection_metrics": best_metrics,
                    "selection_control": select_control,
                    "temperature": best_temperature,
                    "metrics": test_metrics, "control": test_control,
                    "gate_passed": gate_passed, **arch}, indent=2))
    print(f"\nwrote {args.out} (selection evar {best_metrics['evar']:.3f}, "
          f"sealed-test evar {test_metrics['evar']:.3f})")
    print(f"test scalar evar {test_control['evar']:.3f}, net "
          f"{test_metrics['evar']:.3f}, "
          f"gain {test_metrics['evar'] - test_control['evar']:+.3f}")
    from tools import manifest
    artifacts = [args.out]
    artifacts.extend(d / "meta.json" for d in data_dirs if (d / "meta.json").is_file())
    manifest.write(Path(args.out).with_suffix(".manifest.json"), command=sys.argv,
                   artifacts=artifacts,
                   extra={"kind": "value-training", "args": vars(args),
                          "gate_passed": gate_passed,
                          "selection_games": selection_keys,
                          "test_games": test_keys,
                          "data_sources": [str(p) for p in source_dirs]})
    if not gate_passed:
        print("GATE FAILED: insufficient board gain or poor calibration.")
        print("Do not spend a training run on this critic.")
        raise SystemExit(1)
    else:
        print("Gate passed once on sealed complete test games.")


if __name__ == "__main__":
    main()
