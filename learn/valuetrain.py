"""Fit a calibrated game-value model on complete-game train/validation splits.

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


def init_params(key, arch: dict | None = None):
    """Same trunk builder as the policy, a scalar head instead of the move head."""
    import jax

    p = bc.init_params(key, arch)
    for k in ("head_w", "head_b", "pass_w", "pass_b"):
        del p[k]
    p["v_w"] = jax.random.normal(jax.random.fold_in(key, 7),
                                 (p["conv0_w"].shape[0] * VALUE_POOL,)) * 0.01
    p["v_b"] = jax.numpy.zeros(())
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
    return pooled(h, x[:, features.VALID]) @ p["v_w"] + p["v_b"]


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


def split(shards, frac: float, rng, game_ids=None):
    """Hold out whole games. Positions from one game never cross the boundary."""
    if game_ids is None:
        game_ids, _, _ = canonical_game_ids(shards)
    games = np.unique(np.concatenate([game_ids[f] for f in shards]))
    if len(games) < 2:
        raise ValueError("value data needs at least two distinct game ids")
    nval = min(len(games) - 1, max(1, int(round(len(games) * frac))))
    val_games = set(int(x) for x in rng.choice(games, nval, replace=False))
    xs, ys, gs, held = [], [], [], {}
    for f in shards:
        z = np.load(f)
        m = np.isin(game_ids[f], list(val_games))
        held[f] = m
        if m.any():
            xs.append(z["x"][m]); ys.append(z["y"][m])
            gs.append(game_ids[f][m])
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys),
            np.concatenate(gs), held, val_games)


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


def scalar_control(shards, held, xv, yv, gv) -> tuple[np.ndarray, dict]:
    """Least-squares baseline on broadcast scalars, scored on unseen games.

    The scalars are the clock, the parity and the ten counting channels -- every
    number the encoder hands the net without it having to look at the board. A
    linear fit on those is the bar the trunk has to clear, because a critic that
    only ties win probability to "it is turn 300 and I have more land" has
    learned nothing a single matrix could not.

    This is the same control as `sc` in `learn.selfplay`, and it is here because
    the critic has lost to it before. Fitted on the training rows, scored on the
    identical held-out rows the net is scored on, so neither gets an advantage.
    """
    d = (features.BASE_C - features.CLOCK) + 3 + 1
    a = np.zeros((d, d), np.float64)
    b = np.zeros(d, np.float64)
    for f in shards:
        z = np.load(f)
        keep = ~held[f]
        s = np.c_[_scalar_columns(z["x"][keep]), np.ones(int(keep.sum()))]
        a += s.T @ s
        b += s.T @ z["y"][keep].astype(np.float64)
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.02,
                    help="fraction of complete games held out")
    args = ap.parse_args()
    arch = bc.resolve_arch(None, args.layers, args.channels, args.residual, args.context)

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
    xv, yv, gv, held, val_games = split(shards, args.val_frac, rng, game_ids)
    print(f"{len(shards)} shards from {len(source_dirs)} source(s), "
          f"{len(val_games)} held-out complete games, "
          f"{len(yv)} validation rows ({args.val_frac:.0%})")

    _, control = scalar_control(shards, held, xv, yv, gv)
    print(f"scalar control: evar {control['evar']:.3f}  mae {control['mae']:.3f}  "
          f"ece {control['ece']:.3f}\n", flush=True)

    params = init_params(jax.random.PRNGKey(args.seed), arch)
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

    best, best_metrics, t = -np.inf, None, 0
    for epoch in range(args.epochs):
        started, run, nb = time.time(), 0.0, 0
        for si in rng.permutation(len(train)):
            z = np.load(train[si])
            keep = ~held[train[si]]           # never train on a validation row
            xs, ys, gs = z["x"][keep], z["y"][keep], game_ids[train[si]][keep]
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
        pred = np.concatenate([
            np.asarray(jnp.tanh(forward(params, jnp.asarray(xv[i:i + 1024]))))
            for i in range(0, len(yv), 1024)])
        met = metrics(yv, pred, gv)
        score = met["evar"] - 0.1 * met["ece"]
        mark = ""
        if score > best:
            best, best_metrics, mark = score, met, "  <- kept"
            arrays = {k: np.asarray(v) for k, v in params.items()}
            np.savez_compressed(args.out, **arrays, **bc.arch_record(arrays),
                                value_schema=np.int16(2),
                                value_pool=np.int16(VALUE_POOL))
        print(f"epoch {epoch}  loss {run / max(nb, 1):.4f}  "
              f"evar {met['evar']:.3f}  mae {met['mae']:.3f}  "
              f"ece {met['ece']:.3f}  sign {met['sign_acc']:.3f}  "
              f"{time.time() - started:.0f}s{mark}", flush=True)

    gate_passed = bool(best_metrics["target_var"] > 0.05
                       and best_metrics["decided_frac"] > 0.1
                       and best_metrics["evar"] - control["evar"] >= 0.02
                       and best_metrics["ece"] <= 0.15)
    validation_keys = [{"source": game_refs[g][0], "game": game_refs[g][1]}
                       for g in sorted(val_games)]
    Path(args.out).with_suffix(".json").write_text(
        json.dumps({"schema_version": 2, "validation_games": len(val_games),
                    "validation_game_keys": validation_keys,
                    "data_sources": [str(p) for p in source_dirs],
                    "metrics": best_metrics, "control": control,
                    "gate_passed": gate_passed, **arch}, indent=2))
    print(f"\nwrote {args.out} (best evar {best_metrics['evar']:.3f})")
    print(f"scalar evar {control['evar']:.3f}, net {best_metrics['evar']:.3f}, "
          f"gain {best_metrics['evar'] - control['evar']:+.3f}")
    from tools import manifest
    artifacts = [args.out]
    artifacts.extend(d / "meta.json" for d in data_dirs if (d / "meta.json").is_file())
    manifest.write(Path(args.out).with_suffix(".manifest.json"), command=sys.argv,
                   artifacts=artifacts,
                   extra={"kind": "value-training", "args": vars(args),
                          "gate_passed": gate_passed,
                          "validation_games": validation_keys,
                          "data_sources": [str(p) for p in source_dirs]})
    if not gate_passed:
        print("GATE FAILED: insufficient board gain or poor calibration.")
        print("Do not spend a training run on this critic.")
        raise SystemExit(1)
    else:
        print("Gate passed on complete held-out games.")


if __name__ == "__main__":
    main()
