"""Fit the win-probability model on the GPU, export for numpy inference.

Same trunk as the policy so `bot/policy/net.py` can run it on one CPU core; only
the head differs. Parameter names match what ValueNet loads.

    python -m learn.valuetrain --data /local/data/vng205/val --out /local/data/vng205/value.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bot import features
from bot.policy.net import CHANNELS, LAYERS


def init_params(key):
    import jax
    import jax.numpy as jnp
    p, keys, prev = {}, jax.random.split(key, LAYERS + 1), features.C
    for i in range(LAYERS):
        p[f"conv{i}_w"] = jax.random.normal(
            keys[i], (CHANNELS, prev, 3, 3)) * np.sqrt(2.0 / (9 * prev))
        p[f"conv{i}_b"] = jnp.zeros((CHANNELS,))
        prev = CHANNELS
    p["v_w"] = jax.random.normal(keys[LAYERS], (CHANNELS,)) * 0.01
    p["v_b"] = jnp.zeros(())
    return p


def forward(p, x):
    import jax
    import jax.numpy as jnp

    def conv(inp, w, b):
        return jax.lax.conv_general_dilated(
            inp, w, (1, 1), "SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW")) + b[None, :, None, None]

    h = x
    for i in range(LAYERS):
        h = jax.nn.relu(conv(h, p[f"conv{i}_w"], p[f"conv{i}_b"]))
    return h.mean(axis=(2, 3)) @ p["v_w"] + p["v_b"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/local/data/vng205/val")
    ap.add_argument("--out", default="/local/data/vng205/value.npz")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    print("devices:", jax.devices())
    shards = sorted(Path(args.data).glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {args.data}")
    val = np.load(shards[-1])
    xv, yv = val["x"][:8192].astype(np.float32), val["y"][:8192]
    train = shards[:-1] or shards
    print(f"{len(shards)} shards, {len(train)} for training")

    rng = np.random.default_rng(args.seed)
    params = init_params(jax.random.PRNGKey(args.seed))
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(x) for k, x in params.items()}

    def loss_fn(p, xb, yb):
        logit = forward(p, xb)
        return jnp.mean(jnp.logaddexp(0.0, logit) - yb * logit)   # BCE

    @jax.jit
    def step(p, m, v, t, xb, yb):
        loss, g = jax.value_and_grad(loss_fn)(p, xb, yb)
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = {k: b1 * m[k] + (1 - b1) * g[k] for k in p}
        v = {k: b2 * v[k] + (1 - b2) * g[k] ** 2 for k in p}
        p = {k: p[k] - args.lr * (m[k] / (1 - b1 ** t))
             / (jnp.sqrt(v[k] / (1 - b2 ** t)) + eps) for k in p}
        return p, m, v, loss

    @jax.jit
    def acc(p, xb, yb):
        return jnp.mean((forward(p, xb) > 0) == (yb > 0.5))

    best, t = -1.0, 0
    for epoch in range(args.epochs):
        started, run, nb = time.time(), 0.0, 0
        for si in rng.permutation(len(train)):
            z = np.load(train[si])
            xs, ys = z["x"], z["y"]
            order = rng.permutation(len(ys))
            for i in range(0, len(order) - args.batch + 1, args.batch):
                sel = order[i:i + args.batch]
                t += 1
                params, m, v, loss = step(params, m, v, t,
                                          jnp.asarray(xs[sel].astype(np.float32)),
                                          jnp.asarray(ys[sel]))
                run += float(loss); nb += 1
            del xs, ys
        a = float(np.mean([float(acc(params, jnp.asarray(xv[i:i + 1024]),
                                     jnp.asarray(yv[i:i + 1024])))
                           for i in range(0, len(yv), 1024)]))
        mark = ""
        if a > best:
            best, mark = a, "  <- kept"
            np.savez_compressed(args.out, **{k: np.asarray(x) for k, x in params.items()})
        print(f"epoch {epoch}  loss {run / max(nb, 1):.4f}  val acc {a:.3f}  "
              f"{time.time() - started:.0f}s{mark}", flush=True)

    Path(args.out).with_suffix(".json").write_text(json.dumps({"val_acc": best}, indent=2))
    print(f"\nwrote {args.out} (best val accuracy {best:.3f})")
    print("Calibrate before trusting it:")
    print(f"  python -m tools.calibrate --model {args.out}")


if __name__ == "__main__":
    main()
