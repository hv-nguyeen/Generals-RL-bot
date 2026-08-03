"""Behaviour cloning on the GPU, exported for numpy inference.

Pure JAX plus a hand-rolled Adam — the cluster venv has jax with CUDA but no
torch, flax or optax, and adding dependencies to a machine that already works is
a good way to lose an afternoon.

The architecture is fixed by the inference budget, not by taste: `bot/policy/net.py`
runs the same 4x32 stack in numpy on one CPU core inside 150 ms, so the trainer
must not exceed it. Parameter names match what that loader expects.

    python -m learn.train --data /local/data/vng205/bc --out /local/data/vng205/clone.npz

The result is a sparring partner first and a submission second. Our own gauntlet
cannot punish the mistakes the field punishes, which is the single thing that has
been blocking progress; a policy cloned from strong players can.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bot import features
from bot.policy.net import CHANNELS, LAYERS


def init_params(key, cin: int = features.C):
    import jax
    import jax.numpy as jnp

    params = {}
    keys = jax.random.split(key, LAYERS + 2)
    prev = cin
    for i in range(LAYERS):
        # He init: fan_in is 3*3*prev
        scale = np.sqrt(2.0 / (9 * prev))
        params[f"conv{i}_w"] = jax.random.normal(keys[i], (CHANNELS, prev, 3, 3)) * scale
        params[f"conv{i}_b"] = jnp.zeros((CHANNELS,))
        prev = CHANNELS
    params["head_w"] = jax.random.normal(keys[LAYERS], (features.PER_CELL, prev, 3, 3)) * np.sqrt(2.0 / (9 * prev))
    params["head_b"] = jnp.zeros((features.PER_CELL,))
    params["pass_w"] = jax.random.normal(keys[LAYERS + 1], (prev,)) * 0.01
    params["pass_b"] = jnp.zeros(())
    return params


def forward(params, x):
    """x: (B, C, H, W) -> (B, N_ACTIONS). Mirrors bot/policy/net.py exactly."""
    import jax
    import jax.numpy as jnp

    def conv(inp, w, b):
        y = jax.lax.conv_general_dilated(
            inp, w, window_strides=(1, 1), padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"))
        return y + b[None, :, None, None]

    h = x
    for i in range(LAYERS):
        h = jax.nn.relu(conv(h, params[f"conv{i}_w"], params[f"conv{i}_b"]))
    move = conv(h, params["head_w"], params["head_b"])            # (B, 8, H, W)
    flat = jnp.transpose(move, (0, 2, 3, 1)).reshape(x.shape[0], -1)
    pass_logit = h.mean(axis=(2, 3)) @ params["pass_w"] + params["pass_b"]
    return jnp.concatenate([flat, pass_logit[:, None]], axis=1)


def shard_list(data: Path) -> list[Path]:
    shards = sorted(data.glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {data} — run `python -m learn.dataset` first")
    return shards


def load_shard(path: Path):
    """One shard, kept float16 in host memory. The full set does not fit:
    2.2M examples of 12x21x21 float32 is ~46 GB."""
    z = np.load(path)
    return z["x"], z["y"].astype(np.int32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/local/data/vng205/bc")
    ap.add_argument("--out", default="/local/data/vng205/clone.npz")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    print("devices:", jax.devices())
    shards = shard_list(Path(args.data))
    val_shard = shards[-1]
    train_shards = shards[:-1] or shards
    xv16, yv = load_shard(val_shard)
    xv = xv16[:8192].astype(np.float32)
    yv = yv[:8192]
    total = sum(1 for _ in train_shards)
    print(f"{len(shards)} shards ({total} for training, 1 held out), "
          f"{features.N_ACTIONS} classes")

    rng = np.random.default_rng(args.seed)

    key = jax.random.PRNGKey(args.seed)
    params = init_params(key)
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(p) for k, p in params.items()}

    def loss_fn(p, xb, yb):
        logits = forward(p, xb)
        ll = jax.nn.log_softmax(logits)
        return -jnp.mean(ll[jnp.arange(xb.shape[0]), yb])

    @jax.jit
    def step(p, m, v, t, xb, yb):
        loss, g = jax.value_and_grad(loss_fn)(p, xb, yb)
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = {k: b1 * m[k] + (1 - b1) * g[k] for k in p}
        v = {k: b2 * v[k] + (1 - b2) * g[k] ** 2 for k in p}
        mh = {k: m[k] / (1 - b1 ** t) for k in p}
        vh = {k: v[k] / (1 - b2 ** t) for k in p}
        p = {k: p[k] - args.lr * mh[k] / (jnp.sqrt(vh[k]) + eps) for k in p}
        return p, m, v, loss

    @jax.jit
    def accuracy(p, xb, yb):
        return jnp.mean(jnp.argmax(forward(p, xb), axis=1) == yb)

    t = 0
    for epoch in range(args.epochs):
        started, running, nb = time.time(), 0.0, 0
        for sp in rng.permutation(len(train_shards)):
            xs16, ys = load_shard(train_shards[sp])
            order = rng.permutation(len(ys))
            for i in range(0, len(order) - args.batch + 1, args.batch):
                idx = order[i:i + args.batch]
                t += 1
                params, m, v, loss = step(
                    params, m, v, t,
                    jnp.asarray(xs16[idx].astype(np.float32)), jnp.asarray(ys[idx]))
                running += float(loss)
                nb += 1
            del xs16, ys
        accs = [float(accuracy(params, jnp.asarray(xv[i:i + 1024]),
                               jnp.asarray(yv[i:i + 1024])))
                for i in range(0, len(yv), 1024)]
        print(f"epoch {epoch}  loss {running / max(nb, 1):.4f}  "
              f"val top-1 {np.mean(accs):.3f}  {time.time() - started:.0f}s", flush=True)

    out = Path(args.out)
    np.savez_compressed(out, **{k: np.asarray(v_) for k, v_ in params.items()})
    (out.with_suffix(".json")).write_text(json.dumps(
        {"shards": len(shards), "epochs": args.epochs,
         "channels": CHANNELS, "layers": LAYERS}, indent=2))
    print(f"\nwrote {out}")
    print("Use it as a sparring partner:")
    print(f"  python -m arena.runner --a ours --b clone:{out} --games 200 --workers 32")


if __name__ == "__main__":
    main()
