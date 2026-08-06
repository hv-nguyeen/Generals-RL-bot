"""Fit the win-probability model on the GPU, export for numpy inference.

Same trunk as the policy so `bot/policy/net.py` can run it on one CPU core; only
the head differs. The trunk builder is `learn.train`'s, so `--layers`,
`--channels` and `--residual` mean the same thing here, and ValueNet discovers
whichever was used straight out of the npz.

    python -m learn.valuetrain --data /local/data/vng205/val --out /local/data/vng205/value.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bot import features
from learn import train as bc


def init_params(key, arch: dict | None = None):
    """Same trunk builder as the policy, a scalar head instead of the move head."""
    import jax

    p = bc.init_params(key, arch)
    for k in ("head_w", "head_b", "pass_w", "pass_b"):
        del p[k]
    p["v_w"] = jax.random.normal(jax.random.fold_in(key, 7),
                                 (p["conv0_w"].shape[0],)) * 0.01
    p["v_b"] = jax.numpy.zeros(())
    return p


def forward(p, x):
    return bc.trunk(p, x).mean(axis=(2, 3)) @ p["v_w"] + p["v_b"]


def split(shards, frac: float, rng):
    """A validation sample drawn from every shard, not the last one.

    `valuedata` writes shards in source order, so one player's games land
    together and holding out `shards[-1]` holds out a *player*. The first run
    that way had the net at 0.753 against 0.786 for the scalar control while its
    training loss kept falling -- which is what a distribution shift looks like,
    not what an uninformative board looks like.

    Style transfer across players is a real question, but it is not the question
    this gate asks. Sample within each shard so train and val are the same
    distribution, and the gap measures only what the board adds.

    Returns (xv, yv, held) where `held` maps a shard path to the boolean mask of
    the rows that went to validation, so training can exclude exactly those.
    """
    xs, ys, held = [], [], {}
    for f in shards:
        z = np.load(f)
        n = len(z["y"])
        m = np.zeros(n, bool)
        m[rng.choice(n, max(1, int(n * frac)), replace=False)] = True
        held[f] = m
        xs.append(z["x"][m]); ys.append(z["y"][m])
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys), held)


def scalar_control(shards, held, xv, yv) -> float:
    """Held-out accuracy of least squares on the broadcast scalars alone.

    The scalars are the clock, the parity and the ten counting channels -- every
    number the encoder hands the net without it having to look at the board. A
    linear fit on those is the bar the trunk has to clear, because a critic that
    only ties win probability to "it is turn 300 and I have more land" has
    learned nothing a single matrix could not.

    This is the same control as `sc` in `learn.selfplay`, and it is here because
    the critic has lost to it before. Fitted on the training rows, scored on the
    identical held-out rows the net is scored on, so neither gets an advantage.
    """
    d = features.C - features.CLOCK + 1
    a = np.zeros((d, d), np.float64)
    b = np.zeros(d, np.float64)
    for f in shards:
        z = np.load(f)
        keep = ~held[f]
        s = np.c_[z["x"][keep][:, features.CLOCK:, 0, 0].astype(np.float64),
                  np.ones(int(keep.sum()))]
        a += s.T @ s
        b += s.T @ z["y"][keep].astype(np.float64)
    w = np.linalg.solve(a + 1e-6 * np.eye(d), b)
    sv = np.c_[xv[:, features.CLOCK:, 0, 0].astype(np.float64), np.ones(len(yv))]
    return float(np.mean((sv @ w > 0.5) == (yv > 0.5)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/local/data/vng205/val")
    ap.add_argument("--out", default="/local/data/vng205/value.npz")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--residual", action="store_true", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.02,
                    help="fraction of every shard held out, sampled within it")
    args = ap.parse_args()
    arch = bc.resolve_arch(None, args.layers, args.channels, args.residual)

    import jax
    import jax.numpy as jnp

    print("devices:", jax.devices())
    shards = sorted(Path(args.data).glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {args.data}")
    rng = np.random.default_rng(args.seed)
    train = shards
    xv, yv, held = split(shards, args.val_frac, rng)
    print(f"{len(shards)} shards, {len(yv)} validation rows sampled across all "
          f"of them ({args.val_frac:.0%})")

    control = scalar_control(shards, held, xv, yv)
    print(f"scalar control (least squares on {features.C - features.CLOCK} "
          f"broadcast channels): val acc {control:.3f}\n", flush=True)

    params = init_params(jax.random.PRNGKey(args.seed), arch)
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
            keep = ~held[train[si]]           # never train on a validation row
            xs, ys = z["x"][keep], z["y"][keep]
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
            bc.save(params, args.out)
        print(f"epoch {epoch}  loss {run / max(nb, 1):.4f}  val acc {a:.3f}  "
              f"{time.time() - started:.0f}s{mark}", flush=True)

    Path(args.out).with_suffix(".json").write_text(
        json.dumps({"val_acc": best, "control_acc": control, **arch}, indent=2))
    print(f"\nwrote {args.out} (best val accuracy {best:.3f})")
    print(f"scalar control {control:.3f}, net {best:.3f}, "
          f"gain {best - control:+.3f}")
    if best - control < 0.02:
        print("GATE FAILED: the board buys less than 2 points over the scalars.")
        print("Do not spend a training run on this critic.")
    else:
        print("Gate passed. Calibrate before trusting it:")
        print(f"  python -m tools.calibrate --model {args.out}")


if __name__ == "__main__":
    main()
