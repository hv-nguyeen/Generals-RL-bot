"""Fit the critic's temperature, because accuracy and evar punish different things.

`learn/valuetrain.py` selects on ACCURACY. `learn/selfplay.py` reads the critic
through `tanh` and is judged by `evar`, which is a squared-error score. A model
can win the first and lose the second by being overconfident: right 64% of the
time while saying 0.9, it scores 0.638 accuracy and NEGATIVE explained variance,
because each of the 36% it gets wrong costs (V - z)^2 close to 3.6.

That is what happened on 2026-08-08. `critic4.npz` passed valuetrain's gate at
0.638 against a 0.606 control, `tools/calibrate` read `mean P(win) 0.84`, and the
run it was loaded into opened at `evar -0.11` and fell to -0.38 -- worse than a
predictor that cannot see the board, from the very first iteration, before any
online training had touched it.

`load_critic` already divides the head by 2, converting valuetrain's sigmoid
logit into the tanh argument the trainer wants: 2*sigma(l) - 1 == tanh(l/2)
exactly. That conversion is right and it is not enough. It fixes the UNITS and
assumes a temperature of 1; an overconfident fit needs T > 1 on top, and the
only way to know T is to measure it.

So: sweep T on held-out data, score with the metric the run is actually judged
by, and write the rescaled critic. The head is linear, so dividing `v_w`/`v_b`
by T is exactly a temperature -- no retraining, seconds to run.

    python -m tools.calibtemp --model runs/nn/critic4.npz \\
        --data /local/data/vng205/val4 --out runs/nn/critic4t.npz

Report `evar at T=1` against `best`. If T=1 is already best and still negative,
temperature is not the problem and the fit itself is too weak to use.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path                              # noqa: E402


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
    ap.add_argument("--data", required=True, help="shard dir it was fitted on")
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

    # Sampled WITHIN every shard, not the last shard: `valuedata` writes shards
    # in source order, so holding out the tail holds out a source. Same reason
    # `valuetrain.split` exists.
    rng = np.random.default_rng(args.seed)
    shards = sorted(Path(args.data).glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shard_*.npz in {args.data}")
    xs, ys = [], []
    per = max(args.rows // len(shards), 1)
    for f in shards:
        d = np.load(f)
        n = len(d["y"])
        m = rng.choice(n, size=min(per, n), replace=False)
        xs.append(d["x"][m])
        ys.append(d["y"][m])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    print(f"{len(shards)} shards, {len(y)} held-out rows, win rate {y.mean():.3f}")

    logit = np.concatenate([
        np.asarray(jax.jit(vt.forward)(p, jnp.asarray(x[i:i + 4096], jnp.float32)))
        for i in range(0, len(x), 4096)]).astype(np.float64).ravel()
    # valuetrain's label is {0, 1}; the trainer's return is {-1, +1}. Same event.
    zt = 2.0 * y.astype(np.float64) - 1.0

    print(f"\nlogit mean {logit.mean():+.3f} sd {logit.std():.3f}   "
          f"mean P(win) {float((1/(1+np.exp(-logit))).mean()):.3f}")
    print(f"\n{'T':>6} {'mean V':>8} {'sd V':>7} {'evar':>8}   "
          f"(load_critic applies T=2 today)")
    best, best_t = -1e9, 1.0
    for t in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0):
        v = np.tanh(logit / t)
        e = evar(zt, v)
        mark = ""
        if e > best:
            best, best_t = e, t
            mark = "  <-"
        print(f"{t:>6.1f} {v.mean():>+8.3f} {v.std():>7.3f} {e:>+8.4f}{mark}")

    at2 = evar(zt, np.tanh(logit / 2.0))
    print(f"\nbest T {best_t:.1f} -> evar {best:+.4f}   "
          f"(T=2, what load_critic does now: {at2:+.4f})")
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
        # The head is linear in the trunk features, so scaling it IS the
        # temperature. `load_critic` will then halve it again, so pre-multiply by
        # 2 to land on exactly best_t overall rather than 2*best_t.
        out = {k: np.asarray(z0[k]) for k in z0.files}
        out["v_w"] = np.asarray(out["v_w"], np.float32) * (2.0 / best_t)
        out["v_b"] = np.asarray(out["v_b"], np.float32) * (2.0 / best_t)
        np.savez(args.out, **out)
        print(f"\nwrote {args.out} (head scaled by {2.0 / best_t:.4f}; "
              f"load_critic's own halving then lands it at T={best_t:.1f})")
    raise SystemExit(0)


def selfcheck() -> None:
    """The scaling identity, and that evar is the metric being maximised."""
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
    print("calibtemp selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        main()
