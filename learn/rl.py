"""PPO self-play, warm-started from the behaviour clone.

From scratch this is a weeks-long compute project. From a cloned initialisation
it is tractable: the policy already knows how to expand and roughly where to
attack, and RL only has to fix what imitation cannot — that the field's average
move is not the winning move.

Three things are deliberately fixed rather than tuned:

* the network is the same 4x32 stack `bot/policy/net.py` runs on one CPU core in
  under a millisecond. Training something the sandbox cannot run is wasted GPU.
* observations go through `learn/rlenv.encode_jax`, verified identical to the
  encoder in the submission.
* the value head is training-only and is not exported, so the shipped weights
  stay exactly what the numpy policy expects.

    python -m learn.rl --init /local/data/vng205/clone.npz \\
        --out /local/data/vng205/rl.npz --envs 256 --iters 400
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "generals-bots"))

from bot import features                                      # noqa: E402
from bot.policy.net import CHANNELS, LAYERS                   # noqa: E402
from learn import rlenv                                       # noqa: E402
from learn.train import forward, init_params                  # noqa: E402


def add_value_head(params, key):
    import jax
    import jax.numpy as jnp
    params = dict(params)
    params["val_w"] = jax.random.normal(key, (CHANNELS,)) * 0.01
    params["val_b"] = jnp.zeros(())
    return params


def policy_value(params, x):
    """Logits plus a state value. The trunk is shared; only the value head is
    extra, and it is dropped on export."""
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
    move = conv(h, params["head_w"], params["head_b"])
    flat = jnp.transpose(move, (0, 2, 3, 1)).reshape(x.shape[0], -1)
    pooled = h.mean(axis=(2, 3))
    pass_logit = pooled @ params["pass_w"] + params["pass_b"]
    logits = jnp.concatenate([flat, pass_logit[:, None]], axis=1)
    value = pooled @ params["val_w"] + params["val_b"]
    return logits, value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default=None, help="clone weights to warm start from")
    ap.add_argument("--out", default="/local/data/vng205/rl.npz")
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=192, help="rollout length per iteration")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--entropy", type=float, default=0.001)
    ap.add_argument("--kl", type=float, default=0.05,
                    help="pull toward the warm-start policy; 0 disables")
    ap.add_argument("--shape", type=float, default=0.02,
                    help="dense reward per net tile gained; 0 = terminal only")
    ap.add_argument("--epochs", type=int, default=2, help="PPO epochs per batch")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from generals import GeneralsEnv, get_observation

    print("devices:", jax.devices())
    if not any("cuda" in str(d).lower() for d in jax.devices()):
        print("WARNING: no GPU visible. Did you `unset JAX_PLATFORMS`?")

    env = GeneralsEnv(mode="competition")
    key = jr.PRNGKey(args.seed)
    key, k_pool = jr.split(key)
    pool, _ = env.reset(k_pool)

    key, k_init = jr.split(key)
    params = init_params(k_init)
    if args.init:
        z = np.load(args.init)
        loaded = {k: jnp.asarray(z[k]) for k in z.files}
        missing = [k for k in params if k not in loaded]
        params = {**params, **loaded}
        print(f"warm started from {args.init}"
              + (f" (missing {missing}, kept random)" if missing else ""))
    key, k_val = jr.split(key)
    params = add_value_head(params, k_val)
    anchor = {k: v for k, v in params.items() if not k.startswith("val_")}

    opt = {k: (jnp.zeros_like(v), jnp.zeros_like(v)) for k, v in params.items()}

    def obs_batch(states, seat):
        o = jax.vmap(lambda s: get_observation(s, seat))(states)
        hw = jax.vmap(rlenv.board_dims)(states)
        x = jax.vmap(rlenv.encode_jax)(o, hw[0], hw[1])
        mask = jax.vmap(rlenv.legal_mask_jax)(o)
        return x, mask

    @jax.jit
    def act(params, states, seat, key):
        x, mask = obs_batch(states, seat)
        logits, value = policy_value(params, x)
        logits = jnp.where(mask, logits, -1e9)
        idx = jr.categorical(key, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(idx.shape[0]), idx]
        actions = jax.vmap(rlenv.index_to_engine_action)(idx)
        return actions, idx, logp, value, x, mask

    @jax.jit
    def env_step(states, a0, a1):
        actions = jnp.stack([a0, a1], axis=1)
        return jax.vmap(lambda s, a: env.step(s, a, pool))(states, actions)

    def ppo_loss(p, x, mask, idx, old_logp, adv, ret):
        logits, value = policy_value(p, x)
        logits = jnp.where(mask, logits, -1e9)
        lp = jax.nn.log_softmax(logits)
        # Anchor to the clone. PPO's clip bounds each step, not the total drift,
        # so with a noisy advantage signal a good warm start decays into noise —
        # which is exactly what happened: 0.155 -> 0.080 against our own bot.
        ref_logits, _ = policy_value({**p, **anchor}, x)
        ref_lp = jax.nn.log_softmax(jnp.where(mask, ref_logits, -1e9))
        kl = jnp.sum(jnp.exp(ref_lp) * jnp.where(mask, ref_lp - lp, 0.0), axis=1).mean()
        logp = lp[jnp.arange(idx.shape[0]), idx]
        ratio = jnp.exp(logp - old_logp)
        a = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(ratio * a,
                          jnp.clip(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
        vloss = jnp.mean((value - ret) ** 2)
        probs = jnp.exp(lp) * mask
        ent = -jnp.sum(probs * jnp.where(mask, lp, 0.0), axis=1).mean()
        return pg + 0.5 * vloss - args.entropy * ent + args.kl * kl

    @jax.jit
    def update(p, opt, t, batch):
        loss, g = jax.value_and_grad(ppo_loss)(p, *batch)
        b1, b2, eps = 0.9, 0.999, 1e-8
        new_p, new_opt = {}, {}
        for k in p:
            m, v = opt[k]
            m = b1 * m + (1 - b1) * g[k]
            v = b2 * v + (1 - b2) * g[k] ** 2
            mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
            new_p[k] = p[k] - args.lr * mh / (jnp.sqrt(vh) + eps)
            new_opt[k] = (m, v)
        return new_p, new_opt, loss

    key, k_states = jr.split(key)
    states = jax.vmap(env.init_state)(jr.split(k_states, args.envs))

    step_count = 0
    started = time.time()
    for it in range(args.iters):
        buf = {"x": [], "mask": [], "idx": [], "logp": [], "val": [], "rew": [], "done": []}
        lead = None            # land lead; the env reports it in info.land
        terminals = 0
        for _ in range(args.steps):
            key, k0, k1 = jr.split(key, 3)
            a0, i0, lp0, v0, x0, m0 = act(params, states, 0, k0)
            a1, *_ = act(params, states, 1, k1)
            ts, states = env_step(states, a0, a1)
            done = ts.terminated | ts.truncated
            # A win is hundreds of turns away, so terminal-only reward leaves
            # almost every rollout with no signal at all and the update becomes
            # value noise. Reward the change in the land lead each step; the
            # terminal reward still dominates.
            land = ts.info.land
            new_lead = (land[:, 0] - land[:, 1]).astype(jnp.float32)
            delta = jnp.zeros_like(new_lead) if lead is None else new_lead - lead
            shaped = args.shape * jnp.where(done, 0.0, delta)
            lead = jnp.where(done, 0.0, new_lead)
            terminals += int(done.sum())
            buf["x"].append(x0); buf["mask"].append(m0); buf["idx"].append(i0)
            buf["logp"].append(lp0); buf["val"].append(v0)
            buf["rew"].append(ts.reward[:, 0] + shaped)
            buf["done"].append(done)

        # GAE over the rollout, treating each env column independently
        rew = jnp.stack(buf["rew"])
        val = jnp.stack(buf["val"])
        done = jnp.stack(buf["done"]).astype(jnp.float32)
        adv = jnp.zeros_like(rew)
        last = jnp.zeros(rew.shape[1])
        for t in reversed(range(args.steps)):
            nxt = val[t + 1] if t + 1 < args.steps else val[-1]
            delta = rew[t] + args.gamma * nxt * (1 - done[t]) - val[t]
            last = delta + args.gamma * args.lam * (1 - done[t]) * last
            adv = adv.at[t].set(last)
        ret = adv + val

        flat = (jnp.concatenate(buf["x"]), jnp.concatenate(buf["mask"]),
                jnp.concatenate(buf["idx"]), jnp.concatenate(buf["logp"]),
                adv.reshape(-1), ret.reshape(-1))
        for _ in range(args.epochs):
            step_count += 1
            params, opt, loss = update(params, opt, step_count, flat)

        if it % 5 == 0:
            print(f"iter {it:4d}  loss {float(loss):+.4f}  "
                  f"reward/step {float(rew.mean()):+.4f}  terminals {terminals}  "
                  f"|adv| {float(jnp.abs(adv).mean()):.3f}  "
                  f"{time.time() - started:.0f}s", flush=True)
        if it and it % args.save_every == 0:
            export(params, args.out)
    export(params, args.out)
    print(f"\nwrote {args.out}")
    print(f"  python -m arena.runner --a clone:{args.out} --b ours --games 400 --workers 60")


def export(params, path: str) -> None:
    """Save only what the numpy policy loads — the value head stays behind."""
    keep = {k: np.asarray(v) for k, v in params.items() if not k.startswith("val_")}
    np.savez_compressed(path, **keep)


if __name__ == "__main__":
    main()
