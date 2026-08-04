"""Behaviour cloning on the GPU, exported for numpy inference.

Pure JAX plus a hand-rolled Adam — the cluster venv has jax with CUDA but no
torch, flax or optax, and adding dependencies to a machine that already works is
a good way to lose an afternoon.

The architecture is a flag, not a constant: `--layers`, `--channels` and
`--residual` pick it, `bot/policy/net.py` discovers it back out of the exported
npz, and the JAX forward here is derived from the same key names the numpy
forward reads. There is no third place that has to be kept in sync.

    python -m learn.train --data /local/data/vng205/bc --out /local/data/vng205/clone.npz
    python -m learn.train --layers 7 --channels 64 --residual --augment ...
    python -m learn.train --selfcheck          # ~2 s, no data, no GPU, no training

`--augment` is up to 8x more data for free: the board and the action encoding
both transform under the dihedral group, so a rotated game is a real game.
Whether the extra data is worth more capacity is what `tools/scaling.py` answers.

HOW THE VALIDATION NUMBER IS MADE, because a scaling study is only as good as
it. Three whole shards are held out, drawn by a permutation with its own fixed
seed so every architecture is scored on the same games; a shard is a run of
whole replays, so this is a split by game. Each held-out shard is scored
separately and the printed `+-` is the standard error ACROSS those three
blocks — positions inside one game are near-duplicates, so a binomial error bar
over the pooled positions would be off by roughly an order of magnitude. Epochs
are a ceiling with `--patience` early stopping, not a fixed budget: freezing the
budget at the smallest model's optimum is how a capacity study concludes that
capacity does not help.

The result is a sparring partner first and a submission second. Our own gauntlet
cannot punish the mistakes the field punishes, which is the single thing that has
been blocking progress; a policy cloned from strong players can.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from bot import features
from bot.policy.net import (DEFAULT_CHANNELS, DEFAULT_LAYERS, arch_of,
                            arch_record, trunk_keys)


# --------------------------------------------------------------------------
# architecture
def resolve_arch(init: str | None, layers=None, channels=None, residual=None) -> dict:
    """The architecture for a run that may warm start from a checkpoint.

    A checkpoint's own shape wins, always. A flag that disagrees with it is a
    mistake — silently honouring either side produces a run whose weights and
    whose logs describe different networks.
    """
    want = {"layers": layers, "channels": channels, "residual": residual}
    if init:
        have = arch_of(np.load(init))
        bad = {k: f"flag {v} vs checkpoint {have[k]}"
               for k, v in want.items() if v is not None and v != have[k]}
        if bad:
            raise SystemExit(f"--init {init} disagrees with the flags: {bad}")
        return have
    return {"layers": DEFAULT_LAYERS if layers is None else layers,
            "channels": DEFAULT_CHANNELS if channels is None else channels,
            "residual": bool(residual)}


def init_params(key, arch: dict | None = None, cin: int = features.C):
    import jax
    import jax.numpy as jnp

    arch = arch or {"layers": DEFAULT_LAYERS, "channels": DEFAULT_CHANNELS,
                    "residual": False}
    names = trunk_keys(arch["layers"], arch["residual"])
    ch = arch["channels"]
    params = {}
    keys = jax.random.split(key, len(names) + 2)
    prev = cin
    for i, n in enumerate(names):
        # He init: fan_in is 3*3*prev
        scale = np.sqrt(2.0 / (9 * prev))
        # The second conv of a residual block starts at zero, so a fresh residual
        # net is exactly its stem and depth costs nothing until it earns it.
        if n.endswith("b") and n.startswith("res"):
            params[f"{n}_w"] = jnp.zeros((ch, prev, 3, 3))
        else:
            params[f"{n}_w"] = jax.random.normal(keys[i], (ch, prev, 3, 3)) * scale
        params[f"{n}_b"] = jnp.zeros((ch,))
        prev = ch
    params["head_w"] = jax.random.normal(
        keys[len(names)], (features.PER_CELL, prev, 3, 3)) * np.sqrt(2.0 / (9 * prev))
    params["head_b"] = jnp.zeros((features.PER_CELL,))
    params["pass_w"] = jax.random.normal(keys[len(names) + 1], (prev,)) * 0.01
    params["pass_b"] = jnp.zeros(())
    return params


def _conv(inp, w, b):
    import jax
    y = jax.lax.conv_general_dilated(
        inp, w, window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"))
    return y + b[None, :, None, None]


def trunk(params, x):
    """(B, C, H, W) -> relu'd (B, ch, H, W). Mirrors bot/policy/net.py:_trunk.

    The wiring comes from the parameter names, so this function needs no
    architecture argument and cannot be called with the wrong one.
    """
    import jax

    arch = arch_of(params)
    names = trunk_keys(arch["layers"], arch["residual"])
    wb = lambda n: (params[f"{n}_w"], params[f"{n}_b"])          # noqa: E731
    if not arch["residual"]:
        h = x
        for n in names:
            h = jax.nn.relu(_conv(h, *wb(n)))
        return h
    h = _conv(x, *wb(names[0]))
    for a, b in zip(names[1::2], names[2::2]):
        h = h + _conv(jax.nn.relu(_conv(jax.nn.relu(h), *wb(a))), *wb(b))
    return jax.nn.relu(h)


def forward(params, x):
    """x: (B, C, H, W) -> (B, N_ACTIONS). Mirrors bot/policy/net.py exactly."""
    import jax.numpy as jnp

    h = trunk(params, x)
    move = _conv(h, params["head_w"], params["head_b"])     # (B, PER_CELL, H, W)
    flat = jnp.transpose(move, (0, 2, 3, 1)).reshape(x.shape[0], -1)
    pass_logit = h.mean(axis=(2, 3)) @ params["pass_w"] + params["pass_b"]
    return jnp.concatenate([flat, pass_logit[:, None]], axis=1)


def save(params, path) -> None:
    arrays = {k: np.asarray(v) for k, v in params.items()}
    np.savez_compressed(path, **arrays, **arch_record(arrays))


# --------------------------------------------------------------------------
# dihedral augmentation
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))       # bot/features.legal_mask order


@lru_cache(maxsize=None)
def _dihedral_maps(h: int, w: int, g: int):
    """(cell gather, action relabel) for group element g on an h x w board.

    g = 2*k + f: k quarter turns then an optional left-right mirror.

    The board sits at the TOP-LEFT of the 21x21 pad, so the transform is applied
    to the h x w crop and the result is re-placed at the top-left. Rotating the
    padded tensor instead would move the board into another corner and
    manufacture an observation that never occurs at match time.
    """
    pad = features.PAD
    if not (3 <= h <= pad and 3 <= w <= pad):
        raise ValueError(f"board {h}x{w} is not a competition board")
    ids = np.arange(h * w).reshape(h, w)
    k, f = divmod(g, 2)
    t = np.rot90(ids, k)
    if f:
        t = np.fliplr(t)
    h2, w2 = t.shape

    # dst padded cell -> src padded cell. Cells outside the transformed board
    # read the bottom-right pad cell, which encode() leaves zero in every
    # channel; on a full 21x21 board every dst is covered and it is never used.
    srcof = np.full(pad * pad, pad * pad - 1, np.int64)
    src_r, src_c = np.divmod(t.reshape(-1), w)
    dst = (np.arange(h2)[:, None] * pad + np.arange(w2)[None, :]).reshape(-1)
    srcof[dst] = src_r * pad + src_c

    # and the forward direction, src flat id -> dst (row, col)
    fr = np.empty(h * w, np.int64)
    fc = np.empty(h * w, np.int64)
    fr[t.reshape(-1)] = np.repeat(np.arange(h2), w2)
    fc[t.reshape(-1)] = np.tile(np.arange(w2), h2)

    # A rigid motion moves every cell the same way, so read the direction
    # permutation off one interior cell instead of hardcoding eight tables.
    mid = (h // 2) * w + (w // 2)
    dperm = []
    for dr, dc in DIRS:
        nb = mid + dr * w + dc
        v = (int(fr[nb] - fr[mid]), int(fc[nb] - fc[mid]))
        dperm.append(DIRS.index(v))

    per, splits = features.PER_CELL, features.SPLITS
    actmap = np.arange(features.N_ACTIONS, dtype=np.int64)       # pass is a fixed point
    i = np.arange(h * w)
    oldbase = ((i // w) * pad + (i % w)) * per
    newbase = (fr * pad + fc) * per
    for d in range(features.DIRS_N):
        for s in range(splits):
            actmap[oldbase + d * splits + s] = newbase + dperm[d] * splits + s
    # A build is position-only, so it follows the cell and NOT the direction
    # permutation. Leaving it out looks harmless -- actmap starts as the
    # identity -- and silently relabels every augmented build as a move of the
    # unrotated cell, which is a wrong label that trains perfectly cleanly.
    actmap[oldbase + features.BUILD_OFFSET] = newbase + features.BUILD_OFFSET
    return srcof, actmap


def augment(x: np.ndarray, y: np.ndarray, g: int):
    """One dihedral element applied to a whole batch: (B,C,21,21) and (B,) labels.

    Board dimensions come out of the VALID channel, so shards need no metadata
    and existing ones can be augmented as they are. One element per batch rather
    than per example: it is a pure gather that way, and over an epoch the batches
    still cover the group.
    """
    if g % 8 == 0:
        return x, y
    hs = x[:, features.VALID, :, 0].sum(1).astype(int)
    ws = x[:, features.VALID, 0, :].sum(1).astype(int)
    flat = x.reshape(len(x), features.C, -1)
    out = np.empty_like(flat)
    y = y.copy()
    for h, w in set(zip(hs.tolist(), ws.tolist())):
        sel = (hs == h) & (ws == w)
        cellmap, actmap = _dihedral_maps(h, w, g % 8)
        out[sel] = flat[sel][:, :, cellmap]
        y[sel] = actmap[y[sel]]
    return out.reshape(x.shape), y


# --------------------------------------------------------------------------
# Validation split. Fixed and independent of --seed: every architecture in a
# scaling study has to be scored on the SAME held-out games, or the comparison
# is between splits and not between networks.
VAL_SHARDS = 3
VAL_PER_SHARD = 16384
VAL_SPLIT_SEED = 20260803


def shard_list(data: Path) -> list[Path]:
    shards = sorted(data.glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {data} — run `python -m learn.dataset` first")
    # `learn/dataset.py` writes the action-space size it labelled with. A shard
    # built before builds were modelled labels every expert castle build as
    # PASS_INDEX 3528, which under the current scheme decodes to "build at cell
    # 392" -- a wrong label that trains cleanly and cannot be noticed later.
    meta = data / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        for key, want in (("n_actions", features.N_ACTIONS), ("channels", features.C)):
            got = m.get(key)
            if got is not None and int(got) != want:
                raise SystemExit(
                    f"{data} was built with {key}={got}, this build has {want}. "
                    f"Rebuild with `python -m learn.dataset`.")
    else:
        print(f"WARNING no {meta}: cannot check the shards were built for "
              f"{features.N_ACTIONS} actions and {features.C} channels", flush=True)
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
    ap.add_argument("--epochs", type=int, default=20,
                    help="a ceiling, not a plan: --patience ends the run")
    ap.add_argument("--patience", type=int, default=3,
                    help="stop after N epochs with no new best val top-1")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--residual", action="store_true",
                    help="pre-activation residual blocks; needs an odd --layers")
    ap.add_argument("--augment", action="store_true",
                    help="dihedral symmetry, 8x effective data")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    import jax
    import jax.numpy as jnp

    print("devices:", jax.devices())
    arch = resolve_arch(None, args.layers, args.channels, args.residual)
    shards = shard_list(Path(args.data))
    # Held out BY SHARD, and a shard is a run of whole replays, so this is a
    # split by game rather than by position — except for the one replay that
    # straddles each boundary, whose moves land on both sides. Three shards
    # sampled from the whole set, not the tail: `shard_0009` is the last harvest
    # batch, which is a slice of the ladder meta and not a sample of it.
    if len(shards) <= VAL_SHARDS:
        raise SystemExit(f"{len(shards)} shards cannot hold {VAL_SHARDS} out; "
                         "harvest more replays before training")
    pick = np.random.default_rng(VAL_SPLIT_SEED).permutation(len(shards))
    val_shards = [shards[i] for i in pick[:VAL_SHARDS]]
    train_shards = [shards[i] for i in pick[VAL_SHARDS:]]
    val = [(x[:VAL_PER_SHARD], y[:VAL_PER_SHARD])
           for x, y in map(load_shard, val_shards)]
    print(f"{len(shards)} shards ({len(train_shards)} for training, {VAL_SHARDS} "
          f"held out, {sum(len(y) for _, y in val)} val positions), "
          f"{features.N_ACTIONS} classes")

    rng = np.random.default_rng(args.seed)

    key = jax.random.PRNGKey(args.seed)
    params = init_params(key, arch)
    n_par = sum(int(np.asarray(v).size) for v in params.values())
    print(f"{arch['layers']}x{arch['channels']}"
          f"{' residual' if arch['residual'] else ''}, {n_par} parameters"
          f"{', dihedral augmentation' if args.augment else ''}")
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(p) for k, p in params.items()}

    def loss_fn(p, xb, yb):
        logits = forward(p, xb)
        ll = jax.nn.log_softmax(logits)
        return -jnp.mean(ll[jnp.arange(xb.shape[0]), yb])

    @jax.jit
    def step(p, m, v, t, lr, xb, yb):
        loss, g = jax.value_and_grad(loss_fn)(p, xb, yb)
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = {k: b1 * m[k] + (1 - b1) * g[k] for k in p}
        v = {k: b2 * v[k] + (1 - b2) * g[k] ** 2 for k in p}
        mh = {k: m[k] / (1 - b1 ** t) for k in p}
        vh = {k: v[k] / (1 - b2 ** t) for k in p}
        p = {k: p[k] - lr * mh[k] / (jnp.sqrt(vh[k]) + eps) for k in p}
        return p, m, v, loss

    @jax.jit
    def scores(p, xb, yb):
        logits = forward(p, xb)
        ll = jax.nn.log_softmax(logits)
        return (jnp.argmax(logits, axis=1) == yb,
                -ll[jnp.arange(xb.shape[0]), yb])

    def evaluate(p):
        """(top-1, NLL) per held-out shard, NOT pooled.

        Positions inside one game differ by a move and are near-duplicates, so
        the binomial error over 49k positions is a fiction. The spread across
        blocks of different games is the only honest noise scale a single run
        produces, and the scaling study needs that more than another decimal on
        the mean. NLL is reported because top-1 throws away everything about a
        near-tie, which is where a bigger network's gain first shows up.
        """
        out = []
        for x16, y in val:
            ok, nll = [], []
            for i in range(0, len(y), 1024):
                a, n = scores(p, jnp.asarray(x16[i:i + 1024].astype(np.float32)),
                              jnp.asarray(y[i:i + 1024]))
                ok.append(np.asarray(a))
                nll.append(np.asarray(n))
            out.append((float(np.concatenate(ok).mean()),
                        float(np.concatenate(nll).mean())))
        return out

    t = 0
    best, best_epoch, best_nll, best_se, stale = -1.0, -1, float("nan"), 0.0, 0
    for epoch in range(args.epochs):
        # Cosine decay over the FULL budget. Not tuned per size, and it is not
        # meant to be: it removes the one lr failure that scales with width — a
        # constant 2e-3 still thrashing at the end of training — so a big net
        # that loses still loses for a reason other than the last epoch.
        lr = args.lr * 0.5 * (1.0 + np.cos(np.pi * epoch / args.epochs))
        started, running, nb = time.time(), 0.0, 0
        for sp in rng.permutation(len(train_shards)):
            xs16, ys = load_shard(train_shards[sp])
            order = rng.permutation(len(ys))
            for i in range(0, len(order) - args.batch + 1, args.batch):
                idx = order[i:i + args.batch]
                xb = xs16[idx].astype(np.float32)
                yb = ys[idx]
                if args.augment:
                    xb, yb = augment(xb, yb, int(rng.integers(8)))
                t += 1
                params, m, v, loss = step(params, m, v, t, lr,
                                          jnp.asarray(xb), jnp.asarray(yb))
                running += float(loss)
                nb += 1
            del xs16, ys
        blocks = evaluate(params)
        top1 = float(np.mean([b[0] for b in blocks]))
        nll = float(np.mean([b[1] for b in blocks]))
        se = float(np.std([b[0] for b in blocks], ddof=1) / np.sqrt(len(blocks)))
        # Validation accuracy peaks and then falls; keep the best weights rather
        # than whatever the last epoch happened to leave behind. The patience
        # counter is what lets a big net have the epochs it needs without giving
        # a small net the epochs that overfit it — a fixed budget tuned on 4x32
        # would decide the scaling study before it ran.
        marker = ""
        if top1 > best:
            best, best_epoch, best_nll, best_se = top1, epoch, nll, se
            stale, marker = 0, "  <- kept"
            save(params, args.out)
        else:
            stale += 1
        print(f"epoch {epoch}  lr {lr:.2e}  loss {running / max(nb, 1):.4f}  "
              f"val top-1 {top1:.3f} +-{se:.3f}  val nll {nll:.4f}  "
              f"{time.time() - started:.0f}s{marker}", flush=True)
        if stale >= args.patience:
            print(f"early stop: {args.patience} epochs since epoch {best_epoch}")
            break

    out = Path(args.out)
    (out.with_suffix(".json")).write_text(json.dumps(
        {"shards": len(shards), "val_shards": [p.name for p in val_shards],
         "epochs": args.epochs, "best_epoch": best_epoch, "best_val_top1": best,
         "best_val_nll": best_nll, "val_block_se": best_se,
         "params": n_par, "augment": bool(args.augment), "lr": args.lr, **arch},
        indent=2))
    print(f"\nwrote {out} (best val top-1 {best:.3f} +-{best_se:.3f} "
          f"at epoch {best_epoch} of {args.epochs})")
    print("Use it as a sparring partner:")
    print(f"  python -m arena.runner --a ours --b clone:{out} --games 200 --workers 32")


# --------------------------------------------------------------------------
def selfcheck() -> None:
    """Everything that can silently produce a plausible wrong network.

    No data, no GPU, no training — about two seconds, nearly all of it XLA
    compiling five forward passes. If this passes, the numpy forward in the
    submission and the JAX forward that trained the weights are the same
    function at every architecture we can express, an old 4x32 file still plays
    exactly as it did, and the checkpoints that would compute nonsense refuse to
    load at all.
    """
    import tempfile

    import jax
    import jax.numpy as jnp

    from bot import rules
    from bot.obs import Obs
    from bot.policy import net as npnet

    # A CPU-calibrated tolerance fails on GPU, where jax runs f32 convs in TF32
    # and the ~10-bit mantissa costs 1e-2 at these magnitudes. Force true f32 and
    # scale the tolerance by the output magnitude; this test is about the weight
    # layout, not about float formats.
    jax.config.update("jax_default_matmul_precision", "highest")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, features.C, features.PAD, features.PAD)).astype(np.float32)

    def rand_params(arch):
        """Numpy, not init_params: jax.random costs a compile per shape and this
        check is about the forward pass, not about initialisation. Every bias is
        non-zero (a zero bias hides a broadcast bug) and every block's second
        conv is non-zero (a zero one is the identity, which hides a mis-wired
        skip)."""
        p, prev = {}, features.C
        for n in trunk_keys(arch["layers"], arch["residual"]):
            p[f"{n}_w"] = (rng.normal(size=(arch["channels"], prev, 3, 3)) * 0.2).astype("f4")
            p[f"{n}_b"] = (rng.normal(size=arch["channels"]) * 0.2).astype("f4")
            prev = arch["channels"]
        p["head_w"] = (rng.normal(size=(features.PER_CELL, prev, 3, 3)) * 0.2).astype("f4")
        p["head_b"] = (rng.normal(size=features.PER_CELL) * 0.2).astype("f4")
        p["pass_w"] = (rng.normal(size=prev) * 0.2).astype("f4")
        p["pass_b"] = np.float32(0.3)
        return p

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # --- numpy and JAX agree at several shapes, residual on and off
        for layers, channels, residual in ((4, 32, False), (1, 8, False),
                                           (6, 16, False), (3, 24, True),
                                           (7, 16, True)):
            arch = {"layers": layers, "channels": channels, "residual": residual}
            p = rand_params(arch)
            ref = np.asarray(forward(p, jnp.asarray(x)))[0]
            path = td / f"a{layers}x{channels}{int(residual)}.npz"
            save(p, path)
            n = npnet.Net(str(path))
            assert n.arch == arch, (n.arch, arch)
            got = _numpy_logits_of(n, x[0])
            scale = max(float(np.abs(ref).max()), 1.0)
            assert np.abs(got - ref).max() < 1e-4 * scale, \
                f"numpy/JAX disagree at {layers}x{channels} residual={residual}"

        # --- an old-style 4x32 file, written before any of this existed, still
        #     loads and produces byte-identical logits to the old fixed loader
        legacy = {f"conv{i}_w": rng.normal(size=(32, features.C if i == 0 else 32, 3, 3)
                                           ).astype(np.float32) * 0.1 for i in range(4)}
        legacy.update({f"conv{i}_b": rng.normal(size=32).astype(np.float32)
                       for i in range(4)})
        legacy["head_w"] = rng.normal(size=(features.PER_CELL, 32, 3, 3)).astype(np.float32)
        legacy["head_b"] = rng.normal(size=features.PER_CELL).astype(np.float32)
        legacy["pass_w"] = rng.normal(size=32).astype(np.float32)
        legacy["pass_b"] = np.float32(0.25)
        old = td / "legacy.npz"
        np.savez(old, **legacy)                      # no residual marker: 2024 vintage
        n = npnet.Net(str(old))
        assert n.arch == {"layers": 4, "channels": 32, "residual": False}, n.arch
        h = x[0]
        for i in range(4):                           # the loader this replaced
            h = np.maximum(npnet._conv3x3(h, legacy[f"conv{i}_w"], legacy[f"conv{i}_b"]), 0.0)
        mv = npnet._conv3x3(h, legacy["head_w"], legacy["head_b"])
        want = np.concatenate([np.transpose(mv, (1, 2, 0)).reshape(-1),
                               [float(legacy["pass_w"] @ h.mean(axis=(1, 2))
                                      + legacy["pass_b"])]])
        assert np.array_equal(_numpy_logits_of(n, x[0]), want), "legacy logits moved"

        # --- npz round trip, and the refusals
        p = rand_params({"layers": 5, "channels": 12, "residual": True})
        rp = td / "res.npz"
        save(p, rp)
        z = dict(np.load(rp))
        assert int(z["residual"]) == 1 and arch_of(np.load(rp))["layers"] == 5
        # a residual checkpoint renamed into plain conv keys has valid shapes and
        # would run as a 5-layer plain net; the marker is what stops it
        renamed = {"conv0_w": z["conv0_w"], "conv0_b": z["conv0_b"],
                   "residual": z["residual"]}
        for i, n_ in enumerate(["res0a", "res0b", "res1a", "res1b"], start=1):
            renamed[f"conv{i}_w"] = z[f"{n_}_w"]
            renamed[f"conv{i}_b"] = z[f"{n_}_b"]
        renamed.update({k: z[k] for k in ("head_w", "head_b", "pass_w", "pass_b")})
        bad = td / "renamed.npz"
        np.savez(bad, **renamed)
        try:
            npnet.Net(str(bad))
            raise AssertionError("a residual checkpoint loaded as a plain stack")
        except ValueError:
            pass
        # a gap in the numbering is truncation, not a 2-layer net
        gap = {k: v for k, v in legacy.items() if k not in ("conv2_w", "conv2_b")}
        gap["conv4_w"], gap["conv4_b"] = legacy["conv2_w"], legacy["conv2_b"]
        np.savez(td / "gap.npz", **gap)
        try:
            npnet.Net(str(td / "gap.npz"))
            raise AssertionError("a checkpoint with a missing layer loaded")
        except ValueError:
            pass
        # and flags that disagree with an --init are a mistake, not an override
        try:
            resolve_arch(str(rp), layers=4)
            raise AssertionError("--layers 4 accepted against a 5-layer --init")
        except SystemExit:
            pass
        assert resolve_arch(str(rp))["residual"] is True

        # --- the trainer's own init lays down exactly the keys the loader hunts
        p = init_params(jax.random.PRNGKey(1), {"layers": 3, "channels": 2,
                                                "residual": True})
        assert set(p) == {"conv0_w", "conv0_b", "res0a_w", "res0a_b",
                          "res0b_w", "res0b_b", "head_w", "head_b",
                          "pass_w", "pass_b"}, sorted(p)
        assert not np.asarray(p["res0b_w"]).any(), "blocks must start as the identity"
        save(p, td / "init.npz")
        assert arch_of(np.load(td / "init.npz")) == {"layers": 3, "channels": 2,
                                                     "residual": True}

    # --- dihedral augmentation moves the board and the label the same way
    for h, w in ((18, 21), (21, 21), (20, 18)):
        ty = rng.integers(0, 4, size=(h, w)).astype(np.int8)
        ow = rng.integers(-1, 2, size=(h, w)).astype(np.int8)
        ar = rng.integers(0, 9, size=(h, w)).astype(np.int32)
        obs = Obs(H=h, W=w, turn=3, my_land=1, my_army=1, opp_land=1, opp_army=1,
                  type_grid=ty, owner_grid=ow, army_grid=ar)
        enc = features.encode(obs)[None]
        for g in range(8):
            k, f = divmod(g, 2)
            rot = lambda a: np.fliplr(np.rot90(a, k)) if f else np.rot90(a, k)  # noqa: E731
            spun = Obs(H=rot(ty).shape[0], W=rot(ty).shape[1], turn=3, my_land=1,
                       my_army=1, opp_land=1, opp_army=1, type_grid=rot(ty),
                       owner_grid=rot(ow), army_grid=rot(ar))
            gx, _ = augment(enc.copy(), np.zeros(1, np.int64), g)
            assert np.array_equal(gx[0], features.encode(spun)), \
                f"augmented tensor != encoding of the rotated board at g={g}"

            # The cell map is now trusted (it reproduced encode()), so use it as
            # the ground truth for where a labelled move should have gone: the
            # relabelled action must start on the image of its source and point
            # at the image of its destination.
            pad = features.PAD
            to_dst = {int(s): i for i, s in enumerate(_dihedral_maps(h, w, g)[0])}
            for (r0, c0), d0 in (((2, 1), 3), ((0, 0), 1), ((h - 1, w - 1), 0)):
                r1, c1 = r0 + DIRS[d0][0], c0 + DIRS[d0][1]
                lab = np.array([features.action_to_index((rules.MOVE, r0, c0, d0, 1))])
                _, gy = augment(enc.copy(), lab, g)
                _, r, c, d, split = features.index_to_action(int(gy[0]))
                assert split == 1, "the split bit must survive a rotation"
                assert r * pad + c == to_dst[r0 * pad + c0], f"source moved wrong at g={g}"
                assert (r + DIRS[d][0]) * pad + (c + DIRS[d][1]) == to_dst[r1 * pad + c1], \
                    f"augmented action does not point at the moved destination at g={g}"

            # A BUILD follows the cell and NOT the direction permutation, and it
            # must stay a build. Omitting it from actmap is invisible -- actmap
            # starts as the identity -- and relabels every augmented build as a
            # MOVE of the unrotated cell: a wrong label that trains cleanly.
            for r0, c0 in ((2, 1), (0, 0), (h - 1, w - 1)):
                lab = np.array([features.action_to_index((rules.BUILD, r0, c0, 0, 0))])
                _, gy = augment(enc.copy(), lab, g)
                kind, r, c, _, _ = features.index_to_action(int(gy[0]))
                assert kind == rules.BUILD, f"a build became kind {kind} at g={g}"
                assert r * pad + c == to_dst[r0 * pad + c0], f"build moved wrong at g={g}"

    print("train selfcheck OK")


def _numpy_logits_of(net, x):
    """Net.logits takes an Obs; the checks feed a raw encoded tensor."""
    from bot.policy import net as npnet
    h = npnet._trunk(x.copy(), net.layers, net.arch["residual"])
    mv = npnet._conv3x3(h, net.head_w, net.head_b)
    return np.concatenate([np.transpose(mv, (1, 2, 0)).reshape(-1),
                           [float(net.pass_w @ h.mean(axis=(1, 2)) + net.pass_b)]])


if __name__ == "__main__":
    main()
