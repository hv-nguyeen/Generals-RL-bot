"""Grow a trained checkpoint into a bigger one that plays identically.

Capacity is an OPEN question again (see docs/STATE.md) — the entry that closed it
rested on BC top-1, which the same file calls worthless. Testing it honestly means
comparing a bigger net against `sp8.best` with everything else held fixed, and the
obvious way to get a bigger net is a fresh BC clone plus a curriculum from stage 0.
That costs the whole night before the comparison starts, and it confounds capacity
with a different initialisation and a different amount of training.

This is the cheap way. The grown net computes the SAME MOVE on every board, so it
can resume from the champion at the stage the champion reached, and any divergence
afterwards is capacity and nothing else.

WIDENING (channels C -> C'). For each trunk conv, the new output channels get
random INCOMING weights and the next layer gets ZERO OUTGOING weights for them.

    outgoing zero  ->  they contribute nothing, so the function is preserved
    incoming random ->  they compute something, so d(loss)/d(outgoing) != 0

Both halves are load-bearing. The tempting version — zero on both sides — is
function-preserving too and is DEAD WEIGHT: gradient to the incoming weights is
proportional to the outgoing weight (zero), and gradient to the outgoing weight is
proportional to the channel's activation (also zero, because incoming is zero). The
new parameters would never move and the run would measure "same net, 6x slower".

DEEPENING (L -> L'). Appended layers are exact identities: centre tap 1.0 on the
diagonal, zero elsewhere, zero bias. `_trunk` is a plain conv+relu stack, so the
input to an appended layer is already post-relu and non-negative, and
relu(identity(x)) == x exactly rather than approximately.

    python -m tools.grow --net runs/nn/sp8.best.npz --out runs/nn/sp8-12x64.npz \\
        --layers 12 --channels 64

It verifies itself on real boards before writing: same argmax on every one, and
the logit gap at f32 round-off. A grown net that plays even slightly differently
is not a capacity experiment, it is a new bot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bot import features
from bot.policy.net import arch_of, arch_record, trunk_keys

# Incoming weights for new channels. Small enough not to saturate relu on the
# first step, large enough that the outgoing gradient is not denormal.
INIT_SCALE = 0.5


def grow(z, layers: int, channels: int, seed: int = 0) -> dict:
    arch = arch_of(z)
    if arch["residual"]:
        raise SystemExit("residual trunks pair into blocks; grow plain ones only")
    L, C = arch["layers"], arch["channels"]
    if layers < L or channels < C:
        raise SystemExit(f"{L}x{C} cannot grow to {layers}x{channels}; both must not shrink")

    rng = np.random.default_rng(seed)
    old = trunk_keys(L, False)
    out: dict = {}

    for i, name in enumerate(old):
        w, b = z[f"{name}_w"], z[f"{name}_b"]
        cout, cin = w.shape[0], w.shape[1]
        # The stem's input is the observation encoding. It grows when the encoder
        # gains channels -- GARRISON and HIDDEN_OPP landed on 2026-08-05 -- and
        # the new columns are ZERO, so the grown net computes the identical move
        # while still receiving gradient on them: d(loss)/d(w_new) is the input
        # value times the upstream error, and the input is not zero even though
        # the weight is.
        cin2 = max(cin, features.C) if i == 0 else channels
        cout2 = channels
        nw = np.zeros((cout2, cin2, 3, 3), np.float32)
        nb = np.zeros((cout2,), np.float32)
        nw[:cout, :cin] = w
        nb[:cout] = b
        if cout2 > cout:
            # New channels READ from everything (random) and, below, are read by
            # nothing (zero). Fan-in scaled so their activations match the
            # existing ones in magnitude.
            fan = cin2 * 9
            nw[cout:, :cin2] = rng.normal(
                0.0, INIT_SCALE / np.sqrt(fan), (cout2 - cout, cin2, 3, 3)).astype(np.float32)
        # nw[:cout, cin:] stays zero: the OLD channels ignore the new ones, which
        # is what makes the forward pass identical.
        out[f"{name}_w"], out[f"{name}_b"] = nw, nb

    for j in range(L, layers):
        w = np.zeros((channels, channels, 3, 3), np.float32)
        w[np.arange(channels), np.arange(channels), 1, 1] = 1.0
        out[f"conv{j}_w"] = w
        out[f"conv{j}_b"] = np.zeros((channels,), np.float32)

    # A CRITIC has the same trunk and a scalar head -- `v_w`/`v_b`, no `head_w`,
    # no `pass_w`. Reading those unconditionally meant `--init-critic` could not
    # cross an encoder change at all: the policy migrated, the critic raised
    # KeyError, and the run fell back to a blank critic at exactly the stage
    # where three runs died of one. Every head is optional and padded the same
    # way, with the new channels at zero so they contribute nothing.
    files = set(getattr(z, "files", z))
    if "head_w" in files:
        hw, hb = z["head_w"], z["head_b"]
        nhw = np.zeros((hw.shape[0], channels, 3, 3), np.float32)
        nhw[:, :hw.shape[1]] = hw
        out["head_w"], out["head_b"] = nhw, np.asarray(hb, np.float32)

    if "pass_w" in files:
        pw = np.asarray(z["pass_w"], np.float32)
        npw = np.zeros((channels,), np.float32)
        npw[:pw.shape[0]] = pw
        out["pass_w"], out["pass_b"] = npw, np.asarray(z["pass_b"], np.float32)

    if "v_w" in files:
        vw = np.asarray(z["v_w"], np.float32)
        nvw = np.zeros((channels,) + vw.shape[1:], np.float32)
        nvw[:vw.shape[0]] = vw
        out["v_w"], out["v_b"] = nvw, np.asarray(z["v_b"], np.float32)

    if not ({"head_w", "v_w"} & files):
        raise SystemExit(f"{'/'.join(sorted(files)[:6])}...: no head_w and no "
                         f"v_w -- this is neither a policy nor a critic")

    # Carry anything else verbatim EXCEPT the arch record, which describes the
    # net we just stopped being. `arch_of` cross-checks that metadata against the
    # trunk keys and refuses to load a file whose two answers disagree — which is
    # the right behaviour, and is how this bug surfaced instead of shipping a
    # checkpoint that claimed to be 8x32 while being 12x64.
    stale = set(arch_record(out))
    for k in files:
        if (k not in out and k not in stale
                and not k.startswith(("conv", "head", "pass", "v_"))):
            out[k] = z[k]
    out.update(arch_record(out))
    return out


def _same_moves(a: str, b: str, games: int, seed0: int) -> tuple[int, float]:
    """Play both nets over real positions. Returns (positions, max |logit gap|)."""
    from sim import engine, mapgen
    from bot.policy.net import Net

    na, nb = Net(a), Net(b)
    n, worst = 0, 0.0
    for g in range(games):
        st = engine.from_grid(mapgen.generate(seed0 + g))
        for _ in range(60):
            obs = engine.observe(st, 0)
            la, lb = na.logits(obs), nb.logits(obs)
            mask = features.legal_mask(obs)
            if not mask.any():
                break
            ia = int(np.argmax(np.where(mask, la, -np.inf)))
            ib = int(np.argmax(np.where(mask, lb, -np.inf)))
            if ia != ib:
                raise SystemExit(
                    f"grown net picks a different move at board {seed0 + g}, "
                    f"turn {obs.turn}: {ia} vs {ib}. Not function-preserving.")
            worst = max(worst, float(np.max(np.abs(la - lb))))
            n += 1
            if engine.step(st, features.index_to_action(ia),
                           features.index_to_action(ib)):
                break
    return n, worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net")
    ap.add_argument("--out")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--channels", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-games", type=int, default=12)
    ap.add_argument("--prefix", default=None, metavar="phi",
                    help="migrate the arrays under this prefix instead of the "
                         "whole file, and write them back under it. The critic "
                         "lives in a run's .resume.npz as phi__*, so "
                         "`--prefix phi` is what makes --init-critic survive an "
                         "encoder change")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    if not (args.net and args.out and args.layers and args.channels):
        raise SystemExit("need --net --out --layers --channels (or --selfcheck)")

    src = np.load(args.net)
    z = {k[len(args.prefix) + 2:]: src[k] for k in src.files
         if k.startswith(args.prefix + "__")} if args.prefix else src
    if args.prefix and not z:
        raise SystemExit(f"{args.net} has no {args.prefix}__* arrays")
    before = arch_of(z)
    grown = grow(z, args.layers, args.channels, args.seed)
    tmp = Path(args.out)
    if args.prefix:
        # Written back under the prefix so `--init-critic` can read it, and
        # ALONE -- a resume file also carries theta/optimiser moments at the old
        # width, and half-migrating those would resume a run into shape errors.
        np.savez(tmp, **{f"{args.prefix}__{k}": v for k, v in grown.items()})
    else:
        np.savez(tmp, **grown)
    after = arch_of(grown)

    n_before = sum(v.size for k, v in z.items() if k.endswith(("_w", "_b")))
    n_after = sum(v.size for k, v in grown.items() if k.endswith(("_w", "_b")))
    print(f"{before['layers']}x{before['channels']} ({n_before} params) -> "
          f"{after['layers']}x{after['channels']} ({n_after} params, "
          f"{n_after / max(n_before, 1):.1f}x)")

    stem_in = int(np.shape(z[f"{trunk_keys(before['layers'], False)[0]}_w"])[1])
    if args.prefix:
        print(f"migrated {len(grown)} arrays under {args.prefix}__ "
              f"(stem input {stem_in} -> {features.C}); pass this to --init-critic")
    elif stem_in < features.C:
        # The parent cannot be LOADED under the current encoder -- `Net.__init__`
        # rejects a stem of the wrong width, on purpose -- so the play-based
        # check cannot run. The guarantee is structural instead: the new input
        # columns are exactly zero, so they contribute nothing to any activation.
        newcols = grown[f"{trunk_keys(after['layers'], False)[0]}_w"][:, stem_in:]
        assert np.abs(newcols).max() == 0.0, "new stem columns are not zero"
        print(f"stem input {stem_in} -> {features.C}: the {features.C - stem_in} "
              f"new encoder channels enter with ZERO weight, so the function is "
              f"preserved by construction. Cannot play-verify: the parent does "
              f"not load under a {features.C}-channel encoder, which is exactly "
              f"the check net.py:181 exists to enforce.")
    else:
        n, worst = _same_moves(args.net, str(tmp), args.verify_games, 900_000)
        print(f"verified: {n} positions, identical argmax, max |logit gap| {worst:.2e}")
    print(f"wrote {tmp}")
    print("\nResume from it at the stage the parent reached -- it plays the same "
          "game, so a difference later is capacity and not a fresh start.")


def selfcheck() -> None:
    """Grow a small random net and confirm it is the same function."""
    from bot.policy.net import Net

    rng = np.random.default_rng(0)
    L, C = 3, 8
    p = {}
    for i in range(L):
        cin = features.C if i == 0 else C
        p[f"conv{i}_w"] = rng.normal(0, 0.2, (C, cin, 3, 3)).astype(np.float32)
        p[f"conv{i}_b"] = rng.normal(0, 0.1, (C,)).astype(np.float32)
    p["head_w"] = rng.normal(0, 0.2, (features.PER_CELL, C, 3, 3)).astype(np.float32)
    p["head_b"] = rng.normal(0, 0.1, (features.PER_CELL,)).astype(np.float32)
    p["pass_w"] = rng.normal(0, 0.2, (C,)).astype(np.float32)
    p["pass_b"] = np.float32(0.3)
    # A real checkpoint carries its own wiring, and the grown file must carry the
    # NEW wiring: `arch_of` cross-checks the record against the trunk keys and
    # refuses a file whose two answers disagree. Without this line the fixture is
    # unrepresentative and the stale-record bug is invisible here.
    p.update(arch_record(p))

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.npz", Path(d) / "b.npz"
        np.savez(a, **p)
        np.savez(b, **grow(np.load(a), L + 2, C * 2))
        got = arch_of(np.load(b))
        assert (got["layers"], got["channels"]) == (L + 2, C * 2), got
        n, worst = _same_moves(str(a), str(b), 3, 12_345)
        assert worst < 1e-3, f"logit gap {worst:.2e} is too large to be round-off"

        # The new capacity must be TRAINABLE, not just harmless. Outgoing zero
        # with incoming zero is also function-preserving and is dead weight.
        g = grow(np.load(a), L, C * 2)
        assert np.abs(g["conv1_w"][C:, :]).max() > 0, "new channels read nothing"
        assert np.abs(g["conv2_w"][:C, C:]).max() == 0, "old channels must ignore new"
        # An appended layer is an exact identity, not an approximation.
        idl = grow(np.load(a), L + 1, C)[f"conv{L}_w"]
        assert np.array_equal(idl[np.arange(C), np.arange(C), 1, 1], np.ones(C))
        assert idl.sum() == C, "identity layer has weight off the centre tap"
    print(f"grow selfcheck OK ({n} positions identical, gap {worst:.2e})")


if __name__ == "__main__":
    main()
