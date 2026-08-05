"""Ask the critic what it thinks a castle is worth. Ten minutes, no training.

The castle argument has been circular for two days. PPO deletes castles; the
arena says a castle prior is worth +47 Elo; the ladder rejected one. Each of
those is an outcome measurement and none of them says WHY.

There is a mechanism hypothesis that is directly checkable. Under gamma=1 and
`netoracle.LAM = 0.95` the credit window is ~20 plies (that module's own comment),
so the advantage at a build 300 turns from the end contains the terminal outcome
at weight 0.95^299 ~ 2e-7. The policy gradient at a build decision therefore never
reads the win/loss differential. It reads ~20 turns of CRITIC deltas. So the sign
of that advantage is whatever the critic says a castle is worth, and a castle
costs 35 army immediately -- which the critic sees through the army-total scalars
in `features.encode` whether or not it has learned anything about castles.

    dV(castle, free)  ~ 0   ->  the critic does not price the asset at all, the
                               advantage at every build is just minus the cost,
                               and no amount of sampling fixes it. Asymmetric
                               opponents (which hold the castle-count covariate
                               open) are the treatment, not more games.

    dV(castle, free)  > 0   ->  the critic HAS identified it and the problem is
                               elsewhere. Do not build the asymmetric-opponent
                               machinery on this diagnosis.

The same question for the other symmetric blind spot, garrison. Ladder deaths
show the general growing 20->27 on pure regen and never being reinforced, then
emptied with a 40-stack three tiles away. If dV(garrison) ~ 0 the critic cannot
tell a defended general from an empty one and the policy has no reason to send
anything home.

    python -m tools.vprobe --resume runs/nn/sp8.resume.npz --games 24

MECHANISM METRIC, NOT AN ELO CLAIM. It says what the critic believes, which is
how to READ a run. It never justifies promoting a checkpoint.
"""

from __future__ import annotations

import argparse

import numpy as np

from bot import features, rules
from bot.policy import net as pnet


def critic_value(phi_layers, v_w, v_b, obs) -> float:
    """V(s) for one state. Mirrors `valuetrain.forward` + the tanh in selfplay."""
    x = features.encode(obs)
    h = pnet._trunk(x, phi_layers, False)
    return float(np.tanh(h.mean(axis=(1, 2)) @ v_w + v_b))


def _load_phi(path: str):
    z = np.load(path)
    phi = {k[len("phi__"):]: z[k] for k in z.files if k.startswith("phi__")}
    if not phi:
        raise SystemExit(
            f"{path} has no phi__* keys. The critic lives in the RESUME file, not "
            f"in .best.npz or --out -- those carry the policy only, which is why "
            f"every new run refits a critic from scratch.")
    arch = pnet.arch_of(phi)
    layers = [(np.asarray(phi[f"{k}_w"], np.float32), np.asarray(phi[f"{k}_b"], np.float32))
              for k in pnet.trunk_keys(arch["layers"], arch["residual"])]
    return layers, np.asarray(phi["v_w"], np.float32), float(phi["v_b"]), arch


def _rear_tile(obs):
    """An owned plain tile with no visible enemy within 3 — where a build belongs."""
    enemy = np.argwhere(obs.owner_grid == rules.OWNER_OPP)
    for r, c in np.argwhere((obs.owner_grid == rules.OWNER_ME)
                            & (obs.type_grid == rules.T_PLAIN)):
        if not len(enemy) or np.min(np.max(np.abs(enemy - (r, c)), axis=1)) > 3:
            return int(r), int(c)
    return None


def _my_general(obs):
    hit = np.argwhere((obs.type_grid == rules.T_GENERAL)
                      & (obs.owner_grid == rules.OWNER_ME))
    return (int(hit[0][0]), int(hit[0][1])) if len(hit) else None


def probe(resume: str, weights: str, games: int, turns: int, seed0: int) -> dict:
    from sim import engine, mapgen

    phi_layers, v_w, v_b, arch = _load_phi(resume)
    print(f"critic: {arch['layers']}x{arch['channels']} from {resume}")
    pol = pnet.Net(weights)

    free, paid, garr = [], [], []
    for g in range(games):
        # Competition distance: this is where castles are supposed to pay.
        st = engine.from_grid(mapgen.generate(seed0 + g, rules.MIN_GENERALS_DISTANCE, None))
        for _ in range(turns):
            acts = []
            for seat in (0, 1):
                o = engine.observe(st, seat)
                m = features.legal_mask(o)
                if not m.any():
                    acts.append(rules.PASS_ACTION)
                    continue
                acts.append(features.index_to_action(
                    int(np.argmax(np.where(m, pol.logits(o), -np.inf)))))
            if engine.step(st, acts[0], acts[1]):
                break
        obs = engine.observe(st, 0)
        base = critic_value(phi_layers, v_w, v_b, obs)
        site = _rear_tile(obs)
        gen = _my_general(obs)
        if site is None or gen is None:
            continue
        r, c = site

        # (a) the ASSET alone: a castle appears, nothing is paid for it. Isolates
        # whether the critic has learned that a castle is worth anything.
        o2 = engine.observe(st, 0)
        o2.type_grid[r, c] = rules.T_CASTLE
        free.append(critic_value(phi_layers, v_w, v_b, o2) - base)

        # (b) the REAL trade the policy faces: castle, minus 35 army.
        o3 = engine.observe(st, 0)
        o3.type_grid[r, c] = rules.T_CASTLE
        o3.army_grid[r, c] = max(int(o3.army_grid[r, c]) - 35, 1)
        paid.append(critic_value(phi_layers, v_w, v_b, o3) - base)

        # (c) garrison: the ladder deaths, as a counterfactual. Same board, the
        # general holding 25 versus holding 2.
        gr, gc = gen
        o4, o5 = engine.observe(st, 0), engine.observe(st, 0)
        o4.army_grid[gr, gc] = 25
        o5.army_grid[gr, gc] = 2
        garr.append(critic_value(phi_layers, v_w, v_b, o4)
                    - critic_value(phi_layers, v_w, v_b, o5))

    def line(name, xs, reads):
        if not xs:
            print(f"  {name:24s} no samples")
            return 0.0
        a = np.asarray(xs)
        se = a.std() / max(np.sqrt(len(a)), 1)
        print(f"  {name:24s} {a.mean():+.4f} +-{se:.4f}  (n={len(a)})   {reads}")
        return float(a.mean())

    print("\ndV in z units (the reward scale is -1..+1, so 0.01 is 1% of a win):")
    f = line("castle, free", free, "0 -> the asset is not priced")
    p = line("castle, minus 35 army", paid, "sign of the build advantage")
    q = line("garrison 25 vs 2", garr, "0 -> cannot see an undefended general")
    return {"free": f, "paid": p, "garrison": q}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume", help="a run's .resume.npz — the critic lives there")
    ap.add_argument("--weights", help="policy to reach the states with; defaults to the resume's own best")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--turns", type=int, default=150)
    ap.add_argument("--seed0", type=int, default=2_100_000)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    if not (args.resume and args.weights):
        raise SystemExit("need --resume and --weights (or --selfcheck)")

    out = probe(args.resume, args.weights, args.games, args.turns, args.seed0)
    print("\nreading:")
    if abs(out["free"]) < 0.005:
        print("  castle unpriced. The advantage at every build is minus the cost,")
        print("  and more games cannot fix it -- the covariate has no variance to")
        print("  identify from. Asymmetric opponents are the treatment.")
    else:
        print("  the critic DOES price a castle. The blind-critic diagnosis is")
        print("  wrong and the asymmetric-opponent build is not justified by it.")
    if abs(out["garrison"]) < 0.005:
        print("  garrison unpriced: a general holding 25 and one holding 2 look the")
        print("  same, so nothing pushes the policy to send army home.")


def selfcheck() -> None:
    """No checkpoint needed: a random critic must still round-trip the plumbing."""
    from sim import engine, mapgen

    rng = np.random.default_rng(0)
    C, L = 8, 3
    layers = []
    for i in range(L):
        cin = features.C if i == 0 else C
        layers.append((rng.normal(0, 0.2, (C, cin, 3, 3)).astype(np.float32),
                       rng.normal(0, 0.1, (C,)).astype(np.float32)))
    v_w = rng.normal(0, 0.3, (C,)).astype(np.float32)

    st = engine.from_grid(mapgen.generate(11))
    obs = engine.observe(st, 0)
    v = critic_value(layers, v_w, 0.0, obs)
    assert -1.0 <= v <= 1.0, f"tanh'd value out of range: {v}"

    # A turn-0 board owns only the general, so there IS no rear tile — the probe
    # runs at turn ~150 for that reason. Plant one here rather than play 150 turns
    # in a selfcheck.
    gen0 = _my_general(obs)
    assert gen0 is not None
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        rr, cc = gen0[0] + dr, gen0[1] + dc
        if 0 <= rr < obs.H and 0 <= cc < obs.W and obs.type_grid[rr, cc] == rules.T_PLAIN:
            obs.owner_grid[rr, cc] = rules.OWNER_ME
            break

    # The perturbations must actually change the encoding, or the probe would
    # print a confident 0.0000 for every checkpoint and mean nothing.
    site = _rear_tile(obs)
    assert site is not None, "planted tile not found by _rear_tile"
    o2 = engine.observe(st, 0)
    o2.type_grid[site] = rules.T_CASTLE
    assert not np.array_equal(features.encode(obs), features.encode(o2)), \
        "planting a castle did not change the encoding — the probe is blind"

    gen = _my_general(obs)
    o4, o5 = engine.observe(st, 0), engine.observe(st, 0)
    o4.army_grid[gen] = 25
    o5.army_grid[gen] = 2
    assert not np.array_equal(features.encode(o4), features.encode(o5)), \
        "garrison size did not change the encoding"
    print(f"vprobe selfcheck OK (V={v:+.4f}, both perturbations visible to the encoder)")


if __name__ == "__main__":
    main()
