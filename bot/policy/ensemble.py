"""Average several checkpoints' logits, on top of the symmetry averaging.

Test-time augmentation averages ONE net over the eight views of a position and
measured +31.2 Elo for no training. Those eight views are maximally correlated --
same weights, differing only because a stack of 3x3 convolutions is
translation-equivariant and nothing else -- and averaging them was still worth
that much. Independently trained checkpoints are far less correlated, so each one
added attacks a part of the variance the symmetry group cannot reach.

The members available are ~30 Elo below the champion, which is the near-peer
regime where averaging classically beats the best single member. It is also the
regime where it can fail: if the members share a SYSTEMATIC defect rather than
independent noise -- and the ones here all build far fewer castles than the
champion -- averaging reinforces it instead of cancelling it. That is an
empirical question, and it costs 90 seconds an answer in the arena, so it is not
worth arguing about.

Exposes `.net` with the same `logits(obs, tta=, full=)` signature a single `Net`
has, because `bot/policy/guard.py` re-scores through exactly that attribute. A
wrapper that does not provide it makes the guard fall back to `passthrough` and
the shipped artifact stops being the measured one -- which has now happened twice
in this project with `tta` and `full`.
"""

from __future__ import annotations

import numpy as np

from bot import features, rules
from bot.memory import TemporalMemory
from bot.obs import Obs
from bot.policy.net import Net


class EnsembleNet:
    """`Net`-shaped, but the mean over several checkpoints."""

    def __init__(self, paths: list[str], mode: str = "logit"):
        if not paths:
            raise ValueError("an ensemble needs at least one checkpoint")
        self.nets = [Net(p) for p in paths]
        if mode not in ("logit", "prob"):
            raise ValueError(f"mode must be logit or prob, got {mode}")
        self.mode = mode
        self.arch = self.nets[0].arch

    def logits(self, obs: Obs, tta: bool = False, full: bool = False,
               memory=None) -> np.ndarray:
        outs = [n.logits(obs, tta=tta, full=full, memory=memory) for n in self.nets]
        if len(outs) == 1:
            return outs[0]
        if self.mode == "logit":
            # Mean of logits is the GEOMETRIC mean of the distributions: one
            # member that hates a move vetoes it.
            return np.mean(outs, axis=0)
        # Arithmetic mean of probabilities instead: one enthusiastic member can
        # carry a move. Returned as a log so the guard's argsort is unchanged.
        m = np.max(outs, axis=0)
        p = np.mean([np.exp(o - m) for o in outs], axis=0)
        return np.log(p) + m


class EnsemblePolicy:
    """Drop-in agent over several checkpoints. Same shape as `ClonePolicy`."""

    def __init__(self, player_id: int, h: int, w: int, paths: list[str],
                 tta: bool = False, full: bool = False, mode: str = "logit"):
        self.net = EnsembleNet(paths, mode)
        self.H, self.W = h, w
        self.tta = tta
        self.full = full
        self.memory = TemporalMemory(h, w)
        self.last_debug: dict = {}
        # Pay every member's first-forward cost before the first frame, for the
        # same reason ClonePolicy does: construction is outside the move budget.
        warm = np.zeros((features.C, features.PAD, features.PAD), np.float32)
        for n in self.net.nets:
            n._logits_from(warm)
        if tta:
            from bot import symmetry
            for g in symmetry.group(h, w, full):
                symmetry.maps(h, w, g)

    def score_logits(self, obs: Obs) -> np.ndarray:
        self.memory.update(obs)
        return self.net.logits(obs, tta=self.tta, full=self.full,
                               memory=self.memory)

    def act(self, obs: Obs, deadline=None):
        logits = self.score_logits(obs)
        mask = features.legal_mask(obs)
        if not mask.any():
            return rules.PASS_ACTION
        idx = int(np.argmax(np.where(mask, logits, -np.inf)))
        self.last_debug = {"mode": "ensemble", "turn": obs.turn,
                           "members": len(self.net.nets)}
        return features.index_to_action(idx)


def selfcheck() -> None:
    """A one-member ensemble must be the single net, exactly.

    That is the property everything else rests on: if the wrapper changes the
    answer at K=1 it is not averaging, it is a second policy.
    """
    import tempfile
    from pathlib import Path

    from bot.policy.net import arch_record

    rng = np.random.default_rng(0)
    p = {"conv0_w": (rng.normal(size=(32, features.C, 3, 3)) * 0.1).astype(np.float32),
         "conv0_b": np.zeros(32, np.float32)}
    for i in range(1, 8):
        p[f"conv{i}_w"] = (rng.normal(size=(32, 32, 3, 3)) * 0.1).astype(np.float32)
        p[f"conv{i}_b"] = np.zeros(32, np.float32)
    p["head_w"] = (rng.normal(size=(features.PER_CELL, 32, 3, 3)) * 0.1).astype(np.float32)
    p["head_b"] = np.zeros(features.PER_CELL, np.float32)
    p["pass_w"] = (rng.normal(size=32) * 0.01).astype(np.float32)
    p["pass_b"] = np.zeros((), np.float32)
    d = Path(tempfile.mkdtemp())
    np.savez(d / "a.npz", **p, **arch_record(p))
    q = {k: (v + 0.01 if k.endswith("_w") else v) for k, v in p.items()}
    np.savez(d / "b.npz", **q, **arch_record(q))

    from sim import engine, mapgen
    st = engine.from_grid(mapgen.generate(7))
    for _ in range(40):
        engine.step(st, rules.PASS_ACTION, rules.PASS_ACTION)
    obs = engine.observe(st, 0)

    one = EnsembleNet([str(d / "a.npz")])
    solo = Net(str(d / "a.npz"))
    for tta in (False, True):
        assert np.array_equal(one.logits(obs, tta=tta), solo.logits(obs, tta=tta)), \
            f"K=1 ensemble differs from the single net (tta={tta})"

    # two members: the logit mode is the plain mean, and prob mode is not
    two = EnsembleNet([str(d / "a.npz"), str(d / "b.npz")])
    a, b = Net(str(d / "a.npz")).logits(obs), Net(str(d / "b.npz")).logits(obs)
    assert np.allclose(two.logits(obs), (a + b) / 2, atol=1e-5), "logit mode is not the mean"
    pr = EnsembleNet([str(d / "a.npz"), str(d / "b.npz")], mode="prob").logits(obs)
    # log-mean-exp, stabilised elementwise exactly as the implementation does.
    # Shifting by a scalar max instead overflows on the pass logit and compares
    # against the wrong quantity.
    m = np.maximum(a, b)
    want = np.log((np.exp(a - m) + np.exp(b - m)) / 2) + m
    assert np.allclose(pr, want, atol=1e-5), "prob mode is not a log-mean-exp"
    # and it must differ from the logit mean, or the two modes are one mode
    assert not np.allclose(pr, (a + b) / 2, atol=1e-6), "prob mode equals logit mean"
    print("ensemble selfcheck OK (K=1 identical, logit mode is the mean)")


if __name__ == "__main__":
    selfcheck()
