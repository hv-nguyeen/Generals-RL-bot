"""Numpy inference for the cloned policy.

The match sandbox is CPU-only with one core and a 150 ms budget, and `bot/` must
stay numpy-only, so the forward pass is implemented here rather than pulled in
from a framework. Training happens elsewhere (JAX, on a GPU) and exports the
same parameter names into a .npz.

Sized to fit: 4 layers of 32 channels on a 21x21 board is about 14 million
multiply-accumulates, a few milliseconds through BLAS even pinned to one thread.
Anything much larger stops being safe inside the budget.
"""

from __future__ import annotations

import numpy as np

from bot import features, rules
from bot.obs import Obs

CHANNELS = 32
LAYERS = 4


def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(Cin, H, W) x (Cout, Cin, 3, 3) -> (Cout, H, W), zero padded.

    im2col then one matmul: numpy's BLAS does the work, and a hand-rolled loop
    over 9 offsets would be several times slower.
    """
    cin, h, w_ = x.shape
    cout = w.shape[0]
    padded = np.zeros((cin, h + 2, w_ + 2), dtype=np.float32)
    padded[:, 1:h + 1, 1:w_ + 1] = x
    cols = np.empty((cin * 9, h * w_), dtype=np.float32)
    k = 0
    for dy in range(3):
        for dx in range(3):
            cols[k * cin:(k + 1) * cin] = padded[:, dy:dy + h, dx:dx + w_].reshape(cin, -1)
            k += 1
    out = w.reshape(cout, -1) @ cols
    return (out + b[:, None]).reshape(cout, h, w_)


class Net:
    """Loads exported weights and scores a board."""

    def __init__(self, path: str):
        z = np.load(path)
        self.w = [z[f"conv{i}_w"].astype(np.float32) for i in range(LAYERS)]
        self.b = [z[f"conv{i}_b"].astype(np.float32) for i in range(LAYERS)]
        self.head_w = z["head_w"].astype(np.float32)
        self.head_b = z["head_b"].astype(np.float32)
        self.pass_w = z["pass_w"].astype(np.float32)
        self.pass_b = float(z["pass_b"])

    def logits(self, obs: Obs) -> np.ndarray:
        x = features.encode(obs)
        for w, b in zip(self.w, self.b):
            x = _conv3x3(x, w, b)
            np.maximum(x, 0.0, out=x)
        move = _conv3x3(x, self.head_w, self.head_b)          # (8, H, W)
        # (H, W, 8) flattened must match features.action_to_index ordering
        flat = np.transpose(move, (1, 2, 0)).reshape(-1)
        pass_logit = float(self.pass_w @ x.mean(axis=(1, 2)) + self.pass_b)
        return np.concatenate([flat, [pass_logit]])


class ClonePolicy:
    """Drop-in agent: pick the highest-scoring legal action."""

    def __init__(self, player_id: int, h: int, w: int, weights: str):
        self.net = Net(weights)
        self.H, self.W = h, w
        self.last_debug: dict = {}

    def act(self, obs: Obs, deadline=None):
        logits = self.net.logits(obs)
        mask = features.legal_mask(obs)
        if not mask.any():
            return rules.PASS_ACTION
        logits = np.where(mask, logits, -np.inf)
        idx = int(np.argmax(logits))
        self.last_debug = {"mode": "clone", "turn": obs.turn}
        return features.index_to_action(idx)
