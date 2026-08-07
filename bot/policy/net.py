"""Numpy inference for the cloned policy.

The match sandbox is CPU-only with one core and a 150 ms budget, and `bot/` must
stay numpy-only, so the forward pass is implemented here rather than pulled in
from a framework. Training happens elsewhere (JAX, on a GPU) and exports the
same parameter names into a .npz.

The architecture is not assumed, it is DISCOVERED. A checkpoint describes itself
twice over: the trunk key names give the wiring (`conv{i}` for a plain stack,
`conv0` plus `res{i}a`/`res{i}b` pairs for a pre-activation residual stack) and
the weight shapes give every width. Both must agree with the `layers` /
`channels` / `residual` record written alongside them, otherwise loading fails.
A tail-truncated trunk is invisible to the key scan and visible only to the
recorded depth, which is why the depth is recorded. That redundancy is
deliberate — the failure this module has already suffered once is a network that
loads cleanly and computes nonsense, and a silently truncated trunk is exactly
that failure: drop `conv4`/`conv5` from a 6-layer file and the head still
matmuls, the logits still have the right shape, and the bot still plays.

Measured single-core, 21x21, 12 input channels: 4x32 = 0.4 ms/move, 8x64 = 1.2,
10x128 = 3.0. The 150 ms move budget is not what limits depth here — PPO rollout
throughput is.
"""

from __future__ import annotations

import numpy as np

from bot import features, rules, symmetry
from bot.obs import Obs

# Only the starting point for a fresh `learn/train.py` run. Nothing at inference
# time reads these: a loaded checkpoint always wins.
DEFAULT_CHANNELS = 32
DEFAULT_LAYERS = 4


def trunk_keys(layers: int, residual: bool) -> list[str]:
    """Trunk parameter prefixes, in evaluation order.

    Plain: `conv0..conv{L-1}`, each conv+relu. Residual: `conv0` is the stem and
    the remaining layers pair into pre-activation blocks, so the count must be
    odd — a leftover half-block would have to be wired as something other than a
    block, and a trunk with two wirings is a trunk nobody can load blind.
    """
    if layers < 1:
        raise ValueError(f"layers must be >= 1, got {layers}")
    if not residual:
        return [f"conv{i}" for i in range(layers)]
    if layers < 3 or layers % 2 == 0:
        raise ValueError(
            f"--residual needs an odd layer count >= 3 (1 stem + 2 per block), got {layers}")
    return ["conv0"] + [f"res{i}{ab}" for i in range((layers - 1) // 2) for ab in "ab"]


def _run_length(files: set[str], pat: str) -> int:
    n = 0
    while pat.format(n) in files:
        n += 1
    return n


def arch_of(z) -> dict:
    """`{'layers', 'channels', 'residual'}` from an npz or a parameter dict.

    Everything suspicious raises. A gap in the numbering (`conv0,conv1,conv3`)
    would otherwise be read as a 2-layer net and the rest ignored, which is the
    truncation bug this module exists to make impossible.
    """
    files = set(getattr(z, "files", z))
    convs = _run_length(files, "conv{}_w")
    blocks = _run_length(files, "res{}a_w")
    residual = blocks > 0
    if not convs:
        raise ValueError("no conv0_w: this is not a policy or value checkpoint")
    stray = (sum(k.startswith("conv") and k.endswith("_w") for k in files) - convs
             + sum(k.startswith("res") and k.endswith("_w") for k in files) - 2 * blocks)
    if stray:
        raise ValueError("trunk keys are not contiguous: "
                         f"{sorted(k for k in files if k.endswith('_w'))}")
    if residual and convs != 1:
        raise ValueError(f"a residual trunk has exactly one stem conv, found {convs}")
    # .shape, not np.asarray(...).shape: the trainers call this on dicts of jax
    # tracers inside jit, where converting to numpy is an error
    arch = {"layers": 1 + 2 * blocks if residual else convs,
            "channels": int(z["conv0_w"].shape[0]),
            "residual": residual}
    # The key names cannot express a trunk truncated at the TAIL: drop conv4 and
    # conv5 from a 6-layer file and the scan simply stops at conv3, reports a
    # 4-layer net, and everything downstream still has valid shapes. The only
    # defence is the saver writing down what it meant, so any marker present
    # must agree — a disagreement is never a thing to guess about.
    for k, v in arch.items():
        if k in files and int(z[k]) != int(v):
            raise ValueError(f"checkpoint says {k}={int(z[k])} but the trunk keys "
                             f"say {k}={int(v)}; refusing to guess")
    return arch


def arch_record(params: dict) -> dict:
    """Extra npz entries a saver must write so the file states its own wiring.

    All three, not just `residual`: the key names recover the wiring and the
    widths but not the depth the writer intended, so depth is the one the file
    has to carry.
    """
    a = arch_of(params)
    return {"residual": np.int8(a["residual"]), "layers": np.int16(a["layers"]),
            "channels": np.int16(a["channels"])}


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
    # cols is laid out (kh, kw, cin); the weights are stored (cout, cin, kh, kw),
    # so they must be transposed to match before flattening. Getting this wrong
    # produces a plausible-looking network that computes nonsense — it cost a
    # 200-0 arena result before the two paths were compared directly.
    out = w.transpose(0, 2, 3, 1).reshape(cout, -1) @ cols
    return (out + b[:, None]).reshape(cout, h, w_)


def _trunk(x: np.ndarray, layers: list, residual: bool) -> np.ndarray:
    """Shared body of both heads. Returns a relu'd (Cout, H, W) feature map."""
    if not residual:
        for w, b in layers:
            x = _conv3x3(x, w, b)
            np.maximum(x, 0.0, out=x)
        return x
    (w0, b0), rest = layers[0], layers[1:]
    h = _conv3x3(x, w0, b0)
    # Pre-activation: relu lives inside the block so the skip path stays linear
    # all the way through, which is the whole reason deep stacks train.
    for (wa, ba), (wb, bb) in zip(rest[0::2], rest[1::2]):
        h = h + _conv3x3(np.maximum(_conv3x3(np.maximum(h, 0.0), wa, ba), 0.0), wb, bb)
    return np.maximum(h, 0.0)


def _load_trunk(z) -> tuple[list, dict]:
    arch = arch_of(z)
    layers = [(z[f"{n}_w"].astype(np.float32), z[f"{n}_b"].astype(np.float32))
              for n in trunk_keys(arch["layers"], arch["residual"])]
    return layers, arch


class Net:
    """Loads exported weights and scores a board."""

    def __init__(self, path: str):
        z = np.load(path)
        self.layers, self.arch = _load_trunk(z)
        self.head_w = z["head_w"].astype(np.float32)
        self.head_b = z["head_b"].astype(np.float32)
        # The action space gained a build slot per cell (8 -> 9). A stale head
        # loads without complaint here and dies later inside `np.where(mask,
        # logits, -inf)` on a broadcast error -- mid-match, after the handshake,
        # from a line that says nothing about checkpoints.
        if self.head_w.shape[0] != features.PER_CELL:
            raise ValueError(
                f"{path}: head is {self.head_w.shape[0]}-wide, this build needs "
                f"{features.PER_CELL}. The action space changed (castle builds "
                f"are modelled now); retrain, do not reuse.")
        # Same story one layer up: the encoder gained the broadcast scalar
        # channels, so a stem from before them is a different function of a
        # different board. It would otherwise die in a BLAS shape error deep
        # inside _conv3x3, mid-match, naming neither the file nor the reason.
        if self.layers[0][0].shape[1] != features.C:
            raise ValueError(
                f"{path}: stem takes {self.layers[0][0].shape[1]} input channels, "
                f"this build encodes {features.C}. The observation changed (turn "
                f"and the army/land totals are planes now); retrain, do not reuse.")
        self.pass_w = z["pass_w"].astype(np.float32)
        self.pass_b = float(z["pass_b"])

    def _logits_from(self, enc: np.ndarray) -> np.ndarray:
        x = _trunk(enc, self.layers, self.arch["residual"])
        move = _conv3x3(x, self.head_w, self.head_b)   # (PER_CELL, H, W)
        # (H, W, PER_CELL) flattened must match features.action_to_index ordering
        flat = np.transpose(move, (1, 2, 0)).reshape(-1)
        pass_logit = float(self.pass_w @ x.mean(axis=(1, 2)) + self.pass_b)
        return np.concatenate([flat, [pass_logit]])

    def logits(self, obs: Obs, tta: bool = False, full: bool = False) -> np.ndarray:
        """Masked-move logits, optionally averaged over the board's symmetries.

        The rules are symmetric under the eight rigid motions of the square, so
        the same position rotated is the same position. A stack of 3x3 convs is
        translation-equivariant and nothing else, so it answers each orientation
        slightly differently; averaging is variance reduction with no training.

        The budget pays for it easily -- one forward is ~1.2 ms against a 150 ms
        limit -- and this is the only axis of the problem nobody has spent.
        """
        enc = features.encode(obs)
        if not tta:
            return self._logits_from(enc)
        els = symmetry.group(obs.H, obs.W, full)
        return symmetry.average_logits(enc, obs.H, obs.W, self._logits_from, els)


class ClonePolicy:
    """Drop-in agent: pick the highest-scoring legal action."""

    def __init__(self, player_id: int, h: int, w: int, weights: str,
                 tta: bool = False, full: bool = False):
        self.net = Net(weights)
        self.H, self.W = h, w
        self.tta = tta
        self.full = full
        self.last_debug: dict = {}
        # Pay the one-time costs HERE, before the first frame. Construction is
        # outside the per-move budget; the first move is not.
        #
        # `symmetry.maps` is lru_cached, so without this the first TTA move
        # builds every cell-gather and action-relabel table for the board -- and
        # the first forward pays numpy's allocation and BLAS warm-up on top.
        # Measured through the real wire protocol: 110 ms of a 150 ms limit on
        # move one, against 5 ms in steady state. That margin is a forfeit on a
        # slower box, and a forfeit is a loss.
        warm = np.zeros((features.C, features.PAD, features.PAD), np.float32)
        self.net._logits_from(warm)
        if tta:
            for g in symmetry.group(h, w, full):
                symmetry.maps(h, w, g)

    def act(self, obs: Obs, deadline=None):
        logits = self.net.logits(obs, tta=self.tta, full=self.full)
        mask = features.legal_mask(obs)
        if not mask.any():
            return rules.PASS_ACTION
        logits = np.where(mask, logits, -np.inf)
        idx = int(np.argmax(logits))
        self.last_debug = {"mode": "clone", "turn": obs.turn, "tta": self.tta}
        return features.index_to_action(idx)


class ValueNet:
    """Win probability for a position, from the field-outcome model.

    Same trunk as the policy, a scalar head. Used to score positions our own bot
    reaches against what actually wins against real opponents — the local
    gauntlet cannot do that, because none of our opponents punish the mistakes
    the field punishes.
    """

    def __init__(self, path: str):
        z = np.load(path)
        self.layers, self.arch = _load_trunk(z)
        self.head_w = z["v_w"].astype(np.float32)
        self.head_b = float(z["v_b"])

    def win_prob(self, obs: Obs) -> float:
        x = _trunk(features.encode(obs), self.layers, self.arch["residual"])
        logit = float(self.head_w @ x.mean(axis=(1, 2)) + self.head_b)
        return 1.0 / (1.0 + np.exp(-logit))
