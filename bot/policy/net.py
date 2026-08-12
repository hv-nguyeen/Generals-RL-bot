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
from bot.memory import TemporalMemory
from bot.obs import Obs

# Only the starting point for a fresh `learn/train.py` run. Nothing at inference
# time reads these: a loaded checkpoint always wins.
DEFAULT_CHANNELS = 32
DEFAULT_LAYERS = 4
DEFAULT_CONTEXT = True

# The strategy path keeps the board's coarse topology instead of averaging it
# away.  Nine ordered regional means plus a global mean and maximum form the
# token vector; the MLP emits a separate scale/bias pair for each target region.
STRATEGY_REGIONS = 3
STRATEGY_TOKEN_BLOCKS = 2 + STRATEGY_REGIONS * STRATEGY_REGIONS
STRATEGY_OUTPUT_BLOCKS = 2 * STRATEGY_REGIONS * STRATEGY_REGIONS


def strategy_keys() -> set[str]:
    """All-or-nothing parameter set for the global spatial strategy path."""
    return {
        "strategy_w1", "strategy_b1", "strategy_w2", "strategy_b2",
        "strategy_ref1_w", "strategy_ref1_b",
        "strategy_ref2_w", "strategy_ref2_b",
    }


def strategy_spec(z) -> dict | None:
    """Validate and describe an optional coarse-to-local strategy module.

    This is separate from the convolutional trunk wiring so old checkpoints keep
    their historical four-field ``arch_of`` result.  A strategy checkpoint adds
    ``strategy`` and ``strategy_hidden`` to that result; absence adds nothing.
    """
    files = set(getattr(z, "files", z))
    keys = strategy_keys()
    present = keys & files
    if present and present != keys:
        raise ValueError(f"incomplete strategy module: found {sorted(present)}")
    if not present:
        if "strategy" in files and int(z["strategy"]) != 0:
            raise ValueError("checkpoint says strategy=1 but has no strategy weights")
        if "strategy_hidden" in files and int(z["strategy_hidden"]) != 0:
            raise ValueError("checkpoint has strategy_hidden but no strategy weights")
        return None

    ch = int(z["conv0_w"].shape[0])
    w1 = tuple(z["strategy_w1"].shape)
    if len(w1) != 2 or w1[0] != STRATEGY_TOKEN_BLOCKS * ch or w1[1] < 1:
        raise ValueError(
            f"strategy_w1 must be ({STRATEGY_TOKEN_BLOCKS * ch}, hidden), found {w1}")
    hidden = int(w1[1])
    expected = {
        "strategy_b1": (hidden,),
        "strategy_w2": (hidden, STRATEGY_OUTPUT_BLOCKS * ch),
        "strategy_b2": (STRATEGY_OUTPUT_BLOCKS * ch,),
        "strategy_ref1_w": (ch, ch, 3, 3),
        "strategy_ref1_b": (ch,),
        "strategy_ref2_w": (ch, ch, 3, 3),
        "strategy_ref2_b": (ch,),
    }
    bad = {k: (tuple(z[k].shape), shape) for k, shape in expected.items()
           if tuple(z[k].shape) != shape}
    if bad:
        raise ValueError(f"invalid strategy shapes: {bad}")
    if "strategy" in files and int(z["strategy"]) != 1:
        raise ValueError("checkpoint has strategy weights but says strategy=0")
    if "strategy_hidden" in files and int(z["strategy_hidden"]) != hidden:
        raise ValueError(
            f"checkpoint says strategy_hidden={int(z['strategy_hidden'])}, "
            f"weights say {hidden}")
    return {"hidden": hidden}


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
    """Architecture and context wiring from an npz or parameter dict.

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
    context_keys = {"context_global", "context_region"}
    present_context = context_keys & files
    if present_context and present_context != context_keys:
        raise ValueError(f"incomplete context mixer: found {sorted(present_context)}")
    context_dense = False
    if present_context:
        shapes = {tuple(z[k].shape) for k in context_keys}
        ch = int(z["conv0_w"].shape[0])
        if len(shapes) != 1 or next(iter(shapes)) not in {(ch,), (ch, ch)}:
            raise ValueError(f"context mixers must both be ({ch},) or ({ch},{ch}), "
                             f"found {sorted(shapes)}")
        context_dense = len(next(iter(shapes))) == 2
    spec = strategy_spec(z)
    arch = {"layers": 1 + 2 * blocks if residual else convs,
            "channels": int(z["conv0_w"].shape[0]),
            "residual": residual, "context": bool(present_context)}
    if spec is not None:
        arch.update(strategy=True, strategy_hidden=spec["hidden"])
    # The key names cannot express a trunk truncated at the TAIL: drop conv4 and
    # conv5 from a 6-layer file and the scan simply stops at conv3, reports a
    # 4-layer net, and everything downstream still has valid shapes. The only
    # defence is the saver writing down what it meant, so any marker present
    # must agree — a disagreement is never a thing to guess about.
    for k, v in arch.items():
        if k in files and int(z[k]) != int(v):
            raise ValueError(f"checkpoint says {k}={int(z[k])} but the trunk keys "
                             f"say {k}={int(v)}; refusing to guess")
    if "context_dense" in files and int(z["context_dense"]) != int(context_dense):
        raise ValueError(f"checkpoint says context_dense={int(z['context_dense'])} "
                         f"but context weights say {int(context_dense)}")
    if ("input_channels" in files
            and int(z["input_channels"]) != int(z["conv0_w"].shape[1])):
        raise ValueError(f"checkpoint says input_channels={int(z['input_channels'])} "
                         f"but conv0_w takes {z['conv0_w'].shape[1]}")
    return arch


def arch_record(params: dict) -> dict:
    """Extra npz entries a saver must write so the file states its own wiring.

    The record includes residual wiring, depth, width, context, and input width.
    Key names recover most wiring but not every intent, so the redundant record
    is cross-checked on load rather than trusted blindly.
    """
    a = arch_of(params)
    return {"residual": np.int8(a["residual"]), "layers": np.int16(a["layers"]),
            "channels": np.int16(a["channels"]),
            "context": np.int8(a["context"]),
            "context_dense": np.int8(
                a["context"] and np.ndim(params["context_global"]) == 2),
            "strategy": np.int8(a.get("strategy", False)),
            "strategy_hidden": np.int16(a.get("strategy_hidden", 0)),
            "input_channels": np.int16(params["conv0_w"].shape[1])}


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


def _context_mix(h: np.ndarray, valid: np.ndarray,
                 global_scale: np.ndarray | None,
                 region_scale: np.ndarray | None) -> np.ndarray:
    """Inject whole-board and 3x3 regional summaries into every local cell."""
    if global_scale is None:
        return h
    v = valid.astype(np.float32)
    global_mean = (h * v[None]).sum(axis=(1, 2)) / max(float(v.sum()), 1.0)
    regional = np.zeros_like(h)
    edges = (0, 7, 14, features.PAD)
    for ri in range(3):
        for ci in range(3):
            rs, cs = slice(edges[ri], edges[ri + 1]), slice(edges[ci], edges[ci + 1])
            vv = v[rs, cs]
            mean = (h[:, rs, cs] * vv[None]).sum(axis=(1, 2)) / max(float(vv.sum()), 1.0)
            regional[:, rs, cs] = mean[:, None, None]
    if global_scale.ndim == 1:
        global_term = global_scale * global_mean
        region_term = region_scale[:, None, None] * regional
    else:
        global_term = global_scale @ global_mean
        region_term = np.einsum("oi,ihw->ohw", region_scale, regional)
    out = h + global_term[:, None, None] + region_term
    return np.maximum(out, 0.0)


def _strategy_tokens(h: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Ordered global/region summary used by both policy and value trunks."""
    v = valid.astype(np.float32)
    den = max(float(v.sum()), 1.0)
    mean = (h * v[None]).sum(axis=(1, 2)) / den
    maximum = np.where(v[None] > 0, h, -1e9).max(axis=(1, 2))
    parts = [mean, maximum]
    edges = (0, 7, 14, features.PAD)
    for ri in range(STRATEGY_REGIONS):
        for ci in range(STRATEGY_REGIONS):
            rs, cs = slice(edges[ri], edges[ri + 1]), slice(edges[ci], edges[ci + 1])
            vv = v[rs, cs]
            parts.append((h[:, rs, cs] * vv[None]).sum(axis=(1, 2))
                         / max(float(vv.sum()), 1.0))
    return np.concatenate(parts)


def _strategy_mix(h: np.ndarray, valid: np.ndarray,
                  params: dict | None) -> np.ndarray:
    """Condition every region on all regions, then run residual refinement.

    ``strategy_w2`` and ``strategy_ref2_w`` are zero in a migrated checkpoint.
    Consequently this function is exactly the identity at migration time, while
    both final layers receive gradients immediately through non-zero hidden and
    first-refinement activations.
    """
    if params is None:
        return h
    tokens = _strategy_tokens(h, valid)
    hidden = np.maximum(tokens @ params["strategy_w1"] + params["strategy_b1"], 0.0)
    ch = h.shape[0]
    controls = (hidden @ params["strategy_w2"] + params["strategy_b2"]
                ).reshape(STRATEGY_REGIONS * STRATEGY_REGIONS, 2, ch)
    conditioned = h.copy()
    edges = (0, 7, 14, features.PAD)
    k = 0
    for ri in range(STRATEGY_REGIONS):
        for ci in range(STRATEGY_REGIONS):
            rs, cs = slice(edges[ri], edges[ri + 1]), slice(edges[ci], edges[ci + 1])
            scale, bias = controls[k]
            conditioned[:, rs, cs] = (
                h[:, rs, cs] * (1.0 + scale[:, None, None])
                + bias[:, None, None])
            k += 1
    np.maximum(conditioned, 0.0, out=conditioned)
    refine = _conv3x3(conditioned, params["strategy_ref1_w"],
                      params["strategy_ref1_b"])
    np.maximum(refine, 0.0, out=refine)
    refine = _conv3x3(refine, params["strategy_ref2_w"],
                      params["strategy_ref2_b"])
    return np.maximum(conditioned + refine, 0.0)


def _load_strategy(z) -> dict | None:
    if strategy_spec(z) is None:
        return None
    return {k: z[k].astype(np.float32) for k in strategy_keys()}


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
        stem_c = self.layers[0][0].shape[1]
        if stem_c in features.LEGACY_INPUT_CHANNELS and features.C > stem_c:
            # Function-preserving migration: the old policy ignores every new
            # temporal channel until training gives those zero columns weight.
            w, b = self.layers[0]
            grown = np.zeros((w.shape[0], features.C, 3, 3), np.float32)
            grown[:, :stem_c] = w
            self.layers[0] = (grown, b)
        elif stem_c != features.C:
            raise ValueError(
                f"{path}: stem takes {stem_c} input channels, "
                f"this build encodes {features.C}. The observation changed (turn "
                f"and the army/land totals are planes now); retrain, do not reuse.")
        self.pass_w = z["pass_w"].astype(np.float32)
        self.pass_b = float(z["pass_b"])
        self.context_global = (z["context_global"].astype(np.float32)
                               if self.arch["context"] else None)
        self.context_region = (z["context_region"].astype(np.float32)
                               if self.arch["context"] else None)
        self.strategy = _load_strategy(z)

    def _logits_from(self, enc: np.ndarray) -> np.ndarray:
        x = _trunk(enc, self.layers, self.arch["residual"])
        x = _context_mix(x, enc[features.VALID], self.context_global,
                         self.context_region)
        x = _strategy_mix(x, enc[features.VALID], self.strategy)
        move = _conv3x3(x, self.head_w, self.head_b)   # (PER_CELL, H, W)
        # (H, W, PER_CELL) flattened must match features.action_to_index ordering
        flat = np.transpose(move, (1, 2, 0)).reshape(-1)
        pass_logit = float(self.pass_w @ x.mean(axis=(1, 2)) + self.pass_b)
        return np.concatenate([flat, [pass_logit]])

    def logits(self, obs: Obs, tta: bool = False, full: bool = False,
               memory=None) -> np.ndarray:
        """Masked-move logits, optionally averaged over the board's symmetries.

        The rules are symmetric under the eight rigid motions of the square, so
        the same position rotated is the same position. A stack of 3x3 convs is
        translation-equivariant and nothing else, so it answers each orientation
        slightly differently; averaging is variance reduction with no training.

        The budget pays for it easily -- one forward is ~1.2 ms against a 150 ms
        limit -- and this is the only axis of the problem nobody has spent.
        """
        enc = features.encode(obs, memory)
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
        self.memory = TemporalMemory(h, w)
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

    def score_logits(self, obs: Obs) -> np.ndarray:
        self.memory.update(obs)
        return self.net.logits(obs, tta=self.tta, full=self.full,
                               memory=self.memory)

    def act(self, obs: Obs, deadline=None):
        logits = self.score_logits(obs)
        mask = features.legal_mask(obs)
        if not mask.any():
            return rules.PASS_ACTION
        logits = np.where(mask, logits, -np.inf)
        idx = int(np.argmax(logits))
        self.last_debug = {"mode": "clone", "turn": obs.turn, "tta": self.tta}
        return features.index_to_action(idx)


class ValueNet:
    """Direct game value for a position, from the field-outcome model.

    Same trunk as the policy, a scalar head. Used to score positions our own bot
    reaches against what actually wins against real opponents — the local
    gauntlet cannot do that, because none of our opponents punish the mistakes
    the field punishes.
    """

    def __init__(self, path: str):
        raw = np.load(path)
        files = set(raw.files)
        bare = "conv0_w" in files
        prefixed = "phi__conv0_w" in files
        if bare and prefixed:
            raise ValueError(f"{path}: contains both bare and phi__ critic arrays")
        if not bare and prefixed:
            # PPO saves its matched critic under ``phi__*`` so policy and value
            # arrays cannot be confused in a resume file.  Deployment search
            # historically accepted only standalone valuetrain checkpoints,
            # which made the accepted policy's actual matched critic impossible
            # to audit.  Strip the namespace in memory; never rewrite the source.
            z = {k[5:]: raw[k] for k in raw.files if k.startswith("phi__")}
        else:
            z = raw
        self.layers, self.arch = _load_trunk(z)
        stem_c = self.layers[0][0].shape[1]
        if stem_c in features.LEGACY_INPUT_CHANNELS and features.C > stem_c:
            w, b = self.layers[0]
            grown = np.zeros((w.shape[0], features.C, 3, 3), np.float32)
            grown[:, :stem_c] = w
            self.layers[0] = (grown, b)
        elif stem_c != features.C:
            raise ValueError(f"{path}: critic stem has {stem_c} channels, expected "
                             f"{features.C}")
        self.head_w = z["v_w"].astype(np.float32)
        self.head_b = float(z["v_b"])
        self.value_residual = None
        residual_keys = {"v_res1_w", "v_res1_b", "v_res2_w", "v_res2_b"}
        value_files = set(getattr(z, "files", z))
        present = residual_keys & value_files
        if present and present != residual_keys:
            raise ValueError(f"{path}: incomplete residual value head: {sorted(present)}")
        if present:
            self.value_residual = tuple(np.asarray(z[k], np.float32) for k in
                                        ("v_res1_w", "v_res1_b",
                                         "v_res2_w", "v_res2_b"))
        if "value_schema" in value_files:
            self.schema = int(z["value_schema"])
        else:
            # Online PPO critics predate the standalone value-schema marker but
            # use valuetrain's 11-block (mean, max, 3x3 means) pooled head.  Its
            # width is unambiguous, so infer it instead of misreading a 352-wide
            # 32-channel head as the legacy 32-wide global-mean head.
            pooled_width = self.arch["channels"] * (2 + 3 * 3)
            self.schema = 2 if self.head_w.shape[0] == pooled_width else 1
        if self.schema >= 2 and self.head_w.ndim != 1:
            raise ValueError(
                f"{path}: distributional PPO critic head {self.head_w.shape} "
                "cannot be used as a scalar ValueNet")
        self.context_global = (z["context_global"].astype(np.float32)
                               if self.arch["context"] else None)
        self.context_region = (z["context_region"].astype(np.float32)
                               if self.arch["context"] else None)
        self.strategy = _load_strategy(z)
        self.memory = None

    @staticmethod
    def _pooled(h: np.ndarray, valid: np.ndarray) -> np.ndarray:
        v = valid.astype(np.float32)
        mean = (h * v[None]).sum(axis=(1, 2)) / max(float(v.sum()), 1.0)
        maximum = np.where(v[None] > 0, h, -1e9).max(axis=(1, 2))
        parts = [mean, maximum]
        edges = (0, 7, 14, features.PAD)
        for ri in range(3):
            for ci in range(3):
                vv = v[edges[ri]:edges[ri + 1], edges[ci]:edges[ci + 1]]
                q = h[:, edges[ri]:edges[ri + 1], edges[ci]:edges[ci + 1]]
                parts.append((q * vv[None]).sum(axis=(1, 2))
                             / max(float(vv.sum()), 1.0))
        return np.concatenate(parts)

    def value(self, obs: Obs) -> float:
        if (self.memory is None or self.memory.H != obs.H or self.memory.W != obs.W
                or obs.turn < self.memory.turn):
            self.memory = TemporalMemory(obs.H, obs.W)
        self.memory.update(obs)
        enc = features.encode(obs, self.memory)
        return self.value_encoded(enc)

    def value_encoded(self, enc: np.ndarray) -> float:
        """Score an already encoded counterfactual without mutating memory."""
        h = _trunk(enc, self.layers, self.arch["residual"])
        h = _context_mix(h, enc[features.VALID], self.context_global,
                         self.context_region)
        h = _strategy_mix(h, enc[features.VALID], self.strategy)
        if self.schema >= 2:
            pooled = self._pooled(h, enc[features.VALID])
            if self.head_w.shape != pooled.shape:
                raise ValueError(f"value head is {self.head_w.shape}, pooled state is "
                                 f"{pooled.shape}")
            logit = float(self.head_w @ pooled + self.head_b)
            if self.value_residual is not None:
                w1, b1, w2, b2 = self.value_residual
                logit += float(np.maximum(pooled @ w1 + b1, 0.0) @ w2 + b2)
            return float(np.tanh(logit))
        # Legacy value files were BCE logits over P(win).
        logit = float(self.head_w @ h.mean(axis=(1, 2)) + self.head_b)
        return float(2.0 / (1.0 + np.exp(-logit)) - 1.0)

    def win_prob(self, obs: Obs) -> float:
        return 0.5 * (self.value(obs) + 1.0)
