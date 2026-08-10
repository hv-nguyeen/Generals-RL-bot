"""Neural best-response oracle for the PSRO league: widen the oracle class.

WHY THIS EXISTS
---------------
`learn/league.py` searches CONFIG space. A config can only reweight behaviours
`bot/policy/controller.py` already implements, so the oracle cannot return a
strategy the controller cannot express -- and when the oracle class stops
finding best responses, PSRO's own prescription is to widen it, not to run more
generations. This module is the wider class: given the league's Nash mixture
sigma over the archive, it trains a policy network to beat that mixture and
writes weights that plug back in as a `clone:<path>.npz` archive member through
the dispatch `arena/agents.py` already has. Nothing in `bot/` changes.

FIVE PREVIOUS ATTEMPTS FAILED. EACH ONE HAS A MECHANISM HERE
------------------------------------------------------------
1. Behaviour cloning imitates median field play, so its ceiling is the median.
   Here the clone is only the INITIALISATION; the objective is win/loss against
   the archive, and the gate is PAIRED AGAINST THAT INITIALISATION on fresh
   boards -- a net that has not beaten its own starting point never enters the
   archive.
2. PPO self-play shaped by land lead: in a mirror the shaped reward is zero-sum
   and its expectation is exactly zero, so 195 iterations taught nothing. There
   is no shaping term of any kind in this file -- grep it for `land`, there is
   nothing to find -- and there is no mirror: every opponent is an archive
   member.
3. Frozen-opponent PPO learned to farm one bot. The opponent is resampled PER
   GAME from sigma' = 0.85*sigma + 0.15*uniform, the floor surviving even the
   point-mass sigma that `league.report` warns about, and the eval prints a
   per-opponent breakdown so farming is visible rather than hidden in a mean.
4. Terminal-reward REINFORCE with no baseline pushed down 90% of its own moves
   every step and collapsed to a 0% win rate in twelve iterations. There is a
   separate GAE critic (the `learn/valuetrain.py` topology, tanh head) fitted to
   the MONTE-CARLO return, CRITIC-ONLY warmup until the critic actually explains
   something (`--warm-evar`, not a fixed iteration count -- a 1500-iteration run
   released the policy after 3 iterations with evar still at +0.02 and eval fell
   0.175 -> 0.000 by iteration 160), and the advantage is SCALED but not
   re-centred. Centring
   was the bug, not the cure. With gamma=1 and terminal-only reward, lambda
   interpolates between critic-bootstrapped TD advantages and the Monte-Carlo
   endpoint `z - V(s_t)`. At lam=0.95 a weak near-zero critic leaves states more
   than ~60 plies from the end with raw advantage near 0, so subtracting the
   buffer mean hands most of the batch one identical deterministic advantage
   whose sign is set by the batch win rate -- failure 4 verbatim. Higher lambda
   reduces that bootstrap bias but raises variance; critic quality and lambda
   must be read together.
5. The self-imitation exploiter started from random weights over a 3529-action
   space and never learned to play. This starts from clone.npz and is held there
   by three independent bounds: the PPO clip, a k3 KL anchor to the init with a
   beta guardrail, and an in-epoch KL early stop.

Two operational failures also have mechanisms. The 22 GB OOM came from putting
the anchor network inside a full-batch graph; here the anchor and critic
forwards are chunked, no-grad, and outside `value_and_grad`, and updates are
minibatched. The fork deadlock came from forking an initialised multithreaded
JAX; the rollout pool is `spawn` and its workers import only numpy/bot/sim/arena
(`league.py` never imports jax at all -- it calls this module as a subprocess).

READING THE OUTPUT
------------------
    training mixture: 1.78 effective opponents over 6 archive members
    iter  35  games 256  W/D/L 105/12/139  samp  98k  turns 384  train-wr 0.41
              pg -0.0132  v 0.214  evar 0.31  dlp0 4.1e-04  klU 0.011/0.019
              klA 0.38  beta 0.050  ent 2.91  gn 0.42  mb 46  14s
    eval  40  0.512 +-0.035 (96 distinct)  [hunter:1 0.71  greedy 0.83  clone 0.55]

`train-wr` is NOT progress: training boards, sampled play, sigma'-weighted.
The `eval` line is the only progress number -- held-out boards, argmax play (the
way a `clone:` archive member actually plays), raw sigma.

`evar` is the critic's explained variance, 1 - Var(z - V)/Var(z), measured on
the buffer BEFORE this iteration's critic updates. It reports baseline quality,
but its effect on GAE depends on lambda: low lambda relies strongly on the
critic, while lambda=1 gives the Monte-Carlo advantage `z - V(s_t)`. `v` is MSE
against the direct outcome target and can read healthy when the critic is useless.

`dlp0` is max|logp_parent - logp_worker| over the first minibatch. It is a MAX,
not a mean: a mean over per-sample noise averages back to 1.0000 and is blind to
exactly the desync it is supposed to catch.

KILL THE RUN IF:

    dlp0 > 0.05         The numpy forward in the worker and the JAX forward in
                        the parent have diverged. Every gradient is then
                        computed against a policy that did not generate the
                        data. This is the bug class that once cost a 200-0
                        arena result; nothing downstream is salvageable. The
                        threshold is not 0: numpy im2col f32 and XLA f32 differ
                        by ~1e-5 per logit and the buffer is stored float16, so
                        a healthy run prints 1e-4..1e-3. A layout bug prints
                        O(1).
    evar <= 0 after iteration 10
                        At the default lam=0.95 the critic explains nothing, so
                        most early advantages are bootstrap error plus a heavily
                        attenuated terminal residual. At lam near 1 this is more
                        a variance/baseline warning than a bias verdict; do not
                        reuse the default evar gate without measuring that arm.
    W/D/L nearly all D  Every game is hitting --max-turns. train-wr reads 0.50,
                        which is indistinguishable from a healthy 50/50 split,
                        and every advantage is ~0. Eight hours of nothing.
    klA past 1.0 with beta stuck at its 1.0 cap
                        The anchor has lost. The warm start is being spent and
                        this is failure 5 rebuilding itself.
    klU last value pinned at 0.02 every iteration
                        Every update is being clamped by the early stop; the
                        learning rate is too high for this signal.
    turns falling fast while train-wr falls
                        It is learning to die faster -- failure 4's exact
                        signature. Check that the critic warmup ran.
    the eval bracket has ONE entry
                        sigma is a point mass, so this is single-opponent PPO
                        wearing a league's clothes -- failure 3. The startup
                        line says so first: read `effective opponents`, not the
                        member count. Raise --sigma-floor or fix the archive.
    one opponent >= 0.9 and the rest <= 0.3 in a MULTI-entry bracket
                        Farming, not a best response. Also failure 3.
    eval flat at the init's score for 50+ iterations
                        No best response in this class either. That is a real
                        result; stop and write it down.

DEFENSIVE COUNTEREXAMPLE ARM
----------------------------
The default remains unchanged. With ``--defense-aux-weight > 0`` the rollout
workers record only losses to the named ``--defense-opponent`` (normally the
randomized ``snipe:`` member). In the final ``--defense-tail-turns`` decisions,
when the sampled action drains our general while hidden enemy army exceeds the
configured garrison ratio, the worker records the current policy's best legal
non-draining alternative. PPO receives a small pairwise ranking loss for that
measured counterexample. This is not the shipped hidden-army veto: wins, draws,
non-target opponents, and ordinary attack states get no auxiliary label.
``defense-labels`` in the iteration line must be nonzero before this arm can be
judged; a zero count means the experiment is not collecting its intended data.
The arm is incompatible with ``--augment`` until action labels are transformed
through the same dihedral map.

A COLLAPSE is caught rather than watched: an eval 2 se below the `eval 0`
baseline rewinds to the best checkpoint, halves the learning rate and doubles
the anchor. Three of those and the run stops -- the schedule is wrong, not
unlucky. The previous run had no such guard and spent 310 iterations at a zero
win rate, which also spent the warm start it depended on.

THE COMMAND
-----------
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    /local/data/vng205/venv/bin/python -m learn.netoracle \\
        --league runs/league/league.json --init /local/data/vng205/clone.npz \\
        --out runs/league/nn/oracle-0.npz --workers 60 --iters 200

JAX_PLATFORMS must stay UNSET: the parent wants the L4. Normally the league
runs this for you (`--oracle net`).
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from arena import agents
from bot import features, rules
from bot.memory import TemporalMemory
from bot.policy.net import (DEFAULT_CHANNELS, DEFAULT_LAYERS, Net, arch_record,
                            strategy_keys, trunk_keys)
from learn.replay import GameBalancedReplay
from learn.league import (PFSP_WEIGHTINGS, allocate, dense, fictitious_play,
                          pfsp, stderr)
from sim import engine, mapgen

# Board origins. Training boards are GENERATED and there are 128 fresh ones per
# iteration, so the three ranges must not collide; main() asserts the training
# span never reaches NN_EVAL_SEED0.
NN_TRAIN_SEED0 = 400_000
NN_EVAL_SEED0 = 600_000
NN_GATE_SEED0 = 700_000

GAMMA = 1.0          # terminal-only reward; discounting a win 500 turns out is
LAM = 0.95           # historical TD/MC interpolation; 1.0 gives z - V(s_t).
REWARD_MODES = ("terminal", "tempo")
TEMPO_EPS = 0.03     # small terminal-only tie-break for an explicit exploiter arm
CLIP = 0.2
ENTROPY = 0.003
ADV_CLIP = 5.0
SIGMA_FLOOR = 0.15   # training only; eval and the gate use raw sigma
KL_STOP = 0.02       # the clip flattens the gradient, it does not bound the step
BETA0, BETA_MIN, BETA_MAX = 0.05, 0.01, 1.0
ANCHOR_HI, ANCHOR_LO = 1.0, 0.1
WARMUP = 3           # minimum critic-only iterations; --warm-evar is the real gate
REFREEZE_AFTER = 5   # consecutive stale-critic buffers before actor updates stop
MAX_REWINDS = 3      # collapses tolerated before the schedule is declared wrong
CHUNK = 8192         # ingest forward chunk, no-grad, outside value_and_grad

# Defensive action ranking is deliberately opt-in.  The ordinary terminal PPO
# objective must remain byte-for-byte compatible when its weight is zero; this
# arm exists to put a little supervised signal exactly where the snipe
# counterexample is observed instead of asking a sparse terminal reward to
# assign credit over hundreds of actions.
DEFENSE_AUX_WEIGHT = 0.0
DEFENSE_TAIL_TURNS = 60
DEFENSE_HIDDEN_RATIO = 2.5
DEFENSE_MARGIN = 0.05
DEFENSE_OPPONENT = "snipe"
DEFENSE_CF_TOPK = 8
DEFENSE_CF_HORIZON = 32
DEFENSE_CF_STRIDE = 8
DEFENSE_CF_MIN_TURN = 80
DEFENSE_CF_RISK_RATIO = 1.25


def critic_ready(it: int, evar: float, threshold: float) -> bool:
    """Whether this buffer permits an actor update.

    A non-positive threshold is the explicitly documented fixed-warmup mode.
    Otherwise NaN (including an all-draw buffer) is always unready.
    """
    return bool(it >= WARMUP and
                (threshold <= 0.0 or
                 (np.isfinite(evar) and evar >= threshold)))


def readiness_step(warmed: bool, low: int, ready: bool,
                   refreeze_after: int = REFREEZE_AFTER) -> tuple[bool, int, bool]:
    """Standing critic gate with hysteresis; returns (warm, low, refroze)."""
    low = 0 if ready else low + 1
    if not warmed:
        return ready, low, False
    if low >= refreeze_after:
        return False, 0, True
    return True, low, False


def restore_actor_critic(actor: dict, critic: dict, xp=np):
    """Restore a matched snapshot with fresh optimiser states and counters."""
    p = {k: xp.asarray(v) for k, v in actor.items()}
    q = {k: xp.asarray(v) for k, v in critic.items()}
    opt_p = {k: (xp.zeros_like(v), xp.zeros_like(v)) for k, v in p.items()}
    opt_v = {k: (xp.zeros_like(v), xp.zeros_like(v)) for k, v in q.items()}
    return p, q, opt_p, opt_v, 0, 0

def policy_keys(arch: dict) -> set:
    """Exactly the weights `bot/policy/net.Net` loads, for this architecture.

    Derived, not enumerated: the old hardcoded set silently projected a deeper
    checkpoint down onto four layers and trained the truncation.
    """
    keys = ({f"{n}_{s}" for n in trunk_keys(arch["layers"], arch["residual"])
             for s in "wb"} | {"head_w", "head_b", "pass_w", "pass_b"})
    if arch.get("context", False):
        keys |= {"context_global", "context_region"}
    if arch.get("strategy", False):
        keys |= strategy_keys()
    return keys


# --------------------------------------------------------------------------
# objective. `xp` is numpy or jax.numpy: the selfcheck exercises the SAME
# arithmetic the trainer differentiates, so the two cannot drift.
def _log_softmax(z, xp):
    z = z - xp.max(z, axis=-1, keepdims=True)
    return z - xp.log(xp.sum(xp.exp(z), axis=-1, keepdims=True))


def _k3(d, xp):
    """Schulman's k3 KL estimator from stored scalars. Always >= 0, zero at d=0.

    d is clipped because nothing else bounds it: the PPO clip flattens the
    surrogate's gradient, it does not bound logp, and a 3529-way masked softmax
    collapsing onto a few actions reaches d ~ -90 easily. exp overflows in f32 at
    88, and one inf here NaNs every parameter on the next adam step with no
    assertion to catch it. Past |d| = 20 the KL is lost anyway.
    """
    d = xp.clip(d, -20.0, 20.0)
    return xp.mean(xp.exp(d) - d - 1.0)


def policy_loss(logits, mask, idx, old_logp, ref_logp, adv, beta, xp=np,
                clip: float = CLIP, ent_coef: float = ENTROPY):
    """PPO surrogate + entropy + anchor KL. Returns (loss, diagnostics).

    Illegal actions are masked in the logits AND excluded from every term: after
    the mask their probability is exactly zero, so they contribute zero policy
    gradient and zero KL, and the `where` inside the entropy is load-bearing --
    exp(-1e9) underflows to 0 but 0 * -1e9 is -0.0, not 0.0, and the guard also
    keeps the gradient off masked entries.
    """
    n = idx.shape[0]
    lp = _log_softmax(xp.where(mask, logits, -1e9), xp)
    logp = lp[xp.arange(n), idx]

    ratio = xp.exp(logp - old_logp)
    pg = -xp.mean(xp.minimum(ratio * adv, xp.clip(ratio, 1 - clip, 1 + clip) * adv))
    probs = xp.where(mask, xp.exp(lp), 0.0)
    ent = xp.mean(-xp.sum(probs * xp.where(mask, lp, 0.0), axis=1))
    kl_anchor = _k3(ref_logp - logp, xp)
    kl_update = _k3(logp - old_logp, xp)      # KL(behaviour || theta), bounds the step
    loss = pg - ent_coef * ent + beta * kl_anchor
    # The desync statistic is a MAX, not mean(ratio): a per-sample forward
    # mismatch is zero-mean in logp, so its mean ratio is 1 + Var/2 and rounds to
    # 1.0000 however badly the two paths disagree. Only a gross error survives
    # the averaging, and gross errors are not the ones worth a diagnostic.
    dlp = xp.max(xp.abs(logp - old_logp))
    return loss, (dlp, kl_update, kl_anchor, ent, pg)


def defense_pairwise_loss(logits, mask, unsafe_idx, safe_idx, weights,
                          xp=np, margin: float = DEFENSE_MARGIN):
    """Rank a measured safe alternative above the action that lost the game.

    ``safe_idx`` is -1 for rows without a counterexample.  Invalid rows are
    excluded rather than manufacturing a PASS target, so a batch with no loss
    evidence contributes exactly zero.  This is a ranking *auxiliary* only:
    terminal W/D/L remains the objective and the caller controls its weight.
    ``xp`` is numpy or jax.numpy, matching :func:`policy_loss`.
    """
    n = unsafe_idx.shape[0]
    lp = _log_softmax(xp.where(mask, logits, -1e9), xp)
    safe = xp.maximum(safe_idx, 0)
    unsafe = xp.maximum(unsafe_idx, 0)
    safe_lp = lp[xp.arange(n), safe]
    unsafe_lp = lp[xp.arange(n), unsafe]
    valid = (safe_idx >= 0) & (unsafe_idx >= 0) & (safe_idx != unsafe_idx)
    # softplus(margin - (safe - unsafe)); stable for the small margin used here
    # and differentiable even when the current policy already ranks safely.
    delta = margin - (safe_lp - unsafe_lp)
    per = xp.logaddexp(0.0, delta)
    w = xp.where(valid, weights, 0.0)
    denom = xp.maximum(xp.sum(w), 1.0)
    return xp.sum(w * per) / denom


def gae(z: float, v: np.ndarray, gamma: float = GAMMA, lam: float = LAM):
    """Advantages for ONE complete episode with reward z at the end.

    Episodes are whole games, so V(s_T) is 0 by definition and there is no
    truncation boundary to bootstrap across -- `learn/rl.py`'s self-bootstrap at
    the window edge cannot happen here.

    Advantages only. The critic target is separately the MONTE-CARLO outcome z
    at every state and is therefore independent of lambda. With gamma=1,
    lambda interpolates between one-step critic bootstrapping and the exact
    Monte-Carlo advantage: ``lam=1 => A_t = z - V(s_t)``. Moving lambda upward
    reduces bias from an inaccurate critic but increases trajectory-level
    variance; it does not change the critic's reward horizon.

    Do not replace the critic target with ``adv + v``. At lam=0.95 and an
    initially zero critic that lambda-return is ``0.95**(T-1-t) * z`` for most
    of the game, so the critic would train on its own bootstrapped zero rather
    than the direct outcome label that is already available from a complete game.
    """
    T = len(v)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in range(T - 1, -1, -1):
        nxt = v[t + 1] if t + 1 < T else 0.0
        r = z if t == T - 1 else 0.0
        last = (r + gamma * nxt - v[t]) + gamma * lam * last
        adv[t] = last
    return adv


def episode_return(z: float, turns: int, max_turns: int,
                   mode: str = "terminal", tempo_eps: float = TEMPO_EPS) -> float:
    """Return used by the neural oracle's critic and GAE.

    ``terminal`` is the historical objective and is exactly ``z``. ``tempo``
    is an explicitly opt-in, terminal-only tie-break for a learned exploiter:
    wins stay closer to +1 when they finish early, while losses are less
    negative when the policy survives longer. The factor keeps the utility in
    [-1, 1], so the tanh critic remains bounded, and every win still outranks
    every draw, which outranks every loss. Evaluation and promotion always use
    the raw W/D/L outcome, never this shaped value.
    """
    if mode not in REWARD_MODES:
        raise ValueError(f"unknown reward mode {mode!r}; expected {REWARD_MODES}")
    if not 0.0 <= float(tempo_eps) < 1.0:
        raise ValueError(f"tempo_eps must be in [0, 1), got {tempo_eps}")
    if mode == "terminal":
        return float(z)
    if max_turns <= 0:
        raise ValueError(f"max_turns must be positive, got {max_turns}")
    # A faster win stays closer to +1; a slower loss stays closer to zero.
    # Clamp malformed turn counts at the rule boundary so a worker fault cannot
    # manufacture a target outside the critic's representable range.
    frac = min(max(float(turns), 0.0), float(max_turns)) / float(max_turns)
    return float(z) * (1.0 - float(tempo_eps) * frac)


def adam(p, opt, g, t, lr):
    """One Adam step. `lr` is an ARGUMENT, not a closure: the collapse guard
    halves it at runtime and a closed-over python float inside a jit would
    silently keep the original.

    Module scope so `learn/selfplay.py` imports it instead of making the fifth
    copy in this repo. jax is imported in the body, not at module level, because
    the spawned rollout workers import this module for `_rollout` and must never
    pull in jax.
    """
    import jax.numpy as jnp

    b1, b2, eps = 0.9, 0.999, 1e-8
    new_p, new_o = {}, {}
    for k in p:
        m, v = opt[k]
        m = b1 * m + (1 - b1) * g[k]
        v = b2 * v + (1 - b2) * g[k] ** 2
        new_p[k] = p[k] - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
        new_o[k] = (m, v)
    return new_p, new_o


def clip_grads(g, limit):
    """Global-norm gradient clip. Returns (clipped, pre-clip norm)."""
    import jax.numpy as jnp

    norm = jnp.sqrt(sum(jnp.sum(v ** 2) for v in g.values()))
    s = jnp.minimum(1.0, limit / (norm + 1e-8))
    return {k: v * s for k, v in g.items()}, norm


def mixture(sigma, floor: float = SIGMA_FLOOR) -> np.ndarray:
    """sigma' = (1-floor)*sigma + floor*uniform, the TRAINING opponent distribution.

    The floor is not a smoothing nicety: on a transitive matrix fictitious play
    returns a point mass, and a point-mass sigma is failure 3 (one frozen
    opponent) rebuilt inside the league. It is also not a cure -- at the default
    a point mass still sends 85% of games to one bot. `effective` prints what it
    actually bought so the operator can see failure 3 before a game is played.
    """
    p = (1.0 - floor) * np.asarray(sigma, dtype=float) + floor / len(sigma)
    return p / p.sum()


def effective(p) -> float:
    """exp(entropy(p)): how many opponents this distribution is really made of."""
    p = np.asarray(p, dtype=float)
    nz = p[p > 0]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def pfsp_for_checkpoint(state: dict, checkpoint: str,
                        weighting: str) -> tuple[np.ndarray, int]:
    """PFSP distribution for the archive member represented by checkpoint."""
    target = Path(checkpoint).resolve()
    found = []
    for i, member in enumerate(state["archive"]):
        kind, _, arg = member["spec"].partition(":")
        arg = arg.partition("@")[0]
        if kind in ("clone", "ship", "tta", "guard", "snipe") and arg:
            try:
                if Path(arg).resolve() == target:
                    found.append(i)
            except OSError:
                pass
    if len(found) != 1:
        raise SystemExit(
            f"--sampling pfsp needs --init to identify exactly one archive member; "
            f"found indices {found} for {target}. Add clone:{target} once to the archive.")
    i = found[0]
    return pfsp(dense(state["payoff"])[i], weighting, exclude=i), i


def sample_opponents(sigma, n: int, rng, floor: float = SIGMA_FLOOR) -> np.ndarray:
    """Draw n opponent indices from the training mixture."""
    p = mixture(sigma, floor)
    return rng.choice(len(p), size=n, p=p)


def build_jobs(it: int, games: int, sigma, rng, floor: float = SIGMA_FLOOR) -> list:
    """Training jobs for one iteration: every board played from BOTH seats.

    Same board, same opponent, both colours -- the seat-bias cancellation
    `arena/runner._jobs` does, inside the training batch. Seat bias is worth a
    whole result, and `exploit.py` locked the learner to seat 0.
    """
    boards = games // 2
    base = NN_TRAIN_SEED0 + it * boards
    opps = sample_opponents(sigma, boards, rng, floor)
    return [(base + b, int(opps[b]), seat, False)
            for seat in (0, 1) for b in range(boards)]


def match_jobs(sigma, games: int, seed0: int) -> list:
    """Argmax jobs against the raw mixture, allocated by league.allocate."""
    jobs, b = [], 0
    for i, g in enumerate(allocate(sigma, games)):
        for _ in range(g // 2):
            jobs += [(seed0 + b, i, 0, True), (seed0 + b, i, 1, True)]
            b += 1
    return jobs


def distinct_games(jobs: list, maps: str | None) -> int:
    """How many of these measurement jobs are actually DIFFERENT games.

    Every agent here is deterministic in argmax play and `pool_grid` is
    `pool[seed % L]`, so on a map pool a job repeats an earlier one the moment
    the seed wraps -- a replayed game carries exactly zero information. Quoting
    `stderr(len(jobs))` then understates the interval by sqrt(len/distinct): on
    the documented 72-board pool the ORACLE slice is 24 boards, so a 400-game
    gate is 48 real games and its margin was 3x too permissive. The gate decides
    archive membership, so this is the number it must be built from.
    """
    if not maps:
        return len(jobs)
    L = len(mapgen.load_pool(maps))
    return len({(j[0] % L, j[1], j[2]) for j in jobs})


def tally(results: list[dict], specs: list[str]) -> tuple[float, dict]:
    """(pooled score, per-opponent score). Draws count a half, as league does."""
    def rate(rs):
        return sum(1.0 if r["z"] > 0 else (0.5 if r["z"] == 0 else 0.0)
                   for r in rs) / max(len(rs), 1)
    per = {}
    for i, s in enumerate(specs):
        rs = [r for r in results if r["opp"] == i]
        if rs:
            per[s] = round(rate(rs), 3)
    return rate(results), per


# --------------------------------------------------------------------------
# worker side: numpy only, no jax, ever.
_CTX: dict = {}


def _my_general(obs):
    hit = np.argwhere((obs.type_grid == rules.T_GENERAL)
                      & (obs.owner_grid == rules.OWNER_ME))
    return (int(hit[0][0]), int(hit[0][1])) if len(hit) else None


def _hidden_general_drain(obs, idx: int, hidden_ratio: float) -> bool:
    """Whether ``idx`` empties our general while hidden army is threatening it.

    This is intentionally a *training label*, not the shipped GuardedPolicy
    veto.  The latter's unconditional hidden-army trigger was measured at
    strongly negative Elo because it fires in most wins too.  Restricting this
    predicate to the terminal tail of games the policy actually lost keeps the
    signal counterexample-conditioned and avoids turning a noisy correlate into
    a hard rule.
    """
    if hidden_ratio <= 0.0:
        return False
    gen = _my_general(obs)
    if gen is None:
        return False
    act = features.index_to_action(int(idx))
    if act is None or act[0] != rules.MOVE or (int(act[1]), int(act[2])) != gen:
        return False
    gr, gc = gen
    garrison = max(int(obs.army_grid[gr, gc]), 1)
    visible_opp = int(obs.army_grid[obs.owner_grid == rules.OWNER_OPP].sum())
    hidden = max(int(obs.opp_army) - visible_opp, 0)
    return hidden >= hidden_ratio * garrison


def _defense_safe_index(obs, mask, logits, idx: int, hidden_ratio: float) -> int:
    """Return the current policy's best legal non-draining alternative, or -1."""
    if not _hidden_general_drain(obs, idx, hidden_ratio):
        return -1
    for alt in np.argsort(-np.asarray(logits)):
        alt = int(alt)
        if mask[alt] and not _hidden_general_drain(obs, alt, hidden_ratio):
            return alt
    return -1


def _defense_risk(obs, ratio: float) -> bool:
    """Whether the current observation is an actionable hidden-army risk."""
    if ratio <= 0.0:
        return False
    gen = _my_general(obs)
    if gen is None:
        return False
    gr, gc = gen
    garrison = max(int(obs.army_grid[gr, gc]), 1)
    visible_opp = int(obs.army_grid[obs.owner_grid == rules.OWNER_OPP].sum())
    hidden = max(int(obs.opp_army) - visible_opp, 0)
    return hidden >= ratio * garrison


def _branch_score(st, seat: int) -> float:
    """Score a short engine branch from the learner's perspective.

    A resolved win/loss dominates the shaped continuation score.  Unresolved
    branches are ranked by the two quantities the counterexample is about:
    general safety versus hidden army, then army/land advantage.  This score is
    used only to choose a label; PPO/evaluation still use terminal W/D/L.
    """
    if st.winner >= 0:
        return 2.0 if st.winner == seat else -2.0
    obs = engine.observe(st, seat)
    gr, gc = st.gpos[seat]
    garrison = int(st.armies[gr, gc])
    visible_opp = int(obs.army_grid[obs.owner_grid == rules.OWNER_OPP].sum())
    hidden = max(int(obs.opp_army) - visible_opp, 0)
    safety = np.tanh((garrison - hidden) / 40.0)
    army_adv = np.tanh((int(obs.my_army) - int(obs.opp_army)) / 80.0)
    land_adv = np.tanh((int(obs.my_land) - int(obs.opp_land)) / 10.0)
    return float(0.60 * safety + 0.25 * army_adv + 0.15 * land_adv)


def _counterfactual_score(st, memory, foe, net, seat: int, first_action,
                          foe_action, horizon: int) -> float:
    """Play one candidate action through a short, isolated engine branch."""
    branch = st.copy()
    branch_memory = memory.copy()
    branch_foe = copy.deepcopy(foe)
    acts = [None, None]
    acts[seat] = first_action
    acts[1 - seat] = foe_action
    if engine.step(branch, acts[0], acts[1]):
        return _branch_score(branch, seat)
    for _ in range(max(int(horizon) - 1, 0)):
        obs = engine.observe(branch, seat)
        branch_memory.update(obs)
        mask = features.legal_mask(obs)
        logits = np.where(mask, net.logits(obs, memory=branch_memory), -np.inf)
        idx = int(np.argmax(logits))
        try:
            other = branch_foe.act(engine.observe(branch, 1 - seat))
        except Exception:                         # noqa: BLE001
            other = rules.PASS_ACTION
        acts = [None, None]
        acts[seat] = features.index_to_action(idx)
        acts[1 - seat] = other
        if engine.step(branch, acts[0], acts[1]):
            break
    return _branch_score(branch, seat)


def _counterfactual_label(st, obs, memory, foe, net, seat: int, mask,
                          logits, idx: int, foe_action, topk: int,
                          horizon: int, risk_ratio: float) -> tuple[int, float]:
    """Return (better action, positive branch margin), or (-1, 0)."""
    if not _defense_risk(obs, risk_ratio):
        return -1, 0.0
    legal = np.flatnonzero(mask)
    order = legal[np.argsort(np.asarray(logits)[legal])[-max(1, int(topk)):]]
    candidates = {int(x) for x in order}
    candidates.add(int(idx))                 # sampled action is the baseline
    scored = []
    for candidate in candidates:
        action = features.index_to_action(candidate)
        try:
            score = _counterfactual_score(
                st, memory, foe, net, seat, action, foe_action, horizon)
        except Exception:                     # noqa: BLE001
            continue                           # one bad branch cannot poison a batch
        scored.append((score, candidate))
    if not scored:
        return -1, 0.0
    current = next((s for s, c in scored if c == int(idx)), None)
    if current is None:
        return -1, 0.0
    best, candidate = max(scored)
    margin = float(best - current)
    if candidate == int(idx) or margin < 0.10:
        return -1, 0.0
    return int(candidate), min(max(margin, 0.0), 1.0)


def _init(live: str, specs: tuple, maps: str | None, max_turns: int,
          defense_tail_turns: int = 0, defense_hidden_ratio: float = 0.0,
          defense_opponent: str = DEFENSE_OPPONENT,
          defense_counterfactual: bool = False,
          defense_cf_topk: int = DEFENSE_CF_TOPK,
          defense_cf_horizon: int = DEFENSE_CF_HORIZON,
          defense_cf_stride: int = DEFENSE_CF_STRIDE,
          defense_cf_min_turn: int = DEFENSE_CF_MIN_TURN,
          defense_cf_risk_ratio: float = DEFENSE_CF_RISK_RATIO) -> None:
    _CTX.update(live=live, specs=specs, maps=maps, max_turns=max_turns,
                defense_tail_turns=defense_tail_turns,
                defense_hidden_ratio=defense_hidden_ratio,
                defense_opponent=defense_opponent,
                defense_counterfactual=defense_counterfactual,
                defense_cf_topk=defense_cf_topk,
                defense_cf_horizon=defense_cf_horizon,
                defense_cf_stride=defense_cf_stride,
                defense_cf_min_turn=defense_cf_min_turn,
                defense_cf_risk_ratio=defense_cf_risk_ratio)


def _rollout(job):
    """One complete game. job = (seed, opp_idx, seat, greedy)."""
    seed, opp, seat, greedy = job
    # Only the argmax (measurement) jobs may use the board pool. A pool slice is
    # ~24 boards; 200 iterations of training on 24 boards is memorisation, not a
    # hold-out question, so training boards are always generated.
    maps = _CTX["maps"]
    grid = mapgen.pool_grid(maps, seed) if (greedy and maps) else mapgen.generate(seed)
    st = engine.from_grid(grid)
    net = Net(_CTX["live"])
    memory = TemporalMemory(*grid.shape)
    foe = agents.make(_CTX["specs"][opp], 1 - seat, *grid.shape, seed)
    rng = np.random.default_rng(seed * 2 + seat)

    xs, ids, masks, lps = [], [], [], []
    defense_safe, defense_weight = [], []
    turns, faults = 0, 0
    for turns in range(1, _CTX["max_turns"] + 1):
        obs = engine.observe(st, seat)
        memory.update(obs)
        mask = features.legal_mask(obs)          # PASS is always legal: never empty
        lg = np.where(mask, net.logits(obs, memory=memory), -np.inf)
        lg -= lg.max()
        p = np.exp(lg)
        p /= p.sum()
        idx = int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p))
        if not greedy:
            xs.append(features.encode(obs, memory).astype(np.float16))
            ids.append(idx)
            masks.append(mask)
            lps.append(np.log(p[idx]))
            # Only collect this auxiliary's evidence against the named
            # counterexample opponent.  Applying it to ordinary losses would
            # teach the policy to preserve its general even when an attack is
            # the correct winning action.
            target = _CTX.get("defense_opponent", DEFENSE_OPPONENT)
            is_target = _CTX["specs"][opp].partition(":")[0] == target
            safe_idx = (_defense_safe_index(
                obs, mask, lg, idx,
                _CTX.get("defense_hidden_ratio", 0.0))
                        if is_target and not _CTX.get("defense_counterfactual", False)
                        else -1)
            defense_safe.append(safe_idx)
            defense_weight.append(1.0 if safe_idx >= 0 else 0.0)
        # engine.step is positional by seat. Placing these the wrong way round
        # trains the policy on its opponent's rewards and looks like slow noise.
        acts = [None, None]
        acts[seat] = features.index_to_action(idx)
        try:
            acts[1 - seat] = foe.act(engine.observe(st, 1 - seat))
        except Exception:                        # noqa: BLE001
            # No TIME budget here, unlike arena/runner.play: a timing-dependent
            # reward would make the training signal a function of node load. The
            # FAULT budget is the arena's, because the alternative is worse than
            # a poisoned batch -- an agent that raises every turn (an npz read
            # mid-write, a Controller that dies on one board shape) passes for
            # 1200 turns, banks z=+1 and 3x a normal episode's samples of play
            # against a do-nothing bot, and prints 1.00 in the eval bracket where
            # it reads as farming rather than as a crash.
            faults += 1
            if faults >= rules.MAX_FAULTS:
                return None
            acts[1 - seat] = rules.PASS_ACTION
        if (not greedy and _CTX.get("defense_counterfactual", False)
                and is_target
                and turns >= _CTX.get("defense_cf_min_turn", DEFENSE_CF_MIN_TURN)
                and turns % max(_CTX.get("defense_cf_stride", DEFENSE_CF_STRIDE), 1) == 0):
            safe_idx, margin = _counterfactual_label(
                st, obs, memory, foe, net, seat, mask, lg, idx,
                acts[1 - seat], _CTX.get("defense_cf_topk", DEFENSE_CF_TOPK),
                _CTX.get("defense_cf_horizon", DEFENSE_CF_HORIZON),
                _CTX.get("defense_cf_risk_ratio", DEFENSE_CF_RISK_RATIO))
            defense_safe[-1] = safe_idx
            defense_weight[-1] = margin
        if engine.step(st, acts[0], acts[1]):
            break

    # Turn limit and mutual capture (engine.step sets winner -1) are both draws.
    z = 0.0 if st.winner < 0 else (1.0 if st.winner == seat else -1.0)
    assert z in (-1.0, 0.0, 1.0)
    out = {"z": z, "turns": turns, "opp": opp}
    if greedy:
        return out
    if not ids:
        return None
    safe = np.full(len(ids), -1, dtype=np.int32)
    weights = np.zeros(len(ids), dtype=np.float32)
    if z < 0.0:
        if _CTX.get("defense_counterfactual", False):
            safe[:] = np.asarray(defense_safe, dtype=np.int32)
            weights[:] = np.asarray(defense_weight, dtype=np.float32)
        elif _CTX.get("defense_tail_turns", 0) > 0:
            tail = int(_CTX["defense_tail_turns"])
            start = max(0, len(ids) - tail)
            safe[start:] = np.asarray(defense_safe[start:], dtype=np.int32)
            weights[start:] = np.asarray(defense_weight[start:], dtype=np.float32)
    out.update(x=np.stack(xs), idx=np.asarray(ids, dtype=np.int32),
               mask=np.packbits(np.stack(masks), axis=1),
               logp=np.asarray(lps, dtype=np.float32),
               defense_safe=safe,
               defense_w=weights)
    return out


def publish(p: dict, live: Path) -> None:
    """Atomic: 60 workers np.load this path while we write it.

    `exploit.py` savez's straight onto the live file; a torn read is one bad
    iteration you would never diagnose.
    """
    tmp = live.with_suffix(".tmp.npz")
    arrays = {k: np.asarray(v) for k, v in p.items()}
    np.savez(tmp, **arrays, **arch_record(arrays))
    os.replace(tmp, live)


# --------------------------------------------------------------------------
def selfcheck() -> None:
    import tempfile

    from bot.obs import Obs

    # --- GAE, hand computed. z=1, V=[0, .5, .25], gamma=1, lam=.5:
    #     d = [.5, -.25, .75];  A2=.75, A1=-.25+.5*.75=.125, A0=.5+.5*.125=.5625
    a = gae(1.0, np.array([0.0, 0.5, 0.25], dtype=np.float32), 1.0, 0.5)
    assert np.allclose(a, [0.5625, 0.125, 0.75]), a

    # brute force sum_k (gamma*lam)^k delta_{t+k} on a 6-step episode
    v = np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.4], dtype=np.float32)
    g, lam, z = 0.99, 0.95, -1.0
    a = gae(z, v, g, lam)
    d = [(z if t == 5 else 0.0) + (g * v[t + 1] if t < 5 else 0.0) - v[t] for t in range(6)]
    for t in range(6):
        assert abs(a[t] - sum((g * lam) ** k * d[t + k] for k in range(6 - t))) < 1e-5

    # gamma=lam=1 with V==0: every advantage is the outcome
    assert np.all(gae(-1.0, np.zeros(7, dtype=np.float32), 1.0, 1.0) == -1.0)
    # a perfect critic (V == the true return) leaves nothing to explain
    a = gae(1.0, np.ones(9, dtype=np.float32), 1.0, 1.0)
    assert np.abs(a).max() < 1e-6, a
    # At the undiscounted Monte-Carlo endpoint all intermediate bootstrap terms
    # telescope exactly, for an arbitrary imperfect critic.
    v = np.array([0.2, -0.4, 0.7, 0.1], dtype=np.float32)
    assert np.allclose(gae(-1.0, v, 1.0, 1.0), -1.0 - v)
    # the lam=0.95 default is why the critic target is NOT adv+v: 100 plies from
    # the end a zero critic would be asked to predict 0.006 for a won game.
    a = gae(1.0, np.zeros(101, dtype=np.float32))
    assert abs(a[0] - 0.95 ** 100) < 1e-6 and a[0] < 0.01, a[0]
    # Reward shaping is explicitly opt-in and never leaves the critic's range.
    assert episode_return(1, 1, 100) == 1.0
    assert episode_return(1, 1, 100, "tempo") > episode_return(1, 100, 100, "tempo")
    assert episode_return(-1, 1, 100, "tempo") < episode_return(-1, 100, 100, "tempo")
    assert episode_return(0, 1, 100, "tempo") == 0.0

    # --- critic readiness is a standing gate, including NaN/all-draw buffers
    assert not critic_ready(WARMUP - 1, 1.0, 0.10)
    assert critic_ready(WARMUP, 0.10, 0.10)
    assert not critic_ready(WARMUP, float("nan"), 0.10)
    # Explicit compatibility mode: --warm-evar 0 means fixed warmup.
    assert critic_ready(WARMUP, float("nan"), 0.0)
    warm, low = True, 0
    for _ in range(REFREEZE_AFTER - 1):
        warm, low, refroze = readiness_step(warm, low, False)
        assert warm and not refroze
    warm, low, refroze = readiness_step(warm, low, False)
    assert not warm and low == 0 and refroze
    warm, low, refroze = readiness_step(warm, low, True)
    assert warm and low == 0 and not refroze
    p0, q0 = {"p": np.array([1.0])}, {"q": np.array([2.0])}
    p, q, op, ov, tp, tv = restore_actor_critic(p0, q0)
    assert np.array_equal(p["p"], p0["p"])
    assert np.array_equal(q["q"], q0["q"])
    assert all(not np.any(x) for pair in (*op.values(), *ov.values()) for x in pair)
    assert tp == tv == 0

    # --- sigma floor survives a point mass, but does not make it a mixture
    c = np.bincount(sample_opponents(np.array([1.0, 0.0, 0.0]), 20_000,
                                     np.random.default_rng(0)), minlength=3) / 20_000
    assert np.abs(c - [0.90, 0.05, 0.05]).max() < 0.02, c
    assert effective(mixture(np.array([1.0, 0.0, 0.0]))) < 1.5      # failure 3
    assert abs(effective(np.full(4, 0.25)) - 4.0) < 1e-6

    # --- a map pool makes replayed jobs identical games, and the interval knows
    ev = match_jobs(np.array([1.0, 0.0]), 400, NN_GATE_SEED0)
    assert distinct_games(ev, None) == 400
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        fh.write(json.dumps({"grids": [[[i]] for i in range(24)]}))
        pool = fh.name
    assert distinct_games(ev, pool) == 48, distinct_games(ev, pool)   # 24 boards, 2 seats
    Path(pool).unlink()

    # --- job construction
    jobs = build_jobs(3, 256, np.array([0.5, 0.5]), np.random.default_rng(1))
    assert len(jobs) == 256
    per_seat = [sorted(j[0] for j in jobs if j[2] == seat) for seat in (0, 1)]
    assert len(per_seat[0]) == 128 and len(set(per_seat[0])) == 128
    assert per_seat[0] == per_seat[1]          # same boards from both colours
    assert min(j[0] for j in jobs) >= NN_TRAIN_SEED0
    assert max(j[0] for j in jobs) < NN_EVAL_SEED0 < NN_GATE_SEED0
    assert all(j[3] is False for j in jobs)
    ev = match_jobs(np.array([0.5, 0.5]), 8, NN_EVAL_SEED0)
    assert len(ev) == 8 and all(j[3] is True for j in ev)
    assert sorted(j[1] for j in ev) == [0, 0, 0, 0, 1, 1, 1, 1]

    # --- mask round trip through the wire format
    rng = np.random.default_rng(2)
    m = rng.random(features.N_ACTIONS) < 0.1
    packed = np.packbits(np.stack([m, ~m]), axis=1)
    assert packed.shape == (2, -(-features.N_ACTIONS // 8)), packed.shape
    back = np.unpackbits(packed, axis=1)[:, :features.N_ACTIONS].astype(bool)
    assert np.array_equal(back[0], m) and np.array_equal(back[1], ~m)

    # --- illegal actions get exactly zero probability, and the loss is exactly
    #     the PPO term at theta == theta_ref (ratio 1, anchor contributes nothing)
    n, k = 8, 64
    logits = rng.normal(size=(n, k))
    mask = rng.random((n, k)) < 0.3
    mask[:, 0] = True                     # PASS is always legal
    lp = _log_softmax(np.where(mask, logits, -1e9), np)
    probs = np.exp(lp)
    assert np.all(probs[~mask] == 0.0), probs[~mask].max()
    assert np.allclose(probs.sum(axis=1), 1.0)

    idx = np.array([int(np.argmax(np.where(mask[i], logits[i], -np.inf))) for i in range(n)])
    old = lp[np.arange(n), idx]
    adv = rng.normal(size=n)
    loss, (dlp, kl_u, kl_a, ent, pg) = policy_loss(
        logits, mask, idx, old, old.copy(), adv, beta=0.5)
    assert dlp == 0.0, dlp                       # exactly, not approximately
    assert kl_u == 0.0 and kl_a == 0.0
    # WHY the desync statistic is a max and not mean(ratio): per-sample forward
    # noise is zero-mean in logp, so E[exp(d)] = 1 + Var(d)/2 and the mean still
    # rounds to 1.000 while every individual sample is wrong.
    noise = rng.normal(scale=0.03, size=n)
    noise -= noise.mean()
    _, (dlp_n, _, _, _, _) = policy_loss(logits, mask, idx, old + noise, old, adv, 0.5)
    assert abs(dlp_n - np.abs(noise).max()) < 1e-9, dlp_n
    assert abs(float(np.mean(np.exp(-noise))) - 1.0) < 0.01      # the mean is blind
    assert loss == pg - ENTROPY * ent            # the anchor adds literally nothing
    # d(k3)/d(logp) = -(exp(d) - 1) with d = ref - logp, which is exactly 0 at
    # d = 0: the anchor exerts no pull on the first update, only on drift.
    assert float(-(np.exp(0.0) - 1.0)) == 0.0

    # --- defensive ranking is a no-op without labels and prefers the target
    # when one is present. The invalid row exercises the fail-closed path used
    # by ordinary wins, draws, and non-snipe opponents.
    dm = np.ones((2, 5), dtype=bool)
    du = np.array([0, 2], dtype=np.int32)
    ds = np.array([1, -1], dtype=np.int32)
    dw = np.array([1.0, 0.0], dtype=np.float32)
    d0 = defense_pairwise_loss(np.zeros((2, 5), np.float32), dm, du, ds, dw)
    assert d0 > 0.0
    assert defense_pairwise_loss(
        np.zeros((2, 5), np.float32), dm, du,
        np.full(2, -1, np.int32), np.zeros(2, np.float32)) == 0.0
    boosted = np.zeros((2, 5), np.float32)
    boosted[0, 1] = 2.0
    assert defense_pairwise_loss(boosted, dm, du, ds, dw) < d0

    # --- a checkpoint round-trips into the loader the submission uses. Not the
    #     4x32 default: a residual trunk is what would break the key plumbing.
    arch = {"layers": 5, "channels": DEFAULT_CHANNELS, "residual": True,
            "context": False}
    names = trunk_keys(arch["layers"], arch["residual"])
    ch, p, prev = arch["channels"], {}, features.C
    for n in names:
        p[f"{n}_w"] = rng.normal(size=(ch, prev, 3, 3)).astype(np.float32) * 0.1
        p[f"{n}_b"] = np.zeros(ch, np.float32)
        prev = ch
    p["head_w"] = rng.normal(size=(features.PER_CELL, ch, 3, 3)).astype(np.float32) * 0.1
    p["head_b"] = np.zeros(features.PER_CELL, np.float32)
    p["pass_w"] = np.zeros(ch, np.float32)
    p["pass_b"] = np.float32(0.0)
    assert set(p) == policy_keys(arch), set(p) ^ policy_keys(arch)
    assert len(policy_keys({"layers": DEFAULT_LAYERS, "channels": ch,
                            "residual": False, "context": False})) == 2 * DEFAULT_LAYERS + 4
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ck.npz"
        publish(p, path)
        assert set(np.load(path).files) == policy_keys(arch) | set(arch_record(p))
        net = Net(str(path))                     # the exact loader ClonePolicy uses
        assert net.arch == arch, net.arch
    h = w = 18
    ty = np.full((h, w), rules.T_PLAIN, dtype=np.int8)
    ty[0, 0] = rules.T_GENERAL
    ow = np.zeros((h, w), dtype=np.int8)
    ow[0, 0] = rules.OWNER_ME
    ar = np.zeros((h, w), dtype=np.int32)
    ar[0, 0] = 9
    obs = Obs(H=h, W=w, turn=1, my_land=1, my_army=9, opp_land=1, opp_army=1,
              type_grid=ty, owner_grid=ow, army_grid=ar)
    out = net.logits(obs)
    assert out.shape == (features.N_ACTIONS,), out.shape
    lm = features.legal_mask(obs)
    # from (0,0) with 9 army: down and right, each whole or split, plus pass
    assert lm[features.PASS_INDEX] and lm.sum() == 5, lm.sum()
    q = np.exp(_log_softmax(np.where(lm, out, -1e9), np))
    assert np.all(q[~lm] == 0.0) and abs(q.sum() - 1.0) < 1e-6

    print("netoracle selfcheck OK")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", default=None, help="league.json; the ONLY source of sigma")
    ap.add_argument("--init", default="/local/data/vng205/clone.npz",
                    help="behaviour clone to start from and to anchor to")
    ap.add_argument("--init-critic", default=None,
                    help="gated standalone value model or PPO resume critic; "
                         "must match --init architecture")
    ap.add_argument("--out", default="/local/data/vng205/oracle-nn.npz")
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--games", type=int, default=256, help="per iteration, both seats")
    ap.add_argument("--epochs", type=int, default=1,
                    help="PPO/value passes over each rollout buffer. Historical "
                         "transfer runs used one pass; two amplified critic "
                         "overfit and policy drift")
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--critic-lr", type=float, default=1e-4,
                    help="critic Adam step. 1e-3 memorised the on-policy buffer "
                         "in measured full-distance runs; 1e-4 is the safe "
                         "transfer default")
    ap.add_argument("--critic-replay-games", type=int, default=0,
                    help="historical episodes retained for game-balanced critic replay")
    ap.add_argument("--critic-replay-frac", type=float, default=0.5,
                    help="share of critic minibatches drawn from replay")
    ap.add_argument("--critic-replay-source-floor", type=float, default=0.25,
                    help="uniform opponent-style mass inside critic replay")
    ap.add_argument("--value-hidden", type=int, default=64,
                    help="residual critic-head width; 0 is the linear control")
    ap.add_argument("--lam", type=float, default=LAM,
                    help="GAE TD/Monte-Carlo interpolation. 0.95 is historical; "
                         "1.0 gives z-V(s) and relies least on critic bootstraps. "
                         "The value is recorded in the run manifest")
    ap.add_argument("--reward-mode", choices=REWARD_MODES, default="terminal",
                    help="critic/GAE objective: terminal is the historical "
                         "win/draw/loss reward; tempo is an explicit "
                         "terminal-only fast-win/slow-loss tie-break for a "
                         "learned exploiter. Evaluation remains raw W/D/L")
    ap.add_argument("--tempo-eps", type=float, default=TEMPO_EPS,
                    help="tempo tie-break strength in [0,1); ignored by the "
                         "terminal reward mode")
    ap.add_argument("--augment", action="store_true",
                    help="random dihedral PPO minibatches with relabelled "
                         "masks/actions and recomputed frozen-policy log-probs")
    ap.add_argument("--defense-aux-weight", type=float,
                    default=DEFENSE_AUX_WEIGHT,
                    help="opt-in loss-conditioned defensive action ranking "
                         "weight; 0 keeps historical PPO unchanged")
    ap.add_argument("--defense-tail-turns", type=int,
                    default=DEFENSE_TAIL_TURNS,
                    help="late loss tail eligible for defensive labels")
    ap.add_argument("--defense-hidden-ratio", type=float,
                    default=DEFENSE_HIDDEN_RATIO,
                    help="hidden enemy army / garrison threshold for labels")
    ap.add_argument("--defense-margin", type=float, default=DEFENSE_MARGIN,
                    help="safe-over-unsafe log-probability ranking margin")
    ap.add_argument("--defense-opponent", default=DEFENSE_OPPONENT,
                    help="archive agent kind supplying defensive counterexamples")
    ap.add_argument("--defense-counterfactual", action="store_true",
                    help="rank alternatives with short real-engine branches "
                         "instead of the old second-choice label")
    ap.add_argument("--defense-cf-topk", type=int, default=DEFENSE_CF_TOPK)
    ap.add_argument("--defense-cf-horizon", type=int, default=DEFENSE_CF_HORIZON)
    ap.add_argument("--defense-cf-stride", type=int, default=DEFENSE_CF_STRIDE)
    ap.add_argument("--defense-cf-min-turn", type=int, default=DEFENSE_CF_MIN_TURN)
    ap.add_argument("--defense-cf-risk-ratio", type=float,
                    default=DEFENSE_CF_RISK_RATIO)
    ap.add_argument("--policy-scope", choices=("all", "head"), default="all",
                    help="train all policy weights or only the action/pass head")
    ap.add_argument("--warm-evar", type=float, default=0.10,
                    help="hold the policy frozen until the critic explains this "
                         "much of the return; 0 restores the old fixed warmup")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--gate-games", type=int, default=400)
    ap.add_argument("--max-turns", type=int, default=rules.TURN_LIMIT)
    ap.add_argument("--maps", default=None, help="ORACLE slice of the board pool; "
                                                 "measurement games only")
    ap.add_argument("--sigma-floor", type=float, default=SIGMA_FLOOR,
                    help="uniform mass mixed into sigma for TRAINING opponents. "
                         "The default leaves a point-mass sigma sending 85%% of "
                         "games to one bot; raise it when the startup line says "
                         "fewer than two effective opponents")
    ap.add_argument("--role", choices=("main", "main-exploiter", "league-exploiter"),
                    default="league-exploiter",
                    help="population role recorded in the artifact and logs")
    ap.add_argument("--sampling", choices=("nash", "pfsp"), default="nash",
                    help="training opponent distribution; gates remain on raw Nash")
    ap.add_argument("--pfsp-weighting", choices=PFSP_WEIGHTINGS, default="variance",
                    help="variance targets near-50%% games; linear/squared focus weaknesses")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--residual", action=argparse.BooleanOptionalAction, default=None,
                    help="only to assert what --init already is; --no-residual "
                         "asserts a plain trunk, which --layers cannot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    if not 0.0 < args.lam <= 1.0:
        raise SystemExit(f"--lam must be in (0, 1], got {args.lam}")
    if args.critic_replay_games < 0:
        raise SystemExit("--critic-replay-games must be non-negative")
    if not 0.0 <= args.critic_replay_frac < 1.0:
        raise SystemExit("--critic-replay-frac must be in [0, 1)")
    if not 0.0 <= args.critic_replay_source_floor <= 1.0:
        raise SystemExit("--critic-replay-source-floor must be in [0, 1]")
    if args.value_hidden < 0:
        raise SystemExit("--value-hidden must be zero or positive")
    if not 0.0 <= args.tempo_eps < 1.0:
        raise SystemExit(f"--tempo-eps must be in [0, 1), got {args.tempo_eps}")
    if args.defense_aux_weight < 0.0:
        raise SystemExit("--defense-aux-weight must be non-negative")
    if args.defense_tail_turns < 0:
        raise SystemExit("--defense-tail-turns must be non-negative")
    if args.defense_hidden_ratio < 0.0:
        raise SystemExit("--defense-hidden-ratio must be non-negative")
    if args.defense_margin < 0.0:
        raise SystemExit("--defense-margin must be non-negative")
    if args.defense_counterfactual and args.defense_aux_weight <= 0.0:
        raise SystemExit("--defense-counterfactual needs --defense-aux-weight > 0")
    if args.defense_cf_topk < 1 or args.defense_cf_horizon < 1:
        raise SystemExit("counterfactual topk/horizon must be positive")
    if args.defense_cf_stride < 1 or args.defense_cf_min_turn < 0:
        raise SystemExit("counterfactual stride must be positive and min-turn non-negative")
    if args.defense_cf_risk_ratio < 0.0:
        raise SystemExit("--defense-cf-risk-ratio must be non-negative")
    if args.defense_aux_weight > 0.0 and args.augment:
        raise SystemExit("--defense-aux-weight cannot be combined with --augment: "
                         "action labels need an explicit dihedral transform")
    if args.role == "main" and args.reward_mode != "terminal":
        raise SystemExit("the main agent must use terminal reward; tempo is exploiter-only")
    if not args.league:
        raise SystemExit("--league is required: sigma is not stored anywhere else")

    # The node kills a process that lets BLAS thread. Set before the pool exists
    # so the spawned children inherit it; JAX_PLATFORMS stays unset (GPU parent).
    for v in ("OPENBLAS", "OMP", "MKL", "NUMEXPR"):
        os.environ[f"{v}_NUM_THREADS"] = "1"

    state = json.loads(Path(args.league).read_text())
    specs = [m["spec"] for m in state["archive"]]
    if not all(specs):
        raise SystemExit("an archive member has no spec; league.materialise runs first")
    for s in specs:
        name, _, arg = s.partition(":")
        if name in ("ours", "clone") and not Path(arg).exists():
            raise SystemExit(f"archive member {s} points at a missing file")
    if args.defense_aux_weight > 0.0:
        kinds = {s.partition(":")[0] for s in specs}
        if args.defense_opponent not in kinds:
            raise SystemExit(f"--defense-aux-weight needs an archive member of "
                             f"kind {args.defense_opponent!r}; found {sorted(kinds)}")
    # sigma is NOT in the checkpoint (keys: iter/pending/params/archive/payoff);
    # it is a function of the payoff matrix and has to be recomputed here.
    sigma = fictitious_play(dense(state["payoff"]))
    train_sigma, pfsp_index = sigma, None
    if args.sampling == "pfsp":
        train_sigma, pfsp_index = pfsp_for_checkpoint(
            state, args.init, args.pfsp_weighting)
    boards = args.games // 2
    evals = args.iters // max(args.eval_every, 1) + 2
    if NN_TRAIN_SEED0 + args.iters * boards >= NN_EVAL_SEED0:
        raise SystemExit("training boards would run into the eval range; "
                         "lower --iters/--games or move NN_EVAL_SEED0")
    if NN_EVAL_SEED0 + evals * (args.eval_games // 2) >= NN_GATE_SEED0:
        raise SystemExit("eval boards would run into the gate range; the gate must "
                         "see boards no checkpoint was selected on")

    if args.maps:
        # A pool slice is ~24 boards, so `pool_grid`'s seed % L makes eval and
        # gate boards overlap however the seeds are chosen. The gate survives
        # that because it is PAIRED -- both nets replay the same games -- but
        # checkpoint selection is then selecting on ~24 boards, and every
        # interval printed below is sized by distinct_games, not by game count.
        print(f"--maps {args.maps}: {len(mapgen.load_pool(args.maps))} boards. "
              f"Measurement games use the pool, so replayed jobs are identical "
              f"games (every interval below is sized by the DISTINCT count) and "
              f"the gate reuses the boards the checkpoint was selected on. "
              f"Pairing removes board difficulty; it does not remove that "
              f"selection, so read the gate as optimistic by ~1 se. Generated "
              f"boards (no --maps) have neither problem.")

    import jax
    import jax.numpy as jnp

    from learn import train as bc
    from learn import valuetrain as vt

    # XLA runs f32 convs in TF32 on an L4 by default, which costs ~1e-2 per logit
    # (tests/test_all.py forces HIGHEST for exactly this reason) and would make
    # dlp0 report a divergence the trainer does not have.
    jax.config.update("jax_default_matmul_precision", "highest")

    print("devices:", jax.devices())
    raw_neff = effective(sigma)
    print(f"archive {len(specs)} members, raw sigma support "
          f"{int((sigma > 0.01).sum())}: "
          + "  ".join(f"{s}={w:.2f}" for s, w in zip(specs, sigma) if w > 0.01)
          + f"  (raw Neff {raw_neff:.2f})")
    if pfsp_index is not None:
        print(f"role {args.role}: PFSP-{args.pfsp_weighting} from archive member "
              f"{pfsp_index} ({specs[pfsp_index]}), Neff {effective(train_sigma):.2f}")
    # The single number that says whether this is a league or single-opponent
    # PPO, printed before a game is played. Nash on a transitive matrix -- which
    # is what an archive of config variants of one controller is -- is a point
    # mass, and the floor does not rescue it.
    train_p = mixture(train_sigma, args.sigma_floor)
    neff = effective(train_p)
    print(f"training mixture: {neff:.2f} effective opponents over {len(specs)} "
          f"members (floor {args.sigma_floor:.2f})")
    if raw_neff < 2.0:
        print("            WARNING raw equilibrium has fewer than two effective "
              "opponents. The floor diversifies TRAINING only; eval and gate "
              "still measure the raw point mass.")
    if neff < 2.0:
        print("            WARNING training itself has fewer than two effective "
              "opponents. Raise --sigma-floor or grow the archive.")

    raw0 = np.load(args.init)
    # The architecture is the checkpoint's, whatever it is; a flag is only ever
    # an assertion about it. arch_of refuses a truncated or mislabelled trunk.
    arch = bc.resolve_arch(args.init, args.layers, args.channels, args.residual)
    keys = policy_keys(arch)
    missing = keys - set(raw0.files)
    if missing:
        raise SystemExit(f"{args.init} is not a policy checkpoint, missing {sorted(missing)}")
    z0 = {k: np.asarray(raw0[k]) for k in keys}
    if (z0["conv0_w"].shape[1] in features.LEGACY_INPUT_CHANNELS
            and features.C > z0["conv0_w"].shape[1]):
        old = z0["conv0_w"]
        stem = np.zeros((old.shape[0], features.C, 3, 3), np.float32)
        stem[:, :old.shape[1]] = old
        z0["conv0_w"] = stem
        print(f"policy input migration: {old.shape[1]} -> {features.C} channels "
              "with zero temporal columns (function-preserving)")
    # Names and depth are not enough: an npz trained against a different
    # features.C passes both and then dies inside bc.forward at the first policy
    # step, three iterations and 60 dead workers later.
    ch = arch["channels"]
    want = {"conv0_w": (ch, features.C, 3, 3),
            "head_w": (features.PER_CELL, ch, 3, 3)}
    bad = [(k, z0[k].shape, s) for k, s in want.items() if z0[k].shape != s]
    if bad:
        raise SystemExit(f"{args.init} has the wrong topology for this build: {bad}")
    print(f"policy: {arch['layers']}x{ch}"
          + (" residual" if arch["residual"] else "")
          + f", {sum(int(z0[k].size) for k in keys)} parameters")
    theta = {k: jnp.asarray(z0[k]) for k in keys}
    theta_ref = dict(theta)          # frozen; never rebound, never in a grad graph
    theta_init = {k: np.asarray(v) for k, v in theta.items()}
    phi = vt.init_params(jax.random.PRNGKey(args.seed), arch,
                         args.value_hidden)   # critic, same size
    if args.init_critic:
        from learn.selfplay import load_critic
        warm = load_critic(args.init_critic, arch,
                           {k: np.asarray(v) for k, v in phi.items()})
        phi = {k: jnp.asarray(v) for k, v in warm.items()}
        print(f"critic: warm start from {args.init_critic}", flush=True)
    print(f"training reward: {args.reward_mode}"
          f"{f' (tempo-eps {args.tempo_eps:.3f})' if args.reward_mode == 'tempo' else ''}; "
          "gates/evaluation use raw W/D/L", flush=True)
    opt_p = {k: (jnp.zeros_like(v), jnp.zeros_like(v)) for k, v in theta.items()}
    opt_v = {k: (jnp.zeros_like(v), jnp.zeros_like(v)) for k, v in phi.items()}

    def p_objective(p, beta, x, mask, idx, old_logp, ref_logp, adv,
                    defense_safe, defense_w):
        logits = bc.forward(p, x)
        loss, aux = policy_loss(logits, mask, idx, old_logp, ref_logp, adv,
                                beta, xp=jnp)
        if args.defense_aux_weight:
            loss = loss + args.defense_aux_weight * defense_pairwise_loss(
                logits, mask, idx, defense_safe, defense_w,
                xp=jnp, margin=args.defense_margin)
        return loss, aux

    @jax.jit
    def p_step(p, opt, t, beta, lr, batch):
        (loss, aux), g = jax.value_and_grad(p_objective, has_aux=True)(p, beta, *batch)
        if args.policy_scope == "head":
            trainable = {"head_w", "head_b", "pass_w", "pass_b"}
            g = {k: (v if k in trainable else jnp.zeros_like(v))
                 for k, v in g.items()}
        g, norm = clip_grads(g, 0.5)
        # lr is traced, not closed over: the collapse guard halves it at runtime
        # and a closed-over python float would silently keep the original.
        p, opt = adam(p, opt, g, t, lr)
        return p, opt, loss, aux, norm

    @jax.jit
    def v_step(q, opt, t, x, ret):
        loss, g = jax.value_and_grad(
            lambda qq: jnp.mean((jnp.tanh(vt.forward(qq, x)) - ret) ** 2))(q)
        g, _ = clip_grads(g, 1.0)
        q, opt = adam(q, opt, g, t, args.critic_lr)
        return q, opt, loss

    # tanh: with gamma=1 the return range is exactly [-1, 1], so the critic reads
    # as 2*P(win) - 1 and cannot run away from a terminal-only signal.
    @jax.jit
    def values_of(q, x):
        return jnp.tanh(vt.forward(q, x))

    @jax.jit
    def ref_logp_of(p, x, mask, idx):
        lp = _log_softmax(jnp.where(mask, bc.forward(p, x), -1e9), jnp)
        return lp[jnp.arange(idx.shape[0]), idx]

    def to_x(a):
        return jnp.asarray(a.astype(np.float32))

    def to_mask(packed):
        return jnp.asarray(np.unpackbits(packed, axis=1)[:, :features.N_ACTIONS].astype(bool))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    live = out.with_suffix(".live.npz")
    publish(theta_init, live)        # exists before the first worker starts

    pool = ProcessPoolExecutor(max_workers=args.workers,
                               mp_context=mp.get_context("spawn"),
                               initializer=_init,
                               initargs=(str(live), tuple(specs), args.maps,
                                         args.max_turns,
                                         args.defense_tail_turns if args.defense_aux_weight else 0,
                                         args.defense_hidden_ratio if args.defense_aux_weight else 0.0,
                                         args.defense_opponent,
                                         args.defense_counterfactual,
                                         args.defense_cf_topk,
                                         args.defense_cf_horizon,
                                         args.defense_cf_stride,
                                         args.defense_cf_min_turn,
                                         args.defense_cf_risk_ratio))

    def play(jobs, weights):
        """Publish, then dispatch. Every sample comes from exactly one snapshot,
        so the stored logp IS the behaviour density and no staleness filter is
        needed -- which is what makes dlp0 a meaningful desync alarm."""
        publish(weights, live)
        return [r for r in pool.map(_rollout, jobs, chunksize=1) if r is not None]

    rng = np.random.default_rng(args.seed)
    beta = BETA0
    # One counter per optimiser. A shared one advances through the critic-only
    # warmup, so the policy's first update would bias-correct with t~139 against
    # moments that are still zero -- a 1.14x step at the first update rising to
    # 2.4x by the tenth, spent exactly on the warm start this run depends on.
    t_p = t_v = 0
    best_score, best_iter = -1.0, -1
    best_theta = dict(theta_init)
    best_phi = {k: np.asarray(v) for k, v in phi.items()}
    warmed, low = False, 0  # standing gate, not a permanent latch
    base_score = None       # eval 0: the initialisation's own score
    rewinds = 0
    lr_scale = 1.0
    started = time.time()
    value_replay = GameBalancedReplay(args.critic_replay_games)

    for it in range(args.iters):
        t0 = time.time()
        snapshot = {k: np.asarray(v) for k, v in theta.items()}
        theta_old = ({k: jnp.asarray(v) for k, v in snapshot.items()}
                     if args.augment else None)
        results = play(build_jobs(it, args.games, train_sigma, rng, args.sigma_floor),
                       snapshot)
        if not results:
            print(f"iter {it:4d}  no usable games", flush=True)
            continue

        xs = np.concatenate([r["x"] for r in results])
        idxs = np.concatenate([r["idx"] for r in results])
        packed = np.concatenate([r["mask"] for r in results])
        old_logp = np.concatenate([r["logp"] for r in results])
        defense_safe = np.concatenate([r.get(
            "defense_safe", np.full(len(r["idx"]), -1, dtype=np.int32))
            for r in results]).astype(np.int32, copy=False)
        defense_w = np.concatenate([r.get(
            "defense_w", np.zeros(len(r["idx"]), dtype=np.float32))
            for r in results]).astype(np.float32, copy=False)
        # Drop the per-game copies: at 256 games the buffer is ~1 GB and keeping
        # both views of it doubles that for no reason.
        episodes = [
            (len(r["idx"]), episode_return(r["z"], r["turns"], args.max_turns,
                                            args.reward_mode, args.tempo_eps))
            for r in results
        ]
        replay_pending = [
            (r["x"], episodes[i][1], f"opponent-{int(r.get('opp', -1))}")
            for i, r in enumerate(results)]
        for r in results:
            r.pop("x", None)
            r.pop("mask", None)
        n = len(idxs)
        # Fixed-size chunks, tail padded by repeating row 0 and sliced off again:
        # a ragged final chunk changes shape every iteration and pays a full XLA
        # recompile each time.
        vals = np.empty(n, dtype=np.float32)
        ref_logp = np.empty(n, dtype=np.float32)
        for i in range(0, n, CHUNK):
            sel = np.arange(i, min(i + CHUNK, n))
            m = len(sel)
            if m < CHUNK:
                sel = np.concatenate([sel, np.zeros(CHUNK - m, dtype=np.int64)])
            xb = to_x(xs[sel])
            vals[i:i + m] = np.asarray(values_of(phi, xb))[:m]
            ref_logp[i:i + m] = np.asarray(ref_logp_of(
                theta_ref, xb, to_mask(packed[sel]), jnp.asarray(idxs[sel])))[:m]

        adv = np.empty(n, dtype=np.float32)
        ret = np.empty(n, dtype=np.float32)
        frac = np.empty(n, dtype=np.float32)
        k = 0
        for length, z in episodes:
            adv[k:k + length] = gae(z, vals[k:k + length], lam=args.lam)
            ret[k:k + length] = z           # MC return; see gae's docstring
            frac[k:k + length] = (
                np.arange(length, dtype=np.float32) / max(length - 1, 1))
            k += length
        # SCALE ONLY. Not re-centred: with a correct critic GAE advantages are
        # already zero-mean, and the empirical mean here is a win/loss-imbalance
        # constant. Subtracting it gives every state whose raw advantage is ~0 --
        # two thirds of the buffer, everything more than ~60 plies from the end --
        # one identical deterministic advantage, which is failure 4's blanket
        # push rebuilt out of the thing that was supposed to prevent it.
        adv = np.clip(adv / (adv.std() + 1e-8), -ADV_CLIP, ADV_CLIP)
        # The critic's honest score, measured on THIS buffer before this
        # iteration's updates touch it. `v` (MSE) reads healthy when the critic
        # predicts a constant; this does not.
        def _evar(mask, prediction=None):
            r = ret[mask]
            v = vals[mask] if prediction is None else prediction
            vr = float(r.var())
            return (float(1.0 - ((r - v).var() / vr))
                    if vr > 1e-9 else float("nan"))

        evar = _evar(slice(None))
        mid = (frac >= 0.2) & (frac < 0.45)
        early, late = frac < 0.1, frac >= 0.9
        evar_e, evar_m, evar_l = _evar(early), _evar(mid), _evar(late)
        scalar_idx = list(range(features.CLOCK, features.BASE_C)) + [
            features.DELTA_MY_ARMY, features.DELTA_OPP_ARMY,
            features.DELTA_LAND_ADV]
        scalars = np.c_[xs[:, scalar_idx, 0, 0].astype(np.float32),
                        np.ones(n, np.float32)]

        def _scalar_evar(mask):
            a, r = scalars[mask], ret[mask]
            return _evar(mask, a @ np.linalg.lstsq(a, r, rcond=None)[0])

        sc_m, sc_l = _scalar_evar(mid), _scalar_evar(late)

        # Warm up on the CRITIC'S SCORE, not on a fixed iteration count. The
        # 1500-iteration run released the policy after 3 iterations with evar
        # still at +0.00..+0.04, so its first hundred updates were driven by
        # advantages that were pure critic noise; eval fell 0.175 -> 0.000 by
        # iteration 160 and spent 310 iterations at a zero win rate. Worse, that
        # collapse spends the behaviour-clone warm start, so what recovers
        # afterwards is PPO-from-scratch, which is the attempt that plateaued at
        # 0.15. Wait until the critic explains something.
        ready = critic_ready(it, evar, args.warm_evar)
        warmed, low, refroze = readiness_step(warmed, low, ready)
        if refroze:
            print(f"  RE-FREEZE evar under {args.warm_evar} for "
                  f"{REFREEZE_AFTER} iterations; policy frozen until the "
                  "critic recovers", flush=True)
        do_policy = warmed
        dlp0 = float("nan")
        kl_u = kl_last = kl_a = ent = pg = gn = vloss = 0.0
        kl_sum, nb, stop = 0.0, 0, False
        mb = min(args.minibatch, n)
        for epoch in range(args.epochs):
            order = rng.permutation(n)
            for s in range(0, n - mb + 1, mb):
                sel = order[s:s + mb]
                if args.augment:
                    host_mask = np.unpackbits(
                        packed[sel], axis=1)[:, :features.N_ACTIONS].astype(bool)
                    host_x, host_mask, host_idx = bc.augment_ppo(
                        xs[sel], host_mask, idxs[sel], int(rng.integers(8)))
                    xb = to_x(host_x)
                    maskb, idxb = jnp.asarray(host_mask), jnp.asarray(host_idx)
                    if do_policy:
                        oldb = ref_logp_of(theta_old, xb, maskb, idxb)
                        refb = ref_logp_of(theta_ref, xb, maskb, idxb)
                    else:
                        oldb = refb = jnp.zeros(len(sel), jnp.float32)
                else:
                    xb = to_x(xs[sel])
                    maskb, idxb = to_mask(packed[sel]), jnp.asarray(idxs[sel])
                    oldb = jnp.asarray(old_logp[sel])
                    refb = jnp.asarray(ref_logp[sel])
                nb += 1
                if do_policy:
                    batch = (xb, maskb, idxb, oldb, refb,
                             jnp.asarray(adv[sel]),
                             jnp.asarray(defense_safe[sel]),
                             jnp.asarray(defense_w[sel]))
                    t_p += 1
                    theta, opt_p, _, aux, norm = p_step(theta, opt_p, t_p, beta,
                                                        args.lr * lr_scale, batch)
                    d_, ku, ka, en, pgv = (float(x) for x in aux)
                    if epoch == 0 and s == 0:
                        dlp0 = d_
                    kl_sum += ku
                    kl_u, kl_last, kl_a, ent, pg, gn = (
                        kl_sum / nb, ku, ka, en, pgv, float(norm))
                vxb = xb
                vret = np.asarray(ret[sel], np.float32)
                if len(value_replay) and mb > 1 and args.critic_replay_frac > 0:
                    nr = min(mb - 1, max(
                        1, int(round(mb * args.critic_replay_frac))))
                    rx, rz = value_replay.sample(
                        rng, nr, source_floor=args.critic_replay_source_floor)
                    vxb = jnp.concatenate([xb[:mb - nr], to_x(rx)], axis=0)
                    vret = np.concatenate([vret[:mb - nr], rz])
                t_v += 1
                phi, opt_v, vl = v_step(phi, opt_v, t_v, vxb, jnp.asarray(vret))
                vloss = float(vl)
                # The clip bounds the ratio, not the step; this KL bounds it, and
                # it is why the warm start is still there after 200 iterations.
                # Tested on the CURRENT minibatch, not on the running mean: each
                # minibatch moves theta further from the fixed theta_old, so the
                # per-update KL ramps and a running mean is half of it -- the stop
                # fired at 0.04 while claiming to bound the step at 0.02.
                if do_policy and kl_last > KL_STOP:
                    stop = True
                    break
            if stop:
                break

        for replay_x, replay_z, replay_source in replay_pending:
            value_replay.add(replay_x, replay_z, replay_source)

        if do_policy:
            # A fixed beta either does nothing or freezes the policy. This is a
            # wall on the runaway and frozen regimes, not a controller.
            if kl_a > ANCHOR_HI:
                beta = min(beta * 2.0, BETA_MAX)
            elif kl_a < ANCHOR_LO:
                beta = max(beta / 2.0, BETA_MIN)

        # W/D/L, not just the pooled rate: every game hitting --max-turns is all
        # draws and prints train-wr 0.50, which is exactly what a healthy 50/50
        # split prints, while every advantage is ~0 and nothing is being learned.
        w = sum(r["z"] > 0 for r in results)
        d = sum(r["z"] == 0 for r in results)
        wr = (w + 0.5 * d) / len(results)
        turns = sum(r["turns"] for r in results) / len(results)
        tag = "  [critic warmup]" if not do_policy else ""
        print(f"iter {it:4d}  games {len(results)}  W/D/L {w}/{d}/"
              f"{len(results) - w - d}  samp {n // 1000:3d}k  turns {turns:.0f}  "
              f"train-wr {wr:.2f}  defense-labels {int(defense_w.sum())}"
              f"{tag}\n"
              f"          pg {pg:+.4f}  v {vloss:.3f}  evar {evar:+.2f} "
              f"(e{evar_e:+.2f} m{evar_m:+.2f} l{evar_l:+.2f} "
              f"sc m{sc_m:+.2f} l{sc_l:+.2f})  "
              f"dlp0 {dlp0:.1e}  klU {kl_u:.3f}/{kl_last:.3f}  klA {kl_a:.3f}  "
              f"beta {beta:.3f}  ent {ent:.2f}  gn {gn:.2f}  mb {nb}  "
              f"{time.time() - t0:.0f}s", flush=True)

        if it % args.eval_every == 0 or it == args.iters - 1:
            cur = {k: np.asarray(v) for k, v in theta.items()}
            off = (it // max(args.eval_every, 1)) * (args.eval_games // 2)
            jobs = match_jobs(sigma, args.eval_games, NN_EVAL_SEED0 + off)
            ev = play(jobs, cur)
            score, per = tally(ev, specs)
            mark = ""
            if score > best_score:
                best_score, best_iter, best_theta = score, it, cur
                best_phi = {k: np.asarray(v) for k, v in phi.items()}
                # The selected checkpoint on disk, every time it changes: this
                # run is ~1 h and had nothing but `.live.npz` (the CURRENT net,
                # not the best one) to show for a kill at iteration 199.
                publish(cur, out.with_suffix(".best.npz"))
                mark = "  <- kept"
            dg = distinct_games(jobs, args.maps)
            print(f"eval {it:4d}  {score:.3f} +-{stderr(dg):.3f} ({dg} distinct)  "
                  f"[{'  '.join(f'{s} {v:.2f}' for s, v in per.items())}]{mark}",
                  flush=True)

            # Collapse guard. eval 0 is the initialisation's own score and the
            # only baseline that matters: falling well below it means the warm
            # start is being destroyed, and nothing after that point is training
            # from a clone any more. The 1500-iteration run fell to 0.000 and ran
            # 310 iterations there with nothing watching. Rewind and halve the
            # step rather than continue downhill; give up if it keeps happening,
            # because a third rewind means the setting is wrong, not unlucky.
            if base_score is None:
                base_score = score
            elif score < base_score - 2.0 * stderr(dg):
                rewinds += 1
                print(f"  COLLAPSE {score:.3f} is 2 se below the eval-0 baseline "
                      f"{base_score:.3f}; rewinding to iter {best_iter} and "
                      f"halving lr (rewind {rewinds}/{MAX_REWINDS})", flush=True)
                if rewinds > MAX_REWINDS:
                    print("  giving up: the schedule is wrong, not unlucky. "
                          "Lower --lr or raise --warm-evar.", flush=True)
                    break
                theta, phi, opt_p, opt_v, t_p, t_v = restore_actor_critic(
                    best_theta, best_phi, jnp)
                # Restore a policy-compatible critic and invalidate readiness.
                # Optimiser moments from the abandoned actor distribution must
                # not be replayed into the selected snapshot.
                warmed, low = False, 0
                lr_scale *= 0.5
                beta = min(beta * 2.0, BETA_MAX)

    # --- gate: paired against the initialisation. An absolute threshold would be
    # arbitrary; "did PPO improve on its own starting point against this mixture"
    # is the exact question failures 4 and 5 answered no to, and the same jobs are
    # replayed by both nets. On GENERATED boards these seeds are boards nothing
    # was selected on; with --maps they are not (see the warning at startup).
    gate = match_jobs(sigma, args.gate_games, NN_GATE_SEED0)
    score, per = tally(play(gate, best_theta), specs)
    init_score, init_per = tally(play(gate, theta_init), specs)
    pool.shutdown()

    # Sized by the DISTINCT games, not the job count: replaying a deterministic
    # game adds no information, and with --maps a 400-job gate over a 24-board
    # slice is 48 games. Quoting stderr(400) demanded +0.071 where the arithmetic
    # demands +0.204, and this is the number that decides archive membership.
    dg = distinct_games(gate, args.maps)
    margin = 2 * stderr(dg)
    accepted = bool(score - init_score > margin)
    # publish, not a bare savez: `out` is what enters the league archive and
    # becomes the next generation's --init, so it is the LAST file that should
    # ship without the architecture record.
    publish(best_theta, out)
    critic_out = out.with_suffix(".critic.npz")
    critic_tmp = critic_out.with_suffix(".tmp.npz")
    np.savez(critic_tmp, **{f"phi__{k}": np.asarray(v)
                           for k, v in best_phi.items()})
    os.replace(critic_tmp, critic_out)
    publish(theta, out.with_suffix(".last.npz"))
    out.with_suffix(".json").write_text(json.dumps(
        {"accepted": accepted, "score": round(score, 4),
         "init_score": round(init_score, 4), "margin": round(margin, 4),
         "gate_games": len(gate), "gate_distinct": dg,
         "per_opponent": per, "init_per_opponent": init_per,
         "iters_done": args.iters, "best_iter": best_iter,
         "best_eval": round(best_score, 4),
         "reward_mode": args.reward_mode,
         "tempo_eps": round(args.tempo_eps, 6),
         "defense_aux_weight": round(args.defense_aux_weight, 6),
         "defense_tail_turns": args.defense_tail_turns,
         "defense_hidden_ratio": round(args.defense_hidden_ratio, 6),
         "defense_margin": round(args.defense_margin, 6),
         "defense_opponent": args.defense_opponent,
         "defense_counterfactual": args.defense_counterfactual,
         "defense_cf_topk": args.defense_cf_topk,
         "defense_cf_horizon": args.defense_cf_horizon,
         "defense_cf_stride": args.defense_cf_stride,
         "defense_cf_min_turn": args.defense_cf_min_turn,
         "defense_cf_risk_ratio": round(args.defense_cf_risk_ratio, 6),
         "policy_scope": args.policy_scope}, indent=2) + "\n")

    print(f"\ngate on {len(gate)} fresh games ({dg} distinct): trained {score:.3f} "
          f"vs init {init_score:.3f}, needs +{margin:.3f} -> "
          f"{'ACCEPTED' if accepted else 'REJECTED'}")
    print(f"  per opponent {per}")
    print(f"wrote {out} with matched critic {critic_out}  "
          f"({time.time() - started:.0f}s)")
    from tools import manifest
    inputs = [args.init, args.league, out, critic_out,
              out.with_suffix(".last.npz")]
    if args.init_critic:
        inputs.append(args.init_critic)
    manifest.write(out.with_suffix(".manifest.json"), command=sys.argv,
                   artifacts=inputs,
                   extra={"kind": "population-ppo", "args": vars(args),
                          "gate": {"accepted": accepted, "score": score,
                                   "init_score": init_score, "margin": margin},
                          "opponents": specs, "sigma": sigma.tolist(),
                          "training_sigma": np.asarray(train_sigma).tolist(),
                          "role": args.role, "sampling": args.sampling})
    if accepted:
        print("Confirm it before believing it:")
        print(f"  python -m arena.runner --a clone:{out} --b ours:configs/v16.json "
              f"--games 400 --workers {args.workers}")


if __name__ == "__main__":
    main()
