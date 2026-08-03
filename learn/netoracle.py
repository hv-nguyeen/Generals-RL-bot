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
   the MONTE-CARLO return, three iterations of CRITIC-ONLY warmup before the
   first policy step, and the advantage is SCALED but not re-centred. Centring
   was the bug, not the cure: with gamma=1, terminal-only reward and lam=0.95
   every state more than ~60 plies from the end has a raw advantage of ~0, so
   subtracting the buffer mean hands two thirds of the batch one identical
   deterministic advantage whose sign is set by the batch win rate -- failure 4
   verbatim, switching on exactly when the run starts winning. The baseline is
   the critic; watch `evar`.
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
the buffer BEFORE this iteration's critic updates. It is the only number that
says whether GAE is producing signal or filtered noise; `v` is MSE against a
target that is mostly 0 and reads healthy when the critic is useless.

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
                        The critic explains nothing, so every advantage is a
                        lam-filtered difference of critic noise. Nothing the
                        policy learns after this point is credit assignment.
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
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from arena import agents
from bot import features, rules
from bot.policy.net import CHANNELS, LAYERS, Net
from learn.league import allocate, dense, fictitious_play, stderr
from sim import engine, mapgen

# Board origins. Training boards are GENERATED and there are 128 fresh ones per
# iteration, so the three ranges must not collide; main() asserts the training
# span never reaches NN_EVAL_SEED0.
NN_TRAIN_SEED0 = 400_000
NN_EVAL_SEED0 = 600_000
NN_GATE_SEED0 = 700_000

GAMMA = 1.0          # terminal-only reward; discounting a win 500 turns out is
LAM = 0.95           # bias with no upside. lam caps credit at ~20 steps.
CLIP = 0.2
ENTROPY = 0.003
ADV_CLIP = 5.0
SIGMA_FLOOR = 0.15   # training only; eval and the gate use raw sigma
KL_STOP = 0.02       # the clip flattens the gradient, it does not bound the step
BETA0, BETA_MIN, BETA_MAX = 0.05, 0.01, 1.0
ANCHOR_HI, ANCHOR_LO = 1.0, 0.1
WARMUP = 3           # critic-only iterations; a random critic is failure 4
CHUNK = 8192         # ingest forward chunk, no-grad, outside value_and_grad

POLICY_KEYS = ({f"conv{i}_w" for i in range(LAYERS)}
               | {f"conv{i}_b" for i in range(LAYERS)}
               | {"head_w", "head_b", "pass_w", "pass_b"})


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


def gae(z: float, v: np.ndarray, gamma: float = GAMMA, lam: float = LAM):
    """Advantages for ONE complete episode with reward z at the end.

    Episodes are whole games, so V(s_T) is 0 by definition and there is no
    truncation boundary to bootstrap across -- `learn/rl.py`'s self-bootstrap at
    the window edge cannot happen here.

    Advantages only. The critic's target is the MONTE-CARLO return, which with
    gamma=1 and terminal-only reward is exactly z at every state -- unbiased, and
    with no variance to trade away because there is no intermediate reward to
    average over. It is literally the label `learn/valuedata.py` writes. The
    lam-return `adv + v` is not usable here: the critic starts near 0, so the
    target is 0.95^(T-1-t) * z ~ 0 for most of the game, the critic refits its
    own zero, and the terminal signal creeps back 1/(1-lam) = 20 plies per
    iteration -- twenty iterations to reach the opening of a 384-turn game, which
    is the whole window in which the earlier runs collapsed.
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


def _init(live: str, specs: tuple, maps: str | None, max_turns: int) -> None:
    _CTX.update(live=live, specs=specs, maps=maps, max_turns=max_turns)


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
    foe = agents.make(_CTX["specs"][opp], 1 - seat, *grid.shape, seed)
    rng = np.random.default_rng(seed * 2 + seat)

    xs, ids, masks, lps = [], [], [], []
    turns, faults = 0, 0
    for turns in range(1, _CTX["max_turns"] + 1):
        obs = engine.observe(st, seat)
        mask = features.legal_mask(obs)          # PASS is always legal: never empty
        lg = np.where(mask, net.logits(obs), -np.inf)
        lg -= lg.max()
        p = np.exp(lg)
        p /= p.sum()
        idx = int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p))
        if not greedy:
            xs.append(features.encode(obs).astype(np.float16))
            ids.append(idx)
            masks.append(mask)
            lps.append(np.log(p[idx]))
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
    out.update(x=np.stack(xs), idx=np.asarray(ids, dtype=np.int32),
               mask=np.packbits(np.stack(masks), axis=1),
               logp=np.asarray(lps, dtype=np.float32))
    return out


def publish(p: dict, live: Path) -> None:
    """Atomic: 60 workers np.load this path while we write it.

    `exploit.py` savez's straight onto the live file; a torn read is one bad
    iteration you would never diagnose.
    """
    tmp = live.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v) for k, v in p.items()})
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
    # the lam=0.95 default is why the critic target is NOT adv+v: 100 plies from
    # the end a zero critic would be asked to predict 0.006 for a won game.
    a = gae(1.0, np.zeros(101, dtype=np.float32))
    assert abs(a[0] - 0.95 ** 100) < 1e-6 and a[0] < 0.01, a[0]

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
    assert packed.shape == (2, 442), packed.shape
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

    # --- a checkpoint round-trips into the loader the submission uses
    p = {f"conv{i}_w": rng.normal(size=(32, 12 if i == 0 else 32, 3, 3)).astype(np.float32) * 0.1
         for i in range(LAYERS)}
    p.update({f"conv{i}_b": np.zeros(32, np.float32) for i in range(LAYERS)})
    p["head_w"] = rng.normal(size=(features.PER_CELL, 32, 3, 3)).astype(np.float32) * 0.1
    p["head_b"] = np.zeros(features.PER_CELL, np.float32)
    p["pass_w"] = np.zeros(32, np.float32)
    p["pass_b"] = np.float32(0.0)
    assert set(p) == POLICY_KEYS, set(p) ^ POLICY_KEYS
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ck.npz"
        publish(p, path)
        assert set(np.load(path).files) == POLICY_KEYS
        net = Net(str(path))                     # the exact loader ClonePolicy uses
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
    ap.add_argument("--out", default="/local/data/vng205/oracle-nn.npz")
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--games", type=int, default=256, help="per iteration, both seats")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--critic-lr", type=float, default=1e-3)
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return
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
    # sigma is NOT in the checkpoint (keys: iter/pending/params/archive/payoff);
    # it is a function of the payoff matrix and has to be recomputed here.
    sigma = fictitious_play(dense(state["payoff"]))
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
    print(f"archive {len(specs)} members, sigma support "
          f"{int((sigma > 0.01).sum())}: "
          + "  ".join(f"{s}={w:.2f}" for s, w in zip(specs, sigma) if w > 0.01))
    # The single number that says whether this is a league or single-opponent
    # PPO, printed before a game is played. Nash on a transitive matrix -- which
    # is what an archive of config variants of one controller is -- is a point
    # mass, and the floor does not rescue it.
    train_p = mixture(sigma, args.sigma_floor)
    neff = effective(train_p)
    print(f"training mixture: {neff:.2f} effective opponents over {len(specs)} "
          f"members (floor {args.sigma_floor:.2f})")
    if neff < 2.0:
        print("            WARNING fewer than two effective opponents. This is "
              "failure 3 -- frozen-opponent PPO -- and the eval bracket will "
              "print a single entry, so the farming tripwire cannot fire. Raise "
              "--sigma-floor or grow the archive before believing the gate.")

    z0 = np.load(args.init)
    missing = POLICY_KEYS - set(z0.files)
    if missing:
        raise SystemExit(f"{args.init} is not a policy checkpoint, missing {sorted(missing)}")
    # Names are not enough: an npz trained at a different CHANNELS/LAYERS/features.C
    # passes the name check and then dies inside bc.forward at the first policy
    # step, three iterations and 60 dead workers later.
    want = {"conv0_w": (CHANNELS, features.C, 3, 3),
            "head_w": (features.PER_CELL, CHANNELS, 3, 3)}
    bad = [(k, z0[k].shape, s) for k, s in want.items() if z0[k].shape != s]
    if bad:
        raise SystemExit(f"{args.init} has the wrong topology for this build: {bad}")
    theta = {k: jnp.asarray(z0[k]) for k in POLICY_KEYS}
    theta_ref = dict(theta)          # frozen; never rebound, never in a grad graph
    theta_init = {k: np.asarray(v) for k, v in theta.items()}
    phi = vt.init_params(jax.random.PRNGKey(args.seed))
    opt_p = {k: (jnp.zeros_like(v), jnp.zeros_like(v)) for k, v in theta.items()}
    opt_v = {k: (jnp.zeros_like(v), jnp.zeros_like(v)) for k, v in phi.items()}

    def adam(p, opt, g, t, lr):
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
        norm = jnp.sqrt(sum(jnp.sum(v ** 2) for v in g.values()))
        s = jnp.minimum(1.0, limit / (norm + 1e-8))
        return {k: v * s for k, v in g.items()}, norm

    def p_objective(p, beta, x, mask, idx, old_logp, ref_logp, adv):
        return policy_loss(bc.forward(p, x), mask, idx, old_logp, ref_logp, adv,
                           beta, xp=jnp)

    @jax.jit
    def p_step(p, opt, t, beta, batch):
        (loss, aux), g = jax.value_and_grad(p_objective, has_aux=True)(p, beta, *batch)
        g, norm = clip_grads(g, 0.5)
        p, opt = adam(p, opt, g, t, args.lr)
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
                               initargs=(str(live), tuple(specs), args.maps, args.max_turns))

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
    started = time.time()

    for it in range(args.iters):
        t0 = time.time()
        snapshot = {k: np.asarray(v) for k, v in theta.items()}
        results = play(build_jobs(it, args.games, sigma, rng, args.sigma_floor),
                       snapshot)
        if not results:
            print(f"iter {it:4d}  no usable games", flush=True)
            continue

        xs = np.concatenate([r["x"] for r in results])
        idxs = np.concatenate([r["idx"] for r in results])
        packed = np.concatenate([r["mask"] for r in results])
        old_logp = np.concatenate([r["logp"] for r in results])
        # Drop the per-game copies: at 256 games the buffer is ~1 GB and keeping
        # both views of it doubles that for no reason.
        episodes = [(len(r["idx"]), r["z"]) for r in results]
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
        k = 0
        for length, z in episodes:
            adv[k:k + length] = gae(z, vals[k:k + length])
            ret[k:k + length] = z           # MC return; see gae's docstring
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
        vr = float(ret.var())
        evar = float(1.0 - ((ret - vals).var() / vr)) if vr > 1e-9 else float("nan")

        do_policy = it >= WARMUP
        dlp0 = float("nan")
        kl_u = kl_last = kl_a = ent = pg = gn = vloss = 0.0
        kl_sum, nb, stop = 0.0, 0, False
        mb = min(args.minibatch, n)
        for epoch in range(args.epochs):
            order = rng.permutation(n)
            for s in range(0, n - mb + 1, mb):
                sel = order[s:s + mb]
                xb = to_x(xs[sel])
                nb += 1
                if do_policy:
                    batch = (xb, to_mask(packed[sel]), jnp.asarray(idxs[sel]),
                             jnp.asarray(old_logp[sel]), jnp.asarray(ref_logp[sel]),
                             jnp.asarray(adv[sel]))
                    t_p += 1
                    theta, opt_p, _, aux, norm = p_step(theta, opt_p, t_p, beta, batch)
                    d_, ku, ka, en, pgv = (float(x) for x in aux)
                    if epoch == 0 and s == 0:
                        dlp0 = d_
                    kl_sum += ku
                    kl_u, kl_last, kl_a, ent, pg, gn = (
                        kl_sum / nb, ku, ka, en, pgv, float(norm))
                t_v += 1
                phi, opt_v, vl = v_step(phi, opt_v, t_v, xb, jnp.asarray(ret[sel]))
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
              f"train-wr {wr:.2f}{tag}\n"
              f"          pg {pg:+.4f}  v {vloss:.3f}  evar {evar:+.2f}  "
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
                # The selected checkpoint on disk, every time it changes: this
                # run is ~1 h and had nothing but `.live.npz` (the CURRENT net,
                # not the best one) to show for a kill at iteration 199.
                publish(cur, out.with_suffix(".best.npz"))
                mark = "  <- kept"
            dg = distinct_games(jobs, args.maps)
            print(f"eval {it:4d}  {score:.3f} +-{stderr(dg):.3f} ({dg} distinct)  "
                  f"[{'  '.join(f'{s} {v:.2f}' for s, v in per.items())}]{mark}",
                  flush=True)

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
    np.savez_compressed(out, **best_theta)
    np.savez_compressed(out.with_suffix(".last.npz"),
                        **{k: np.asarray(v) for k, v in theta.items()})
    out.with_suffix(".json").write_text(json.dumps(
        {"accepted": accepted, "score": round(score, 4),
         "init_score": round(init_score, 4), "margin": round(margin, 4),
         "gate_games": len(gate), "gate_distinct": dg,
         "per_opponent": per, "init_per_opponent": init_per,
         "iters_done": args.iters, "best_iter": best_iter,
         "best_eval": round(best_score, 4)}, indent=2) + "\n")

    print(f"\ngate on {len(gate)} fresh games ({dg} distinct): trained {score:.3f} "
          f"vs init {init_score:.3f}, needs +{margin:.3f} -> "
          f"{'ACCEPTED' if accepted else 'REJECTED'}")
    print(f"  per opponent {per}")
    print(f"wrote {out}  ({time.time() - started:.0f}s)")
    if accepted:
        print("Confirm it before believing it:")
        print(f"  python -m arena.runner --a clone:{out} --b ours:configs/v16.json "
              f"--games 400 --workers {args.workers}")


if __name__ == "__main__":
    main()
