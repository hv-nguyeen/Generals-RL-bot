"""Self-play PPO under a curriculum on the distance between the two generals.

WHY THIS EXISTS
---------------
Every RL run in this project has trained on the competition board distribution,
where the two generals are at least 17 BFS steps apart. That means ~500 turns of
play for ONE bit of reward, and `docs/ml-log.md` attempt 6 is what that costs: a
critic that never explained more than 12% of the return, a policy released on
its noise, 310 iterations at a 0% win rate, and a plateau at 0.22 that is
plausibly just PPO-from-scratch after the warm start was spent.

The reward is not sparse because the game is hard. It is sparse because the
generals start far apart. Seat them 2-6 steps apart and a game is ~80 turns, the
outcome is largely determined by the board, and the critic has something to fit.
That is the hypothesis, and the whole design exists to test it and to be
falsifiable if it is wrong (kill 2 below).

TAKEN FROM AVERAGEJOE, NOT TAKEN
--------------------------------
strakam's AverageJoe (#1 on the 1v1 ladder) reached that with 26 billion
transitions; we get ~195 million. Copying the run is not on the table, so this
takes only the techniques that attack OUR measured failure:

  TAKEN   the generals-distance curriculum, advancing on a win-rate gate --
          the technique AverageJoe uses, with the stage tails re-fitted to our
          ruleset (18-21 grids at 24-26% mountains top out near 35 steps, so
          their 17-42 final stage would be mostly unreachable here). The gate is
          against the policy's OWN stage-entry weights, not against a fixed bot:
          every fixed opponent we have is calibrated to distance 17 and would be
          measuring its own blind spot at stage 0;
  TAKEN   self-play against the CURRENT parameters -- not an EMA copy, not a
          snapshot pool. It is what makes the low stages non-degenerate: against
          a fixed strong opponent at distance 4 the outcome is near-deterministic
          and an all-loss batch is failure 4 verbatim. Self-play pins the
          training win rate at 0.5 at every stage;
  TAKEN   one epoch per batch. We ran two, which amplified failure 6's collapse.

  FLAG    HL-Gauss distributional value, `--value-head hlgauss`, DEFAULT OFF.
          Our returns have support {-1, 0, +1} exactly, so at their 128 bins and
          sigma 0.04 the three label rows are DISJOINT and this is a 3-way
          softmax with fixed smoothing -- it adds no distributional information
          and the curriculum, not the loss, is the treatment for a critic with
          nothing to fit. What it does fix is narrower and real: the MSE target
          sits ON the tanh asymptote, so the gradient carries a (1 - tanh^2)
          factor that vanishes exactly where the critic is confidently wrong.
          Turn it on only if the `l` field of evar says the critic fails on
          nearly-decided positions (see READING THE OUTPUT), and merge it only
          on an arena.runner verdict;
  NOT     the magnet KL. We already have it wearing different clothes: the k3
          anchor to the behaviour clone is a permanent prior with a beta
          guardrail, not a decaying one;
  NOT     adv_top_frac 0.25 (positive-only learning is failure 5's design), EMA
          eval weights (their 0.999 is 1/100 of a run for them and the whole run
          for us), the entropy schedule (0.05 is calibrated to training from
          random; we warm start from a clone at ~2.9 nats);
  NOT     the vectorised JAX env. The CPU pool calls `bot/features.encode` --
          the submission's own code -- so encoder drift between training and
          match play cannot happen, and whole-game episodes give V(s_T) = 0 by
          definition with no truncation boundary. The curriculum makes the pool
          FASTER, not slower: short games are the cheap ones. Revisit when games
          rather than ideas are the bottleneck.

READING THE OUTPUT
------------------
    iter   247  stage 2 (7-13)  games 256  W/D/L 128/4/124  samp  71k
                turns 138  dist 9.8
                pg -0.0132  v 0.214  evar 0.31 (e+0.04 l+0.88)  dlp0 4.1e-04
                klU 0.011/0.019
                klA 0.38  beta 0.050  ent 2.91  gn 0.42  mb 17  6s
    stage-eval  250  0.71 +-0.050 (200/200 games, dist 7-13, vs stage-entry self)
                2/2 over 0.60 -> STAGE 3
    comp-eval   250  0.24 +-0.035 (400/400 games, dist 17+)  base 0.19
                kill<0.089  <- kept

The self-play win rate is 0.5 by construction and is never printed as progress.
COMP-EVAL IS THE ONLY PROGRESS NUMBER: 400 argmax games on the COMPETITION board
distribution against a fixed `ours:configs/v16.json`, at every stage regardless
of where the curriculum is, so it is comparable across the run and directly
against `docs/ml-log.md` -- the clone's 0.175 is this number. `comp-eval` at
iteration 0 is `base_score`, the initialisation's own score; recompute it, do not
import it from the log.

Comp-eval is a PROGRESS READOUT below stage 3 and a CONTROL SIGNAL at stage 3+,
and the difference is deliberate. Below stage 3 the policy is training at
distance 4-13 and this number is measured at 17+; a fall there is the transfer
gap the curriculum is spending iterations against, not a collapse, so it prints
and selects the best checkpoint but never rewinds. Acting on it at stage 0 would
revert to the iteration-0 clone and halve lr for the rest of the run, on the one
number the design already expects to be bad.

`stage-eval` decides promotion and NOTHING else, and it plays THIS POLICY'S OWN
WEIGHTS AS OF STAGE ENTRY. That is a fixed point: two argmax copies of one
network play the same deterministic game whichever way they are seated, and
`eval_jobs` plays both seats of every board, so an unchanged policy scores
exactly 0.500 and 0.60 means "beat the weights you entered this stage with".
It replaces a gate against v16, which was not measuring theta at all --
`bot/belief.py` builds v16's enemy-general candidate set as
`passable & (dist_from_my_general >= 17)`, so below stage 4 the true general is
not in that set and v16 hunts an empty ring while its own base sits undefended a
few tiles away. That gate was satisfied by the INITIALISATION, before a single
policy step, and would have walked the curriculum to stage 4 by iteration ~160
with comp-eval unchanged at 0.175.

Being a fixed point also makes stage-eval the only collapse detector measured ON
the distribution being trained: below 0.35 (3 x stderr(200) under 0.500) the
policy has lost to its own recent self and the rewind fires at any stage.

`dist` is the mean MEASURED generals distance over the iteration's boards.
`mapgen.generate` falls back to a wider ring rather than failing, so this is the
only proof the curriculum reached the generator.

`evar` is measured on the buffer BEFORE this iteration's critic updates;
measuring after is self-congratulation. `dlp0` is a MAX, not a mean: a mean over
per-sample forward noise averages back to 1.0000 and is blind to exactly the
desync it exists to catch.

`evar` PRINTS THREE NUMBERS AND ONLY ONE OF THEM IS A QUALITY SCORE. With
terminal-only reward the critic's target is z at every state, so the optimal
critic V* is a martingale: Var(V*_t) rises from whatever the map alone
determines at t=0 to Var(z) at t=T. The pooled number is therefore an average
over that ramp -- it has a ceiling well under 1.0 that nobody has measured, and
"0.31 is bad" is not a claim until that ceiling is. `e` is the first 10% of each
episode's plies and `l` is the last 10%. In the last 10% the outcome is nearly
decided, so the ceiling there is near 1.0 and `l` IS readable as a fitting
score. `e` << `l` with `l` high says the pooled number is the martingale and the
critic is fine. `l` low says the critic cannot fit even a nearly-decided
position, which is the only reading that justifies touching the value loss.

The measurement, ~30 iterations at competition distance with the policy pinned
(`--warm-evar 0.99` never lets `warmed` become true, so the critic fits a
STATIONARY distribution and `l` means what it says):

    python -m learn.selfplay --backend gpu --start-stage 5 --iters 30 \\
        --warm-evar 0.99 --out /local/data/vng205/ceil.npz

`l >= 0.85` -> the critic fits where the answer is knowable; spend the night on
the receptive field instead. `l < 0.60` -> it does not, and `--value-head
hlgauss` is the cheapest thing to try (same command plus that flag; the kill
number is +0.10 absolute on `l`).

KILL THE RUN IF:

    dlp0 > 0.05          worker numpy forward and parent JAX forward diverged.
                         Healthy is 1e-4..1e-3 (f32 plus a float16 buffer); a
                         layout bug prints O(1). This is the bug class that once
                         cost a 200-0 arena result. Nothing downstream survives.
    evar < 0.10 for 20 consecutive iterations after iteration 30, AT STAGE 0
                         THE HYPOTHESIS'S OWN FALSIFIER. If the outcome is not
                         predictable even at distance 2-6, reward density was
                         never the problem. AUTOMATIC (KILL 2): the run stops
                         itself. Without it `warmed` has no timeout and a critic
                         that never fits leaves the policy frozen until morning
                         with nothing promoted and no question answered. The
                         counter is in the iteration line: `[critic warmup 7/20]`.
    D / games > 0.5      every game hitting the turn limit. The rate reads 0.50
                         exactly as a healthy split does while every advantage
                         is ~0.
    turns > 200 at stage 0
                         the short boards are not producing fast kills, so the
                         density argument and the throughput estimate are both
                         void. Visible in iteration 1.
    dist outside the stage range by > 1.0
                         mapgen fell through to its fallback; the curriculum is
                         a no-op wearing a log line.
    stage-eval < 0.35    the policy lost to its own stage-entry weights on the
                         boards it is training on. Measured against a 0.500 fixed
                         point, so this one is unambiguous. Automatic rewind.
    comp-eval below base_score - 0.100 twice running, at stage >= 3
                         THE NEW FAILURE MODE THIS DESIGN INTRODUCES: the
                         curriculum taught something that does not transfer to
                         distance 17. The collapse guard rewinds; a second one
                         at a late stage means stop. 0.100 and not 0.071 because
                         the quantity is a DIFFERENCE of two 400-game estimates
                         and its SE is stderr(400) * sqrt(2) = 0.050; 2 * 0.0354
                         was a 1.41 sigma test at a 7.9% per-eval false-positive
                         rate, i.e. ~2 spurious rewinds a run, each permanently
                         halving lr. The line prints its own threshold as
                         `kill<...`.
    klA past 1.0 with beta stuck at its 1.0 cap
                         the anchor has lost and the warm start is being spent
                         -- failure 5 rebuilding itself.
    klU last value pinned at 0.02 every iteration
                         every update clamped by the early stop; lr too high.
    stage 5 before iteration 200
                         not a kill: the clone walked the ladder and the honest
                         result is "the curriculum did nothing". Much less likely
                         now that the gate is self-referential -- walking it means
                         beating your own weights 60/40 five times, which IS
                         learning -- but the reading still stands.
    rewinds > 3          automatic stop; the schedule is wrong, not unlucky.
    turns falling fast while comp-eval falls
                         learning to die faster -- failure 4's signature. Check
                         the critic warmup ran.

THE COMMANDS, IN ORDER
----------------------
FIRST, the cheap test. ~10 minutes, and it is the whole hypothesis in one number.
Do not start the overnight run until this says the critic can fit at distance 2-6.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    /local/data/vng205/venv/bin/python -m learn.selfplay \\
        --init /local/data/vng205/clone.npz \\
        --out  /local/data/vng205/probe.npz \\
        --workers 60 --games 256 --probe 60

THEN the night, if and only if the probe passed:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
    /local/data/vng205/venv/bin/python -m learn.selfplay \\
        --init /local/data/vng205/clone.npz \\
        --out  /local/data/vng205/sp.npz \\
        --workers 60 --iters 1200 --games 256

`--stage-cap` defaults to 200 because 5 forced promotions at 250 land on
iteration 1250 and `--iters` is 1200: the run would end BEFORE stage 5 and the
final gate would score a policy that never trained on a competition board.
5 x 200 = 1000 leaves 200 iterations at the competition distribution. The
argument parser prints a warning if you break this.

JAX_PLATFORMS stays UNSET: the parent wants the L4, and the spawned workers
import only numpy/bot/sim/arena. Preemption is expected on that node -- rerun the
SAME command with `--resume` appended; without it a restart begins the curriculum
again at distance 2 with a policy that has already left it.

The ship decision is still an SPRT, not any number printed here -- and in
particular not `best_comp_eval` in the output json, which is a max over ~24 draws
at +-0.035 and carries the winner's curse `docs/ml-log.md` already records:

    python -m arena.runner --a clone:<out> --b ours --games 400
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from arena import agents
from bot import features, rules
from bot.policy.net import Net, arch_of, arch_record
from learn.league import stderr
from learn.netoracle import (ADV_CLIP, ANCHOR_HI, ANCHOR_LO, BETA0, BETA_MAX,
                             BETA_MIN, CHUNK, KL_STOP, MAX_REWINDS, WARMUP,
                             _log_softmax, adam, clip_grads, gae, policy_keys,
                             policy_loss, publish, tally)
from sim import engine, mapgen

# (dmin, dmax, win rate needed to ENTER this stage). Ranges overlap on purpose:
# a stage boundary is a shift of the distribution, not a cut. Stage 5 is
# `generate(seed)` with its defaults, i.e. EXACTLY the competition generator, so
# the final stage exercises the code path that already exists and its boards are
# comparable with every earlier run. Grid size stays 18-21 at every stage --
# copying the curriculum by shrinking boards instead would change the
# observation statistics and break the transfer that is the whole point.
STAGES: list[tuple[int, int | None, float]] = [
    (2, 6, 0.00),        # 0: entry, thresholdless
    (4, 9, 0.60),
    (7, 13, 0.60),
    (11, 17, 0.60),
    (17, 24, 0.60),
    (17, None, 0.60),    # 5: the competition distribution
]

# Board origins. Nothing is shared between these four blocks; main() asserts it
# before a worker starts.
SP_TRAIN_SEED0 = 1_000_000
SP_STAGE_SEED0 = 1_400_000     # promotion gate, current stage's distance
SP_COMP_SEED0 = 1_500_000      # progress, ALWAYS competition distance
SP_GATE_SEED0 = 1_600_000      # final paired gate, seen by nothing else
SP_POOL_SEED0 = 2_000_000      # `tools/pools.py`, the prebuilt training boards

PROMOTE_EVAR = 0.10   # promoting into sparser reward with a critic that already
                      # explains nothing is failure 6 with a countdown
PROMOTE_OVER = 2      # consecutive evals over threshold. stderr(200) = 0.05, so
                      # one eval reads +-0.10 around the gate and promotion is
                      # permanent; two halves the confirmed decision's SE to 0.035
DNP_ROWS = 32         # stored rows through the submission's numpy forward each
                      # iteration under --backend gpu; see the `dnp` block
MIN_RESIDENCY = 30    # without it the clone walks five stages in five evals and
                      # the run is a distance-17 run with extra logging
REFREEZE_AFTER = 5    # consecutive low-evar iterations that re-freeze the policy
FROZEN_KILL = 20      # consecutive low-evar iterations at stage 0 that END the
                      # run. This is the hypothesis's own falsifier, and it used
                      # to exist only as prose in the docstring above
STAGE_COLLAPSE = 0.35 # stage-eval is 0.500 by construction against the policy's
                      # own stage-entry snapshot (see run_stage_eval), so 0.35 is
                      # 3 x stderr(200) below a fixed point, measured ON the
                      # distribution being trained. This is the collapse detector
                      # that comp-eval cannot be below stage 3

# Everything save_resume must carry. Asserted at the writer so a key added to
# one caller and not to `selfcheck`'s round-trip cannot reach the cluster and
# KeyError on the resume that the resume exists to survive.
RESUME_SCALARS = frozenset({
    "it", "stage", "over", "resident", "beta", "lr_scale", "t_p", "t_v",
    "warmed", "low", "rewinds", "best_score", "best_iter", "base_score",
    "comp_low", "rng_state"})


def outcome(winner: int, seat: int) -> float:
    """Terminal reward for `seat`. ONE definition, used by both rollout modes.

    It existed twice -- once for measurement, once for self-play -- and the
    self-play copy is the highest-risk expression in the file: flip it on one
    seat and every gradient trains each seat on the other's result, which reads
    as slow noise and not as a bug. `selfcheck` pins the whole table.
    """
    return 0.0 if winner < 0 else (1.0 if winner == seat else -1.0)


def hl_table(m: int, sigma: float, lo: float = -1.0, hi: float = 1.0):
    """Bin centres `(m,)` and the HL-Gauss label table `(3, m)`, indexed `round(z)+1`.

    Row 0 is the label for a loss, row 1 a draw, row 2 a win -- `outcome` above
    has support exactly {-1, 0, +1}, so the whole of HL-Gauss for this reward is
    THREE CONSTANT ROWS built once at startup. No erf at training time, no
    per-sample Gaussian. And it is why the 128 bins buy nothing distributional
    here: at sigma/w = 2.56 the three rows occupy 13/26/13 bins and are
    DISJOINT, so the loss is a 3-way classification wearing 128 outputs. The
    reason to use it anyway is in `--value-head`'s help.

    Each row is N(z, sigma^2) TRUNCATED to [lo, hi] and integrated over the bins
    (the erf difference across bin EDGES, not the density at centres -- the
    density form underflows to 0/0 = NaN once sigma/w < ~0.03, inside a jit,
    with nothing printing why the critic died). `p` sums to 1 by construction
    for any z and any sigma > 0; nothing is clipped.

    At z = +-1 the Gaussian is centred ON the range edge, so half its mass falls
    outside and is renormalised away: row 2 recovers +0.9677, not +1.0. That
    shrink is a UNIFORM SCALE -- row 1 recovers 0 exactly, so the CE optimum
    predicts kappa * V*(s) for one constant kappa -- and GAE differences V and
    then std-normalises the result, so the advantage cannot see it. `selfcheck`
    pins all of that.
    """
    e = np.linspace(lo, hi, m + 1)
    c = (e[:-1] + e[1:]) / 2.0
    F = np.array([[0.5 * (1.0 + math.erf((x - z) / (sigma * math.sqrt(2.0))))
                   for x in e] for z in (-1.0, 0.0, 1.0)])
    p = np.diff(F, axis=1)
    return c.astype(np.float32), (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


def numpy_forward(params, x: np.ndarray) -> np.ndarray:
    """(B, C, 21, 21) -> (B, N_ACTIONS) through bot/policy/net.py's OWN code.

    This is the submission's forward pass, driven from a parameter dict instead
    of a .npz, and it exists for one reason: `dnp`. Under --backend gpu the
    behaviour policy and the training policy are the same JAX function, so the
    numpy/JAX equivalence that `dlp0` used to check every iteration stops being
    checked by anything. A numpy conv with the wrong weight layout produced
    plausible logits, passed every smoke test and lost 200-0; the check is not
    optional. `selfcheck` pins this against `Net.logits` itself.
    """
    from bot.policy import net as npnet

    arch = arch_of(params)
    layers = [(np.asarray(params[f"{k}_w"], np.float32),
               np.asarray(params[f"{k}_b"], np.float32))
              for k in npnet.trunk_keys(arch["layers"], arch["residual"])]
    hw = np.asarray(params["head_w"], np.float32)
    hb = np.asarray(params["head_b"], np.float32)
    pw = np.asarray(params["pass_w"], np.float32)
    pb = float(params["pass_b"])
    out = []
    for row in np.asarray(x, dtype=np.float32):
        h = npnet._trunk(row, layers, arch["residual"])
        move = npnet._conv3x3(h, hw, hb)
        pass_logit = float(pw @ h.mean(axis=(1, 2)) + pb)
        out.append(np.concatenate(
            [np.transpose(move, (1, 2, 0)).reshape(-1), [pass_logit]]))
    return np.stack(out)


def _pack(buf: list, winner: int, turns: int, dist: int) -> list[dict] | None:
    """The two seats' trajectories, each with its OWN states and its OWN reward.

    Pure, and separate from `_rollout` for one reason: a rollout short enough to
    run inside `selfcheck` cannot produce a winner (generals hold 1 army, a move
    needs `src_army - 1 > 0`), so every stub game is a draw and `z0 == -z1`
    reduces to `0.0 == -0.0`. That is vacuously true of a policy that files both
    seats under seat 0's result, which is the mutation worth catching. Handed a
    winner directly, this is testable in microseconds.

    None -- for the WHOLE game, never one seat -- when a trajectory came out
    empty, so a half-recorded game cannot leave an unpaired trajectory in the
    buffer and break the zero-sum property the advantage scaling rests on.
    """
    out = []
    for s in (0, 1):
        xs, ids, masks, lps = buf[s]
        if not ids:
            return None
        out.append({"z": outcome(winner, s), "turns": turns, "dist": dist,
                    "seat": s, "opp": 0,
                    "x": np.stack(xs), "idx": np.asarray(ids, dtype=np.int32),
                    "mask": np.packbits(np.stack(masks), axis=1),
                    "logp": np.asarray(lps, dtype=np.float32)})
    return out


# --------------------------------------------------------------------------
def build_jobs(it: int, games: int, stage: int) -> list[tuple]:
    """Self-play training jobs. ONE job = one board = one game = TWO trajectories.

    There is no seat loop and no opponent index. Both seats are the same
    network on the same board, so seat bias cancels inside the game rather than
    across a pair of them, and the batch is exactly zero-sum.
    """
    dmin, dmax, _ = STAGES[stage]
    base = SP_TRAIN_SEED0 + it * games
    return [(base + b, dmin, dmax, 0, 0) for b in range(games)]


def eval_jobs(n: int, dmin: int, dmax: int | None, seed0: int,
              mode: int = 1) -> list[tuple]:
    """n//2 boards x 2 seats, argmax, the net against `specs[mode - 1]`.

    Argmax because that is how a `clone:` agent actually plays; both seats
    because seat bias is worth a whole result. `mode` doubles as the opponent
    index so a job stays a 5-tuple and the pool never has to be rebuilt to
    change instruments -- 1 is the comp-eval opponent, 2 the stage-entry
    snapshot the promotion gate plays.

    BOTH SEATS ON THE SAME BOARD is what makes the snapshot gate read 0.500
    exactly for an unchanged policy: two argmax copies of the same weights play
    the same deterministic game on a seed whichever way round they are seated,
    so the pair sums to 1.0 and the score is a fixed point, not an estimate.
    """
    assert mode >= 1
    return [(seed0 + b, dmin, dmax, mode, seat)
            for seat in (0, 1) for b in range(n // 2)]


def seed_span(iters: int, games: int, stage_every: int, stage_games: int,
              comp_every: int, comp_games: int) -> tuple[int, int, int]:
    """Highest board seed each block would reach. Compared against the next
    block's origin: an overlap silently trains on the boards that decide
    promotion, or gates on boards a checkpoint was selected on."""
    return (SP_TRAIN_SEED0 + iters * games,
            SP_STAGE_SEED0 + (iters // max(stage_every, 1) + 2) * (stage_games // 2),
            SP_COMP_SEED0 + (iters // max(comp_every, 1) + 2) * (comp_games // 2))


def promote(stage: int, wr: float, evar: float, over: int,
            resident: int, cap: int) -> tuple[int, int]:
    """(new stage, new consecutive-hit count). Pure, so it is unit-testable.

    PROMOTION ONLY -- `stage` can never decrease and at most one stage moves per
    call, so oscillation is impossible by construction and the cost of a noisy
    eval is bounded to one premature promotion. The tempting alternative,
    demoting when the eval craters, is deliberately not built: weight collapse
    already has an owner (the rewind guard), and giving one symptom two
    mechanisms that can fight each other is how a trainer becomes unreadable.

    The threshold belongs to the stage being ENTERED, which is what makes stage 0
    thresholdless. `cap` is a forced promotion: the 0.60 gate is calibrated
    against nothing -- v16 is a strong controller and the gate may simply be
    unreachable at stage 3 -- and a run that stalls all night answers no
    question. Forced, the curriculum degrades into a fixed annealing schedule.
    """
    if stage >= len(STAGES) - 1:
        return stage, 0
    if resident >= cap:
        return stage + 1, 0
    # `not (evar >= ...)`, not `evar < ...`: evar is nan when the buffer's return
    # variance is zero, i.e. when every game drew, and `nan < 0.10` is False --
    # so the naive spelling promotes into a sparser stage precisely in the state
    # (kill 3, every game hitting the turn limit) where nothing was learned.
    if not (evar >= PROMOTE_EVAR) or resident < MIN_RESIDENCY:
        return stage, 0
    over = over + 1 if wr >= STAGES[stage + 1][2] else 0
    return (stage + 1, 0) if over >= PROMOTE_OVER else (stage, over)


# --------------------------------------------------------------------------
# worker side: numpy only, no jax, ever.
_CTX: dict = {}


def _init(live: str, specs: tuple, max_turns: int) -> None:
    _CTX.update(live=live, specs=specs, max_turns=max_turns)


def _act(net: Net, obs, rng):
    """(index, mask, log prob) from the masked softmax. `rng` None means argmax."""
    mask = features.legal_mask(obs)        # PASS is always legal: never empty
    lg = np.where(mask, net.logits(obs), -np.inf)
    lg -= lg.max()
    p = np.exp(lg)
    p /= p.sum()
    idx = int(np.argmax(p)) if rng is None else int(rng.choice(len(p), p=p))
    return idx, mask, float(np.log(p[idx]))


def _rollout(job):
    """One complete game. job = (seed, dmin, dmax, mode, seat).

    mode 0    -- self-play training: BOTH seats are this same network, both
                 sample, both trajectories are returned. 256 games therefore
                 yield 512 trajectories, twice `netoracle`'s samples per game.
    mode >= 1 -- measurement: the net plays `seat` with argmax against
                 `specs[mode - 1]`, and the result is tagged `opp = mode - 1`
                 so `netoracle.tally` scores it per instrument.
    """
    seed, dmin, dmax, mode, seat = job
    grid = mapgen.generate(seed, dmin, dmax)
    dist = mapgen.generals_distance(grid)
    st = engine.from_grid(grid)
    net = Net(_CTX["live"])
    max_turns = _CTX["max_turns"]

    if mode:
        foe = agents.make(_CTX["specs"][mode - 1], 1 - seat, *grid.shape, seed)
        faults, turns = 0, 0
        for turns in range(1, max_turns + 1):
            idx, _, _ = _act(net, engine.observe(st, seat), None)
            acts = [None, None]
            acts[seat] = features.index_to_action(idx)
            try:
                acts[1 - seat] = foe.act(engine.observe(st, 1 - seat))
            except Exception:                        # noqa: BLE001
                # The arena's FAULT budget, not a TIME budget: a timing-dependent
                # reward would make the signal a function of node load. An agent
                # that raises every turn otherwise passes for 1200 turns, banks a
                # win against a do-nothing bot and prints 1.00 where it reads as
                # strength rather than as a crash.
                faults += 1
                if faults >= rules.MAX_FAULTS:
                    return None
                acts[1 - seat] = rules.PASS_ACTION
            if engine.step(st, acts[0], acts[1]):
                break
        return [{"z": outcome(st.winner, seat), "turns": turns, "dist": dist,
                 "seat": seat, "opp": mode - 1}]

    # One shared stream, seat 0 drawn before seat 1, so a seed reproduces the
    # whole game exactly.
    rng = np.random.default_rng(seed)
    buf = [([], [], [], []) for _ in range(2)]
    turns = 0
    for turns in range(1, max_turns + 1):
        acts = [None, None]
        for s in (0, 1):
            obs = engine.observe(st, s)
            idx, mask, lp = _act(net, obs, rng)
            xs, ids, masks, lps = buf[s]
            xs.append(features.encode(obs).astype(np.float16))
            ids.append(idx)
            masks.append(mask)
            lps.append(lp)
            # engine.observe and engine.step are positional by seat. Placing
            # these the wrong way round trains each seat on the other's rewards
            # and looks like slow noise; selfcheck pins both the sign (via
            # `outcome`) and the buffer-to-seat filing (via frame 0).
            acts[s] = features.index_to_action(idx)
        if engine.step(st, acts[0], acts[1]):
            break

    # Turn limit and mutual capture (engine.step sets winner -1) are both draws.
    return _pack(buf, st.winner, turns, dist)


# --------------------------------------------------------------------------
def save_resume(path: Path, arrays: dict, scalars: dict) -> None:
    """Atomic checkpoint of everything a restart cannot recompute.

    The Adam moments and the curriculum position are the two that matter:
    restoring weights while dropping moments re-applies whatever step was in
    flight, and losing the stage index silently restarts the curriculum at
    distance 2 with a policy that has already left it.

    ONE file, ONE os.replace. The scalars used to live in a sibling .json with
    its own replace, and preemption -- which is expected on this node -- between
    the two left NEW weights beside OLD scalars with nothing to detect it: `best`
    from the new npz paired with `best_score` from the old json, so a later and
    genuinely worse comp-eval clears a stale lower bar and overwrites a better
    checkpoint.
    """
    assert set(scalars) == RESUME_SCALARS, set(scalars) ^ RESUME_SCALARS
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v) for k, v in arrays.items()},
             __scalars__=np.array(json.dumps(scalars)))
    os.replace(tmp, path)


def load_resume(path: Path) -> dict | None:
    if not path.exists():
        return None
    z = np.load(path)
    if "__scalars__" not in z.files:
        return None                       # a pre-single-file checkpoint
    return {"arrays": {k: z[k] for k in z.files if k != "__scalars__"},
            "scalars": json.loads(str(z["__scalars__"]))}


def _flat(prefix: str, d: dict) -> dict:
    return {f"{prefix}__{k}": v for k, v in d.items()}


def _unflat(prefix: str, arrays: dict) -> dict:
    p = prefix + "__"
    return {k[len(p):]: v for k, v in arrays.items() if k.startswith(p)}


# --------------------------------------------------------------------------
def selfcheck() -> None:
    """Everything provable without a game, a GPU or a worker. ~0.6 s."""
    import tempfile

    from bot.obs import Obs
    from bot.policy.net import DEFAULT_CHANNELS, trunk_keys

    rng = np.random.default_rng(0)

    # --- the stage table is a curriculum, and its last stage is the competition
    dmins = [s[0] for s in STAGES]
    assert dmins == sorted(dmins), dmins
    highs = [s[1] if s[1] is not None else 10 ** 9 for s in STAGES]
    assert highs == sorted(highs), highs
    assert all(lo <= hi for lo, hi, _ in zip(dmins, highs, STAGES))
    assert STAGES[-1][0] == rules.MIN_GENERALS_DISTANCE and STAGES[-1][1] is None
    assert STAGES[0][2] == 0.0                       # stage 0 is thresholdless

    # --- the final stage IS `generate`'s default, byte for byte
    for s in range(20):
        assert np.array_equal(mapgen.generate(s), mapgen.generate(s, 17, None))

    # --- map generation honours a requested range. The ONLY check that catches a
    #     curriculum that never reached the generator: `generate` falls back to a
    #     wider ring rather than failing, so the request has to be MEASURED.
    #     24 seeds per stage rather than 60 keeps this under a second at 2.7 ms
    #     a board; every stage came out 40/40 in range when it was written.
    for dmin, dmax, _ in STAGES:
        hi = dmax if dmax is not None else 10 ** 9
        d = [mapgen.generals_distance(mapgen.generate(SP_STAGE_SEED0 + s, dmin, dmax))
             for s in range(24)]
        frac = sum(dmin <= x <= hi for x in d) / len(d)
        assert frac >= 0.95, f"stage {dmin}-{dmax}: only {frac:.2f} in range, {d}"
        print(f"  stage {dmin}-{dmax}: {frac:.0%} in range, mean {np.mean(d):.1f}")

    # --- promote(): the whole advance rule, and it cannot oscillate
    OK = dict(evar=0.5, resident=100, cap=250)
    assert promote(1, 0.55, over=0, **OK) == (1, 0)          # below threshold
    assert promote(1, 0.65, over=0, **OK) == (1, 1)          # one hit is not enough
    assert promote(1, 0.65, over=1, **OK) == (2, 0)          # two consecutive
    assert promote(1, 0.65, over=1, evar=0.05,
                   resident=100, cap=250) == (1, 0)          # critic explains nothing
    assert promote(1, 0.65, over=1, evar=0.5,
                   resident=20, cap=250) == (1, 0)           # residency
    assert promote(1, 0.65, over=1, evar=0.5,
                   resident=29, cap=30) == (1, 0)            # cap not reached yet
    assert promote(1, 0.10, over=0, evar=0.5,
                   resident=250, cap=250) == (2, 0)          # forced, wr irrelevant
    assert promote(1, 0.00, over=0, evar=-1.0,
                   resident=0, cap=0) == (2, 0)              # cap overrides all
    assert promote(len(STAGES) - 1, 1.0, over=99, **OK) == (len(STAGES) - 1, 0)
    # a hit after a miss restarts the count: "consecutive" means consecutive
    assert promote(1, 0.55, over=1, **OK) == (1, 0)
    # an all-draw buffer gives evar = nan, and `nan < 0.10` is False
    assert promote(1, 0.99, over=1, evar=float("nan"),
                   resident=100, cap=250) == (1, 0)
    for _ in range(2000):                             # never goes backwards
        st = int(rng.integers(0, len(STAGES)))
        ns, no = promote(st, float(rng.random()), float(rng.normal()),
                         int(rng.integers(0, 5)), int(rng.integers(0, 400)),
                         250)
        assert ns >= st and ns - st <= 1 and no >= 0

    # --- reward signs. The WHOLE table, not one row: the mutation that matters
    #     is one seat's sign, and `z == -z` on a draw is true of every mutant.
    assert outcome(0, 0) == 1.0 and outcome(0, 1) == -1.0
    assert outcome(1, 0) == -1.0 and outcome(1, 1) == 1.0
    assert outcome(-1, 0) == 0.0 and outcome(-1, 1) == 0.0
    for winner in (-1, 0, 1):                        # antisymmetric across seats
        assert outcome(winner, 0) == -outcome(winner, 1)

    # --- _pack: each seat's OWN states carry each seat's OWN reward, with a real
    #     winner. Seat s's marker value is s, so a swapped pairing is visible.
    fake = [([np.full((1, 2, 2), float(s), np.float16)], [s],
             [np.array([True, False])], [-0.5 - s]) for s in (0, 1)]
    for winner, want in ((0, [1.0, -1.0]), (1, [-1.0, 1.0]), (-1, [0.0, 0.0])):
        pk = _pack(fake, winner, 7, 4)
        assert [r["z"] for r in pk] == want, (winner, [r["z"] for r in pk])
        assert [r["seat"] for r in pk] == [0, 1]
        assert [r["turns"] for r in pk] == [7, 7] and [r["dist"] for r in pk] == [4, 4]
        for s in (0, 1):
            assert pk[s]["idx"].tolist() == [s], s
            assert float(pk[s]["x"].flat[0]) == float(s), s
            assert float(pk[s]["logp"][0]) == -0.5 - s, s
    # an empty trajectory drops the WHOLE game, not one seat
    assert _pack([([], [], [], []), fake[1]], 0, 7, 4) is None
    assert _pack([fake[0], ([], [], [], [])], 0, 7, 4) is None
    z0 = 1.0
    a0 = gae(z0, np.zeros(5, dtype=np.float32), 1.0, 1.0)
    a1 = gae(-z0, np.zeros(5, dtype=np.float32), 1.0, 1.0)
    assert np.all(a0 == 1.0) and np.all(a1 == -1.0)
    assert abs(float(np.concatenate([a0, a1]).mean())) < 1e-9

    # --- seed blocks do not collide at the documented budget
    tr, sg, cp = seed_span(1200, 256, 10, 200, 50, 400)
    assert SP_TRAIN_SEED0 < SP_STAGE_SEED0 < SP_COMP_SEED0 < SP_GATE_SEED0 < SP_POOL_SEED0
    assert tr < SP_STAGE_SEED0 and sg < SP_COMP_SEED0 and cp < SP_GATE_SEED0, (tr, sg, cp)
    # Pool boards are TRAINING boards under the vectorised backend, so they get
    # their own block: reusing SP_TRAIN_SEED0 would put the gate's and the
    # comp-eval's boards inside the training distribution the moment the pool
    # grew. `tools.pools` owns the top of the range; the gate block ends well
    # below it at 1.6M + a few thousand.
    from tools.pools import pool_seed_span
    assert SP_GATE_SEED0 < SP_POOL_SEED0
    assert pool_seed_span() > SP_POOL_SEED0
    # jobs land inside their own block
    jt = build_jobs(1199, 256, 0)
    assert len(jt) == 256 and len({j[0] for j in jt}) == 256
    assert max(j[0] for j in jt) < SP_STAGE_SEED0
    assert all(j[3] == 0 for j in jt)
    je = eval_jobs(200, 7, 13, SP_STAGE_SEED0)
    assert len(je) == 200 and all(j[3] == 1 for j in je)
    assert sorted(j[4] for j in je) == [0] * 100 + [1] * 100
    assert len({j[0] for j in je}) == 100               # 100 boards, both seats

    # --- illegal actions get exactly zero probability, and the PPO ratio is
    #     exactly 1 on the first minibatch of an update (theta == theta_old)
    n, k = 8, 64
    logits = rng.normal(size=(n, k))
    mask = rng.random((n, k)) < 0.3
    mask[:, 0] = True                                   # PASS is always legal
    lp = _log_softmax(np.where(mask, logits, -1e9), np)
    probs = np.exp(lp)
    assert np.all(probs[~mask] == 0.0), probs[~mask].max()
    assert np.allclose(probs.sum(axis=1), 1.0)
    idx = np.array([int(np.argmax(np.where(mask[i], logits[i], -np.inf))) for i in range(n)])
    old = lp[np.arange(n), idx]
    assert np.all(np.exp(old - old) == 1.0)             # the ratio, exactly
    loss, (dlp, kl_u, kl_a, ent, pg) = policy_loss(
        logits, mask, idx, old, old.copy(), rng.normal(size=n), beta=0.5)
    assert dlp == 0.0 and kl_u == 0.0 and kl_a == 0.0   # exactly, not approximately

    # --- mask round trip through the wire format
    m = rng.random(features.N_ACTIONS) < 0.1
    packed = np.packbits(np.stack([m, ~m]), axis=1)
    back = np.unpackbits(packed, axis=1)[:, :features.N_ACTIONS].astype(bool)
    assert np.array_equal(back[0], m) and np.array_equal(back[1], ~m)

    # --- a checkpoint round-trips into the loader the SUBMISSION uses, with
    #     exactly the keys bot/policy/net.py requires
    arch = {"layers": 4, "channels": DEFAULT_CHANNELS, "residual": False}
    p, prev = {}, features.C
    for name in trunk_keys(arch["layers"], arch["residual"]):
        p[f"{name}_w"] = (rng.normal(size=(arch["channels"], prev, 3, 3)) * 0.1).astype("f4")
        p[f"{name}_b"] = np.zeros(arch["channels"], np.float32)
        prev = arch["channels"]
    p["head_w"] = (rng.normal(size=(features.PER_CELL, prev, 3, 3)) * 0.1).astype("f4")
    p["head_b"] = np.zeros(features.PER_CELL, np.float32)
    p["pass_w"] = np.zeros(prev, np.float32)
    p["pass_b"] = np.float32(0.0)
    assert set(p) == policy_keys(arch), set(p) ^ policy_keys(arch)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ck.npz"
        publish(p, path)
        assert set(np.load(path).files) == policy_keys(arch) | set(arch_record(p))
        net = Net(str(path))                            # the loader ClonePolicy uses
        assert net.arch == arch, net.arch

        # a live observation still produces a legal, normalised distribution
        ty = np.full((18, 18), rules.T_PLAIN, dtype=np.int8)
        ty[0, 0] = rules.T_GENERAL
        ow = np.zeros((18, 18), dtype=np.int8)
        ow[0, 0] = rules.OWNER_ME
        ar = np.zeros((18, 18), dtype=np.int32)
        ar[0, 0] = 9
        obs = Obs(H=18, W=18, turn=1, my_land=1, my_army=9, opp_land=1, opp_army=1,
                  type_grid=ty, owner_grid=ow, army_grid=ar)
        i, lm, logp = _act(net, obs, None)
        # from (0,0) with 9 army: down and right, whole or split, plus pass.
        # No build: a general is not a legal build site and 9 < 35 anyway.
        assert lm[features.PASS_INDEX] and lm.sum() == 5, lm.sum()
        assert lm[i] and logp <= 0.0

        # --- `numpy_forward` IS Net.logits, driven from a dict. It is what the
        #     `dnp` alarm measures the trainer against once --backend gpu makes
        #     dlp0 structurally zero, so a drift between the two would disable
        #     the only remaining numpy/JAX equivalence check, silently.
        got = numpy_forward(p, features.encode(obs)[None])
        assert got.shape == (1, features.N_ACTIONS), got.shape
        assert np.abs(got[0] - net.logits(obs)).max() == 0.0

        # --- the worker's seat plumbing, on a 3-turn stub rather than a game.
        #     Nothing can capture a general in 3 turns (generals hold 1 army and
        #     a move needs src_army - 1 > 0), so the OUTCOME here is always a
        #     draw and `z0 == -z1` is `0.0 == -0.0` -- vacuous, and it used to be
        #     the only thing asserted. The sign is pinned by `outcome` above; what
        #     this stub has to pin instead is that each seat's states are FILED
        #     UNDER ITS OWN SEAT, which the frame-0 comparison below does.
        _init(str(path), (), 3)
        seed = SP_TRAIN_SEED0
        got = _rollout((seed, 2, 6, 0, 0))
        assert got is not None and len(got) == 2
        assert [r["seat"] for r in got] == [0, 1]
        for r in got:
            assert r["turns"] == 3 and len(r["idx"]) == 3
            assert r["x"].shape == (3, features.C, features.PAD, features.PAD)
            assert r["logp"].shape == (3,) and np.all(r["logp"] <= 0.0)
            assert 2 <= r["dist"] <= 6
        # --- the CPU rollout can now BUILD, which is the point of phase 1 and
        #     needs no GPU: `_rollout` routes index_to_action(idx) straight into
        #     engine.step, and a build index has to survive that trip. Under the
        #     old 3529 space no index decoded to a build at all.
        st = engine.from_grid(mapgen.generate(SP_TRAIN_SEED0, 2, 6))
        r, c = st.gpos[0]
        rc = (r, c + 1 if c + 1 < st.armies.shape[1] else c - 1)
        st.own[0][rc] = True
        st.neutral[rc] = False
        st.armies[rc] = 99                             # 47 = 35 + 12 at distance 1
        m = features.legal_mask(engine.observe(st, 0))
        cells = m[:features.PAD * features.PAD * features.PER_CELL].reshape(
            features.PAD, features.PAD, features.PER_CELL)
        assert cells[rc[0], rc[1], features.BUILD_OFFSET], "a 99-army tile cannot build?"
        idx = features.action_to_index((rules.BUILD, rc[0], rc[1], 0, 0))
        assert m[idx] and features.index_to_action(idx) == (rules.BUILD, *rc, 0, 0)
        engine.step(st, features.index_to_action(idx), rules.PASS_ACTION)
        assert st.castles[rc] and int(st.armies[rc]) == 99 - 47, int(st.armies[rc])

        # the two seats see DIFFERENT boards (their own fog), which is the whole
        # reason each carries its own encode rather than sharing one
        assert not np.array_equal(got[0]["x"], got[1]["x"])
        # ...and seat s's first frame is the encoding of seat s's OWN opening
        # observation, recomputed here from a fresh state. `buf[s] -> buf[1 - s]`
        # survives every other assertion in this block: both buffers still exist
        # and still differ, they are merely swapped.
        fresh = engine.from_grid(mapgen.generate(seed, 2, 6))
        for r in got:
            want = features.encode(engine.observe(fresh, r["seat"])).astype(np.float16)
            assert np.array_equal(r["x"][0], want), r["seat"]
        _CTX.clear()

        # --- resume round-trips: arrays exact, scalars exact, RNG stream intact
        r = np.random.default_rng(7)
        r.random(3)
        st = {"it": 41, "stage": 2, "over": 1, "resident": 12, "beta": 0.05,
              "lr_scale": 0.5, "t_p": 100, "t_v": 130, "warmed": True, "low": 0,
              "rewinds": 1, "best_score": 0.31, "best_iter": 40,
              "base_score": 0.19, "comp_low": 1, "rng_state": r.bit_generator.state}
        # save_resume asserts against RESUME_SCALARS, so a key added to the
        # trainer's write and not to this dict fails HERE rather than as a
        # KeyError on the cluster, after the preemption the resume exists to
        # survive. `comp_low` was already missing when that was only a comment.
        assert set(st) == RESUME_SCALARS, set(st) ^ RESUME_SCALARS
        rp = Path(td) / "sp.resume.npz"
        save_resume(rp, {**_flat("theta", p), **_flat("optp_m", p)}, st)
        got = load_resume(rp)
        assert got is not None and got["scalars"] == st
        assert set(_unflat("theta", got["arrays"])) == set(p)
        for kk, vv in _unflat("theta", got["arrays"]).items():
            assert np.array_equal(vv, p[kk]), kk
        r2 = np.random.default_rng(0)
        r2.bit_generator.state = got["scalars"]["rng_state"]
        assert np.array_equal(r2.random(5), np.random.default_rng(7).random(8)[3:])
        assert load_resume(Path(td) / "absent.npz") is None

    # --- HL-Gauss (--value-head hlgauss). Numpy only, so this exercises the
    #     same arithmetic the jitted trainer differentiates.
    c, T = hl_table(128, 0.04)
    w = float(c[1] - c[0])
    assert c.shape == (128,) and T.shape == (3, 128)
    assert np.allclose(T.sum(axis=1), 1.0, atol=1e-6), T.sum(axis=1)
    # INTERIOR ROUND-TRIP: encode an arbitrary scalar, recover it by expectation
    # over the centres, within one bin width. Catches an edges/centres swap, a
    # sigma in the wrong units, and a normalisation over the wrong axis.
    edges = np.linspace(-1.0, 1.0, 129)
    for z in np.linspace(-0.85, 0.85, 51):
        p = np.diff([0.5 * (1 + math.erf((x - z) / (0.04 * math.sqrt(2))))
                     for x in edges])
        assert abs(float((p / p.sum()) @ c) - z) < w, z      # measured max 1.4e-05
    # The BOUNDARY deliberately does NOT round-trip, and by exactly the
    # half-normal mean: the Gaussian at z = +-1 is centred on the edge, so half
    # its mass is renormalised away. This is the assert that fails if someone
    # "fixes" the shrink by clipping z instead of understanding it.
    assert abs(float(T[2] @ c) - (1 - 0.04 * math.sqrt(2 / math.pi))) < w
    assert abs(float(T[1] @ c)) < 1e-7                       # draw recovers 0 exactly
    # ODD readout. `adv` is scaled but NOT re-centred, and that rests on the
    # critic being antisymmetric across the two mirrored seat trajectories.
    assert np.abs(T[0] - T[2][::-1]).max() < 1e-6
    # A PERFECT prediction costs the TARGET'S ENTROPY, not zero -- an assert of
    # ce < 1e-6 would pass only for a one-hot target, i.e. for sigma -> 0. This
    # is also why v_step_hl reports KL and not raw CE: `v 1.67` for a flawless
    # critic reads as a regression next to the scalar head's MSE.
    H = -(T * np.log(T + 1e-30)).sum(axis=1)
    assert H.min() > 0.5, H                                  # measured [1.67, 2.37, 1.67]
    ce = -(T * _log_softmax(np.log(T + 1e-30), np)).sum(axis=1)
    assert np.abs(ce - H).max() < 1e-4, (ce, H)
    kl = (T * (np.log(T + 1e-12) - _log_softmax(np.log(T + 1e-30), np))).sum(axis=1)
    assert np.abs(kl).max() < 1e-4, kl                       # KL of a perfect fit is 0
    # Recovery is bounded by the simplex for ANY logits, including garbage
    # during warmup -- the property the tanh head bought with a saturating
    # nonlinearity that also kills its own gradient.
    v = np.exp(_log_softmax(rng.normal(size=(7, 128)) * 5, np)) @ c
    assert np.all(np.abs(v) <= abs(float(c[0]))) and np.all(np.isfinite(v))
    # The label index. `rint`, not `astype`: astype TRUNCATES, so a return of
    # -1e-8 would be labelled a LOSS with nothing crashing and nothing printing.
    assert [int(round(outcome(wn, s) + 1)) for wn in (1, -1, 0) for s in (0, 1)] \
        == [0, 2, 1, 1, 2, 0]
    assert int(np.rint(np.float32(-1e-8) + 1.0)) == 1
    print("selfplay selfcheck OK")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="/local/data/vng205/clone.npz",
                    help="behaviour clone to start from and to anchor to")
    ap.add_argument("--out", default="/local/data/vng205/sp.npz")
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--games", type=int, default=256,
                    help="self-play games per iteration; each yields TWO trajectories")
    ap.add_argument("--epochs", type=int, default=1,
                    help="AverageJoe uses 1; we ran 2, which amplified failure 6. "
                         "0 IS LEGAL AND IS A MEASUREMENT: it keeps the rollout, "
                         "the D2H copy, the host pack, the ingest pass and dnp, "
                         "and drops the update. The cross-check for the "
                         "roll/pack/ing/trn split, needing no trust in where a "
                         "timer was placed. TWO THINGS THE OBVIOUS RECIPE GETS "
                         "WRONG. (1) Compare the PRINTED per-iteration seconds, "
                         "not wall clock: comp-eval and stage-eval run after that "
                         "print and --probe fires a 400-game comp-eval at it 0. "
                         "(2) Run A must actually train: `warmed` needs "
                         "it >= WARMUP (3) AND evar >= --warm-evar, so a plain "
                         "--probe 3 never runs p_step and the A/B then prices "
                         "v_step alone and reads as 'not update bound' whatever "
                         "is true. Use --probe 6 --warm-evar -1 and read "
                         "iterations 3-5. Then (A-B)/A is the update's share")
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--critic-lr", type=float, default=1e-3)
    ap.add_argument("--value-head", choices=("scalar", "hlgauss"), default="scalar",
                    help="scalar is tanh+MSE and is THE DEFAULT. hlgauss is a "
                         "distributional head; with our 3-atom return support it "
                         "is a 3-way softmax with fixed smoothing and adds no "
                         "distributional information, so the ONLY reason to use "
                         "it is that `ret` sits ON the tanh asymptote, where the "
                         "MSE gradient carries a (1 - tanh^2) factor that "
                         "vanishes exactly where the critic is confidently "
                         "wrong. Do not merge it on an evar number: run "
                         "arena.runner on the two trained policies")
    ap.add_argument("--hl-bins", type=int, default=128,
                    help="AverageJoe's, lifted whole; not load-bearing")
    ap.add_argument("--hl-sigma", type=float, default=0.04,
                    help="sigma/w = 2.56 at 128 bins over [-1, 1]. THE RATIO is "
                         "the parameter, not sigma: Farebrother's flat band is "
                         "0.5..2 and the failure modes (one-hot label, no "
                         "ordinal structure, NaN in the density form) are all at "
                         "ratios well under 1, so 2.56 is the safe side")
    ap.add_argument("--warm-evar", type=float, default=PROMOTE_EVAR,
                    help="hold the policy frozen until the critic explains this "
                         "much of the return, and re-freeze if it stops")
    ap.add_argument("--opp", default="ours:configs/v16.json",
                    help="the fixed instrument for comp-eval, the only progress "
                         "number; it never generates a training gradient and it "
                         "no longer decides promotion")
    ap.add_argument("--stage-eval-every", type=int, default=10)
    ap.add_argument("--stage-eval-games", type=int, default=200)
    ap.add_argument("--comp-eval-every", type=int, default=50)
    ap.add_argument("--comp-eval-games", type=int, default=400)
    ap.add_argument("--gate-games", type=int, default=400)
    ap.add_argument("--stage-cap", type=int, default=200,
                    help="iterations at one stage before promotion is FORCED. "
                         "5 x this must be under --iters or the run never "
                         "reaches the competition distribution it is gated on")
    ap.add_argument("--max-turns", type=int, default=rules.TURN_LIMIT)
    ap.add_argument("--probe", type=int, default=0, metavar="ITERS",
                    help="CHEAP HYPOTHESIS TEST, run this before the night. Pins "
                         "stage 0, disables promotion, runs ITERS iterations and "
                         "prints whether the critic ever explained --warm-evar of "
                         "the return at distance 2-6. That is the whole claim: if "
                         "it fails, reward density was never the problem and the "
                         "overnight run answers nothing. ~10 min at 60 workers")
    ap.add_argument("--backend", choices=("cpu", "gpu", "scan"), default="cpu",
                    help="rollout backend for TRAINING games only. cpu is the "
                         "process pool and stays the default until the gpu path "
                         "is measured against it; every eval runs on cpu either "
                         "way, because they play arena opponents. scan is gpu "
                         "with the per-turn python loop replaced by a "
                         "--scan-chunk-turn lax.scan: 40x fewer kernel launches "
                         "and 32x fewer host syncs at chunk 40, same boards, "
                         "actions and lengths, logp to 1 ulp (tests/test_all.py).")
    # 40 divides 1200 and gives a break granularity of 40 turns against the
    # python loop's 32, so the wasted-step overshoot is unchanged in practice.
    # Not imported from vecroll: vecroll imports THIS module.
    ap.add_argument("--scan-chunk", type=int, default=40,
                    help="--backend scan: turns per lax.scan. Must divide "
                         "--max-turns. Raising it cuts host round trips and "
                         "raises the device buffer (chunk x 2 x games x 17.6 kB); "
                         "lower this before lowering --games on an OOM.")
    ap.add_argument("--start-stage", type=int, default=0,
                    help="skip the curriculum and pin the first stage. "
                         "MEASUREMENT SCAFFOLDING: a stage-5 rollout number "
                         "otherwise costs an overnight run to reach. Promotion "
                         "still works from here.")
    ap.add_argument("--pool-dir", default="runs/pools",
                    help="--backend gpu/scan: prebuilt boards from `tools.pools`")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    if not 0 <= args.start_stage < len(STAGES):
        raise SystemExit(f"--start-stage {args.start_stage} is not a stage "
                         f"(0..{len(STAGES) - 1})")
    # Fail here, not on the first rollout: a partial last chunk would step past
    # the turn limit, and the turn limit IS the draw rule.
    if args.backend == "scan" and (args.scan_chunk < 1
                                   or args.max_turns % args.scan_chunk):
        raise SystemExit(f"--scan-chunk {args.scan_chunk} must be a positive "
                         f"divisor of --max-turns {args.max_turns}")
    if args.probe:
        # Everything the probe does is subtraction: no promotion (so stage 0 is
        # pinned without a second code path), no comp-eval past the iteration-0
        # baseline, no 800-game final gate.
        args.iters = args.probe
        args.stage_eval_every = args.comp_eval_every = args.probe + 1
    # `it % every` below, and the resume checkpoint rides on the stage eval.
    args.stage_eval_every = max(args.stage_eval_every, 1)
    args.comp_eval_every = max(args.comp_eval_every, 1)
    forced_end = (len(STAGES) - 1) * args.stage_cap
    if not args.probe and forced_end >= args.iters:
        print(f"WARNING --stage-cap {args.stage_cap} x {len(STAGES) - 1} promotions "
              f"= iteration {forced_end}, past --iters {args.iters}. If the 0.60 "
              f"gate is never met the run ends before stage {len(STAGES) - 1} and "
              "the final gate scores a policy that never trained on a competition "
              f"board. Use --stage-cap {(args.iters - 1) // (len(STAGES) - 1)} or "
              "lower.", flush=True)

    # The node kills a process that lets BLAS thread. Set before the pool exists
    # so the spawned children inherit it; JAX_PLATFORMS stays unset (GPU parent).
    for v in ("OPENBLAS", "OMP", "MKL", "NUMEXPR"):
        os.environ[f"{v}_NUM_THREADS"] = "1"

    tr, sg, cp = seed_span(args.iters, args.games, args.stage_eval_every,
                           args.stage_eval_games, args.comp_eval_every,
                           args.comp_eval_games)
    if not (tr < SP_STAGE_SEED0 and sg < SP_COMP_SEED0 and cp < SP_GATE_SEED0):
        raise SystemExit(f"seed blocks collide (train->{tr}, stage->{sg}, comp->{cp}): "
                         "training would run on the boards that decide promotion, "
                         "or the gate on boards a checkpoint was selected on. "
                         "Lower --iters/--games or move the block origins.")
    # specs[0] is the comp-eval instrument; specs[1] is the promotion gate's
    # opponent, which is a FILE this run rewrites on every stage entry (see
    # stage_ref below) rather than a second external bot.
    name, _, arg = args.opp.partition(":")
    if name in ("ours", "clone") and not Path(arg).exists():
        raise SystemExit(f"opponent {args.opp} points at a missing file")
    try:
        # Build it once here rather than discovering a typo inside 60 spawned
        # workers, where `_rollout` would raise per game and the eval would
        # come back empty with no line saying why.
        agents.make(args.opp, 0, 18, 18, 0)
    except Exception as e:                           # noqa: BLE001
        raise SystemExit(f"opponent {args.opp} does not construct: {e}") from e

    import jax
    import jax.numpy as jnp

    from learn import train as bc
    from learn import valuetrain as vt

    # XLA runs f32 convs in TF32 on an L4 by default, which costs ~1e-2 per logit
    # and would make dlp0 report a divergence the trainer does not have.
    jax.config.update("jax_default_matmul_precision", "highest")
    # Cold start compiles the rollout kernel, the two train steps and the two
    # oracles; on the L4 that is minutes before iteration 0 prints anything, and
    # it is paid again on every resume and every probe. Cached next to --out so
    # a stale cache dies with the run directory rather than outliving a change
    # to the kernel.
    jax.config.update("jax_compilation_cache_dir",
                      str(Path(args.out).parent / "jaxcache"))
    # 0.0, not the 1.0 default: this run compiles about six kernels total, so
    # "cache everything" costs a handful of files and the default silently
    # excludes every kernel that takes under a second.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    print("devices:", jax.devices())

    z0 = np.load(args.init)
    arch = bc.resolve_arch(args.init, None, None, None)
    keys = policy_keys(arch)
    missing = keys - set(z0.files)
    if missing:
        raise SystemExit(f"{args.init} is not a policy checkpoint, missing {sorted(missing)}")
    ch = arch["channels"]
    want = {"conv0_w": (ch, features.C, 3, 3),
            "head_w": (features.PER_CELL, ch, 3, 3)}
    bad = [(k, z0[k].shape, s) for k, s in want.items() if z0[k].shape != s]
    if bad:
        raise SystemExit(f"{args.init} has the wrong topology for this build: {bad}")
    print(f"policy: {arch['layers']}x{ch}" + (" residual" if arch["residual"] else "")
          + f", {sum(int(z0[k].size) for k in keys)} parameters")
    # Saved trunk activations dominate the L4's 18.4 GB default pool and scale
    # with layers * channels * minibatch. --init carries the arch and --minibatch
    # is an independent flag, so a bigger clone walks into an OOM with nothing
    # saying why: 4x32 at mb 4096 is 0.9 GB, 6x64 is 2.8 GB, 8x128 is 7.4 GB and
    # OOMs at ~85% of the pool.
    trunk_gb = arch["layers"] * ch * 441 * 4 * args.minibatch / 2 ** 30
    if trunk_gb > 3.0:
        print(f"WARNING saved trunk activations are {trunk_gb:.1f} GB at "
              f"--minibatch {args.minibatch}; the L4 pool is ~18 GB and the "
              f"backward pass roughly doubles this. Halve --minibatch or raise "
              "XLA_PYTHON_CLIENT_MEM_FRACTION.", flush=True)

    theta = {k: jnp.asarray(z0[k]) for k in keys}
    theta_ref = dict(theta)          # frozen; never rebound, never in a grad graph
    theta_init = {k: np.asarray(z0[k]) for k in keys}
    phi = vt.init_params(jax.random.PRNGKey(args.seed), arch)   # critic, same trunk
    if args.value_head == "hlgauss":
        # SAME KEYS, wider. `arch_of` run-length-scans conv*/res* only, so
        # reshaping v_w is invisible to it; `opt_v`, `_flat("phi", ...)` and the
        # resume are comprehensions over phi.items() and need no change; and
        # `vt.forward` -- (B, ch) @ (ch, m) + (m,) -- returns (B, m) logits with
        # no edit at all, so valuetrain.main() and tools/calibrate keep the
        # scalar head they were written against.
        phi["v_w"] = jax.random.normal(jax.random.fold_in(
            jax.random.PRNGKey(args.seed), 8), (ch, args.hl_bins)) * 0.01
        phi["v_b"] = jnp.zeros((args.hl_bins,))
    opt_p = {k: (jnp.zeros_like(v), jnp.zeros_like(v)) for k, v in theta.items()}
    opt_v = {k: (jnp.zeros_like(v), jnp.zeros_like(v)) for k, v in phi.items()}

    def p_objective(p, beta, x, mask, idx, old_logp, ref_logp, adv):
        return policy_loss(bc.forward(p, x), mask, idx, old_logp, ref_logp, adv,
                           beta, xp=jnp)

    @jax.jit
    def p_step(p, opt, t, beta, lr, batch):
        (loss, aux), g = jax.value_and_grad(p_objective, has_aux=True)(p, beta, *batch)
        g, norm = clip_grads(g, 0.5)
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

    hl_c, hl_t = hl_table(args.hl_bins, args.hl_sigma)
    hl_c, hl_t = jnp.asarray(hl_c), jnp.asarray(hl_t)

    @jax.jit
    def v_step_hl(q, opt, t, x, ret):
        # rint, not astype: astype TRUNCATES, so a return of -1e-8 would be
        # labelled a LOSS with nothing crashing and nothing printing.
        tgt = hl_t[jnp.rint(ret + 1.0).astype(jnp.int32)]

        def kl(qq):
            # KL, not raw CE: a perfect prediction scores the TARGET'S ENTROPY
            # (1.67 nats on a win row), so raw CE would print `v 1.67` for a
            # flawless critic and read as a regression next to the scalar head's
            # MSE. The entropy is a constant, so the gradient is the CE's.
            lp = _log_softmax(vt.forward(qq, x), jnp)
            return jnp.mean(jnp.sum(tgt * (jnp.log(tgt + 1e-12) - lp), axis=1))
        loss, g = jax.value_and_grad(kl)(q)
        g, _ = clip_grads(g, 1.0)
        q, opt = adam(q, opt, g, t, args.critic_lr)
        return q, opt, loss

    @jax.jit
    def values_of_hl(q, x):
        # MEAN of the predicted categorical, not the argmax: GAE is linear in V.
        # Bounded by [c_0, c_-1] for any logits, garbage included.
        return jax.nn.softmax(vt.forward(q, x), axis=-1) @ hl_c

    if args.value_head == "hlgauss":
        v_step, values_of = v_step_hl, values_of_hl

    @jax.jit
    def ref_logp_of(p, x, mask, idx):
        lp = _log_softmax(jnp.where(mask, bc.forward(p, x), -1e9), jnp)
        return lp[jnp.arange(idx.shape[0]), idx]

    def to_x(a):
        # Upload the f16 and widen ON THE DEVICE. Casting first materialised a
        # second, twice-as-large host array and pushed 35.3 kB a row over PCIe
        # instead of 17.6 -- ~2.8 GB an iteration at 256 games / stage 0
        # (512 columns x ~156 turns x 17.6 kB, and the whole buffer goes up
        # twice: ingest chunks, then the training epoch).
        # f16 -> f32 is exact in numpy and in XLA, so dnp and dlp0 cannot move.
        return jnp.asarray(a).astype(jnp.float32)

    def to_mask(packed):
        # Unpack on the device too: `vecroll.make_step` packs there, so the 497
        # bytes a row are what crosses PCIe in both directions. Big-endian on
        # both sides; `vecroll.selfcheck` asserts jnp/np packbits agree.
        return jnp.unpackbits(jnp.asarray(packed), axis=1)[:, :features.N_ACTIONS] != 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    live = out.with_suffix(".live.npz")
    resume_path = out.with_suffix(".resume.npz")
    # The promotion gate's opponent: this policy's own weights as they were when
    # it entered the current stage. A fixed path whose CONTENTS change, exactly
    # like `live`, so the spec string handed to the pool stays constant.
    stage_ref = out.with_suffix(".stageref.npz")
    publish(theta_init, live)        # exists before the first worker starts
    publish(theta_init, stage_ref)
    specs = [args.opp, f"clone:{stage_ref}"]
    stage_mode = 2

    rng = np.random.default_rng(args.seed)
    start_it, stage, over, resident = 0, args.start_stage, 0, 0
    # ONE predicate for the five sites that used to test `== "gpu"`. Miss the
    # `dnp` one and a new backend runs with the numpy/JAX weight-layout kill
    # silently disarmed -- which is the check that exists because a conv layout
    # bug cost this repo a 200-0 arena result.
    vec_backend = args.backend != "cpu"
    beta, lr_scale = BETA0, 1.0
    t_p = t_v = 0
    warmed, low, rewinds = False, 0, 0
    best_score, best_iter = -1.0, -1
    best_theta = dict(theta_init)
    stage_theta = dict(theta_init)   # what stage_ref currently holds
    base_score = None
    comp_low = 0

    st = load_resume(resume_path) if args.resume else None
    if st is not None:
        a, s = st["arrays"], st["scalars"]
        stored_init = _unflat("init", a)
        for k in keys:
            if not np.array_equal(stored_init[k], theta_init[k]):
                raise SystemExit(f"--resume: {args.init} is not the checkpoint this "
                                 f"run started from ({k} differs). The anchor and "
                                 f"the paired gate would both silently change.")
        theta = {k: jnp.asarray(v) for k, v in _unflat("theta", a).items()}
        phi = {k: jnp.asarray(v) for k, v in _unflat("phi", a).items()}
        # A --value-head mismatch here is a shape error deep inside a jitted
        # v_step, hours after the preemption this resume exists to survive.
        want_v = ((ch, args.hl_bins) if args.value_head == "hlgauss" else (ch,))
        if phi["v_w"].shape != want_v:
            raise SystemExit(f"--resume: this checkpoint's critic head is "
                             f"{tuple(phi['v_w'].shape)} but --value-head "
                             f"{args.value_head} wants {want_v}.")
        opt_p = {k: (jnp.asarray(v), jnp.asarray(_unflat("optp_v", a)[k]))
                 for k, v in _unflat("optp_m", a).items()}
        opt_v = {k: (jnp.asarray(v), jnp.asarray(_unflat("optv_v", a)[k]))
                 for k, v in _unflat("optv_m", a).items()}
        best_theta = _unflat("best", a)
        # Restore the promotion gate's opponent too: without it the gate silently
        # re-anchors to wherever the policy happened to be at the restart, which
        # makes every post-preemption promotion decision incomparable with the
        # ones before it.
        stage_theta = _unflat("stageref", a)
        publish(stage_theta, stage_ref)
        missing = RESUME_SCALARS - set(s)
        if missing:
            raise SystemExit(f"--resume: {resume_path} predates this build, "
                             f"missing scalars {sorted(missing)}. Start fresh.")
        start_it = s["it"] + 1
        stage, over, resident = s["stage"], s["over"], s["resident"]
        beta, lr_scale, t_p, t_v = s["beta"], s["lr_scale"], s["t_p"], s["t_v"]
        warmed, low, rewinds = s["warmed"], s["low"], s["rewinds"]
        best_score, best_iter = s["best_score"], s["best_iter"]
        base_score, comp_low = s["base_score"], s["comp_low"]
        rng.bit_generator.state = s["rng_state"]
        print(f"resumed from {resume_path} at iteration {start_it}, stage {stage}, "
              f"best {best_score:.3f}")
        if args.start_stage:            # the checkpoint wins; say so out loud
            print(f"  --start-stage {args.start_stage} IGNORED: the stage comes "
                  f"from the checkpoint")
    elif args.resume:
        print(f"--resume: no {resume_path}, starting fresh")

    pool = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn"),
        initializer=_init, initargs=(str(live), tuple(specs), args.max_turns))

    def play(jobs, weights):
        """Publish, then dispatch. Every sample comes from exactly one snapshot,
        so the stored logp IS the behaviour density and no staleness filter is
        needed -- which is what makes dlp0 a meaningful desync alarm."""
        publish(weights, live)
        res = []
        for r in pool.map(_rollout, jobs, chunksize=1):
            if r:
                res.extend(r)
        return res

    # --- the vectorised training rollout, behind the SAME boundary as `play`.
    #     `play` keeps all four of its other callers (comp-eval, stage-eval and
    #     the two halves of the final paired gate): those run numpy heuristic
    #     opponents through `arena.agents` and cannot move onto the device, and
    #     they are the ground truth the gpu path has to be measured against.
    vec = {"stage": None, "pool": None, "step": None, "scan": None}

    def play_train(it: int, stage: int):
        from learn import vecroll
        from tools import pools

        if vec["stage"] != stage:
            host = pools.load(args.pool_dir, stage)
            vec.update(stage=stage, pool=vecroll.device_pool(host))
            print(f"  pool: stage {stage}, {len(host['dist'])} boards, "
                  f"mean dist {host['dist'].mean():.1f}", flush=True)
        if vec["step"] is None:
            vec["step"] = vecroll.make_step(bc.forward)
            vec["scan"] = vecroll.make_scan(vec["step"], args.scan_chunk)
        # Every env starts on a DIFFERENT board on every iteration. The starter
        # kit's own pool cannot manage that: `pool_idx` starts at 0 in every
        # environment, so episode k of every env is the same board.
        npool = len(vec["pool"]["dist"])
        idx = (it * args.games + np.arange(args.games)) % npool
        key = jax.random.fold_in(jax.random.PRNGKey(args.seed), it)
        if args.backend == "scan":
            return vecroll.rollout_scan(vec["scan"], vec["pool"], idx, theta, key,
                                        args.max_turns, args.scan_chunk)
        return vecroll.rollout(vec["step"], vec["pool"], idx, theta, key,
                               args.max_turns)

    def rewind(why: str) -> bool:
        """Revert to the best checkpoint, halve lr, double beta. True = stop.

        One definition for two triggers (comp-eval at stage >= 3, stage-eval at
        every stage). Zeroing the Adam moments is the part that is easy to forget
        and impossible to notice: keeping them re-applies the very step that
        caused the collapse.
        """
        nonlocal rewinds, theta, opt_p, lr_scale, beta
        rewinds += 1
        print(f"  COLLAPSE {why}; rewinding to iter {best_iter}, halving lr "
              f"(rewind {rewinds}/{MAX_REWINDS})", flush=True)
        if rewinds > MAX_REWINDS:
            print("  giving up: the schedule is wrong, not unlucky. "
                  "Lower --lr or raise --warm-evar.", flush=True)
            return True
        theta = {k: jnp.asarray(v) for k, v in best_theta.items()}
        opt_p = {k: (jnp.zeros_like(v), jnp.zeros_like(v))
                 for k, v in theta.items()}
        lr_scale *= 0.5
        beta = min(beta * 2.0, BETA_MAX)
        return False

    print(f"opponent: comp-eval {specs[0]} (progress), "
          "stage-eval this policy's own stage-entry weights (promotion only)")
    if args.probe:
        print(f"PROBE: stage 0 only, {args.iters} iterations, no promotion, no "
              f"gate. The question is whether evar reaches {args.warm_evar:.2f} "
              "at distance 2-6.", flush=True)
    started = time.time()
    evar_max = float("-inf")

    for it in range(start_it, args.iters):
        t0 = time.time()
        dmin, dmax, _ = STAGES[stage]
        snapshot = {k: np.asarray(v) for k, v in theta.items()}
        results = (play_train(it, stage) if vec_backend
                   else play(build_jobs(it, args.games, stage), snapshot))
        # WHERE THE ITERATION GOES, printed every iteration, because two
        # speedup attempts were designed against a guess about this and both
        # under-delivered. Four phases, and they are the four terms of the
        # argument: `roll` is the rollout and its device->host copy, `pack` is
        # the host memcpy that gathers the columns into one buffer, `ing` is the
        # no-grad value/anchor pass (plus GAE), `trn` is dnp and the update.
        # Honest despite JAX's async dispatch WITHOUT any added sync: every one
        # of these boundaries already blocks -- the rollout ends in np.asarray
        # per chunk, the ingest loop in np.asarray per chunk, the update in
        # float(vl) per minibatch. Do not add a block_until_ready at a phase
        # BOUNDARY; if one of those syncs is ever removed, this split silently
        # becomes a lie. (`vecroll._to_host` does block, inside `roll` and not at
        # a boundary, to split `d2h` off from compute -- and it waits for a
        # computation `np.asarray` waited for anyway, so it moves no boundary.)
        t_roll = time.time()
        # ...and inside `roll`, the one term the hypothesis is actually about.
        # `roll` alone cannot decide anything: it is env transition + observation
        # + encode + forward + copy, and only the last term is what a resident
        # buffer removes. Imported here rather than at module scope because
        # vecroll imports THIS module (see --scan-chunk).
        d2h = ""
        if vec_backend:
            from learn import vecroll
            d2h = f"/d2h {vecroll.D2H:.1f}"
        resident += 1
        if not results:
            print(f"iter {it:5d}  no usable games", flush=True)
            continue

        # Preallocate and pop, do NOT concatenate: `x` is 10.6 kB a sample and a
        # concatenate holds the sources and the destination at once. At stage 5
        # with 256 games that is 2.1 GB doubled to 4.2 GB, and 13 GB if games run
        # to the turn limit -- on the box that is also hosting 60 workers. The
        # other three are 24x smaller and not worth the same care.
        n = sum(len(r["idx"]) for r in results)
        xs = np.empty((n, features.C, features.PAD, features.PAD), np.float16)
        k = 0
        for r in results:
            m = len(r["idx"])
            xs[k:k + m] = r.pop("x")
            k += m
        idxs = np.concatenate([r["idx"] for r in results])
        packed = np.concatenate([r["mask"] for r in results])
        old_logp = np.concatenate([r["logp"] for r in results])
        episodes = [(len(r["idx"]), r["z"]) for r in results]
        for r in results:
            r.pop("mask", None)
        # Fixed-size chunks, tail padded by repeating row 0 and sliced off again:
        # a ragged final chunk changes shape every iteration and pays a full XLA
        # recompile each time.
        t_pack = time.time()
        vals = np.empty(n, dtype=np.float32)
        ref_logp = np.empty(n, dtype=np.float32)
        for i in range(0, n, CHUNK):
            m = min(CHUNK, n - i)
            # A slice indexes a VIEW; `np.arange(i, i + CHUNK)` is a fancy index
            # and materialises 8192 x 17.6 kB = 144 MB of host copy per chunk to
            # hand `to_x` exactly the rows a view already points at -- ~1.5 GB an
            # iteration at stage 0. Only the ragged tail needs the fancy index,
            # and only to pad with row 0.
            sel = (slice(i, i + CHUNK) if m == CHUNK else
                   np.concatenate([np.arange(i, n),
                                   np.zeros(CHUNK - m, dtype=np.int64)]))
            xb = to_x(xs[sel])
            vals[i:i + m] = np.asarray(values_of(phi, xb))[:m]
            ref_logp[i:i + m] = np.asarray(ref_logp_of(
                theta_ref, xb, to_mask(packed[sel]), jnp.asarray(idxs[sel])))[:m]

        adv = np.empty(n, dtype=np.float32)
        ret = np.empty(n, dtype=np.float32)
        frac = np.empty(n, dtype=np.float32)   # position within the episode, 0..1
        k = 0
        for length, z in episodes:
            adv[k:k + length] = gae(z, vals[k:k + length])
            ret[k:k + length] = z           # MC return; see netoracle.gae
            frac[k:k + length] = np.arange(length, dtype=np.float32) / max(length - 1, 1)
            k += length
        # SCALE ONLY, not re-centred. Every game contributes one winning and one
        # losing trajectory of equal length, so the RETURNS are exactly zero-sum
        # and the advantage mean is -(sum of V over both seats)/n -- exactly 0
        # only while the critic outputs ~0, and near 0 afterwards to the extent a
        # critic trained on mirrored pairs is antisymmetric across them. NOT
        # "zero by construction". It is still small and still not a win-rate
        # constant, which is the reason not to subtract it: subtracting would
        # hand every state whose raw advantage is ~0 one identical deterministic
        # value, i.e. failure 4's blanket push rebuilt out of its own cure.
        adv = np.clip(adv / (adv.std() + 1e-8), -ADV_CLIP, ADV_CLIP)
        def _evar(m):
            """1 - Var(z - V)/Var(z) over a subset of the buffer.

            POOLED evar IS NOT A QUALITY SCORE. V* is a martingale, so its
            explainable share of Var(z) rises from map-determined at t=0 to 1.0
            at t=T, and the pooled number is mostly a statement about episode
            length. Read `l` (last 10% of plies): the outcome there is nearly
            determined, so the ceiling is near 1 and any shortfall is the critic
            failing to FIT, not information it lacks. `e` (first 10%) is the
            other end: e << l with l high is the martingale, not a bad critic.
            """
            r, v = ret[m], vals[m]
            rv = float(r.var())
            return float(1.0 - ((r - v).var() / rv)) if rv > 1e-9 else float("nan")
        evar = _evar(slice(None))
        evar_e, evar_l = _evar(frac < 0.1), _evar(frac >= 0.9)
        if evar == evar:
            evar_max = max(evar_max, evar)
        t_ing = time.time()      # GAE is a host python loop; it counts as ingest

        # Warm up on the CRITIC'S SCORE, not on a fixed iteration count -- the
        # 1500-iteration run released the policy after 3 iterations with evar
        # still at +0.02 and eval fell 0.175 -> 0.000 by iteration 160. And the
        # gate is STANDING, not a latch: in self-play the critic chases the
        # outcome of a game between two policies that are both moving, so evar
        # can be healthy and then decay, and a permanent latch would not notice.
        low = 0 if evar >= args.warm_evar else low + 1
        if not warmed:
            warmed = it >= WARMUP and evar >= args.warm_evar
        elif low >= REFREEZE_AFTER:
            warmed, low = False, 0
            print(f"  RE-FREEZE evar under {args.warm_evar} for {REFREEZE_AFTER} "
                  f"iterations; policy frozen until the critic recovers", flush=True)
        do_policy = warmed
        # KILL 2, the hypothesis's own falsifier, in code rather than in prose.
        # Without it a critic that never fits leaves the policy frozen for ever:
        # `warmed` has no timeout, `promote` blocks on evar, and the run prints
        # `[critic warmup]` until morning and answers nothing. Fires after this
        # iteration's log line, so the kill has its own context above it.
        # Armed at EVERY stage, not just stage 0. The first curriculum run froze
        # at stage 2 -- games there run 333 turns against stage 0's 156, evar
        # fell below the line, and with the kill gated on `stage == 0` nothing
        # stopped it: `[critic warmup 234/20]`, pg +0.0000, for 250 iterations
        # until the run was killed by hand. A frozen policy at any stage is the
        # same wasted night.
        kill2 = (not warmed and it > WARMUP * 10 and low >= FROZEN_KILL)

        # `dlp0` compares the worker's NUMPY logp against the parent's JAX one
        # and is the top kill criterion. Under --backend gpu both sides are the
        # same JAX forward, so it becomes structurally 0.0 while still printing
        # -- the check would be gone with nothing announcing it. `dnp` replaces
        # it: 32 stored rows through bot/policy/net.py's own numpy forward
        # against the trainer's, on the SAME f16 x the gradient sees. Strictly
        # stronger -- no sampling noise, and it exercises the submission's code
        # path rather than the worker's. 32 x 0.4 ms = 13 ms an iteration.
        dnp = float("nan")
        if vec_backend and n >= DNP_ROWS:
            sel = rng.choice(n, DNP_ROWS, replace=False)
            ref = np.asarray(bc.forward(theta, to_x(xs[sel])))
            dnp = float(np.abs(numpy_forward(snapshot, xs[sel]) - ref).max())
            scale = max(float(np.abs(ref).max()), 1.0)
            if dnp > 1e-4 * scale:
                print(f"  KILL dnp {dnp:.2e} over {1e-4 * scale:.2e}: the numpy "
                      "forward in the submission and the JAX forward in the "
                      "trainer disagree. Every gradient so far was computed "
                      "against a policy that did not generate the data.",
                      flush=True)
                break

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
                    theta, opt_p, _, aux, norm = p_step(theta, opt_p, t_p, beta,
                                                        args.lr * lr_scale, batch)
                    d_, ku, ka, en, pgv = (float(x) for x in aux)
                    if epoch == 0 and s == 0:
                        dlp0 = d_
                    kl_sum += ku
                    kl_u, kl_last, kl_a, ent, pg, gn = (
                        kl_sum / nb, ku, ka, en, pgv, float(norm))
                t_v += 1
                phi, opt_v, vl = v_step(phi, opt_v, t_v, xb, jnp.asarray(ret[sel]))
                vloss = float(vl)
                # Tested on the CURRENT minibatch, not the running mean: each
                # minibatch moves theta further from the fixed theta_old, so the
                # per-update KL ramps and a running mean is half of it -- the
                # stop fired at 0.04 while claiming to bound the step at 0.02.
                if do_policy and kl_last > KL_STOP:
                    stop = True
                    break
            if stop:
                break

        if do_policy:
            if kl_a > ANCHOR_HI:
                beta = min(beta * 2.0, BETA_MAX)
            elif kl_a < ANCHOR_LO:
                beta = max(beta / 2.0, BETA_MIN)

        # W/D/L over GAMES, counted from the seat-0 trajectories only: every game
        # contributes both a win and a loss to the buffer, so counting the
        # trajectories would print a 50/50 split whatever happened.
        g0 = [r for r in results if r["seat"] == 0]
        w = sum(r["z"] > 0 for r in g0)
        d = sum(r["z"] == 0 for r in g0)
        turns = sum(r["turns"] for r in g0) / len(g0)
        dist = sum(r["dist"] for r in g0) / len(g0)
        hi = "+" if dmax is None else f"-{dmax}"
        # Batch utilisation: the vectorised backend runs every column until the
        # LONGEST game finishes, so this is the fraction of stepped ticks that
        # became samples. Meaningless on cpu (each worker stops at its own
        # game's end), printed only where it can be acted on.
        util = (f"  util {turns / max(r['turns'] for r in g0):.2f}"
                if vec_backend else "")
        # Builds per game, both seats. The action space gained a build slot per
        # cell so the policy COULD build; whether it ever does is a different
        # question and nothing else in this run answers it. A run that sits at
        # bld 0.0 forever has an action it never took -- either builds are wrong
        # here or the curriculum band cannot afford them (stage 0 reaches 35 army
        # around turn 68 but has no rear to build in), and either way the 3529 ->
        # 3970 change bought nothing at that stage.
        builds = int((idxs % features.PER_CELL == features.BUILD_OFFSET).sum())
        # The freeze counter is in the tag, not just the fact of it: a frozen
        # policy is a legitimate state for a few iterations and a dead run after
        # FROZEN_KILL, and the line has to say which one you are looking at.
        tag = f"  [critic warmup {low}/{FROZEN_KILL}]" if not do_policy else ""
        print(f"iter {it:5d}  stage {stage} ({dmin}{hi})  games {len(g0)}  "
              f"W/D/L {w}/{d}/{len(g0) - w - d}  samp {n // 1000:3d}k  "
              f"turns {turns:.0f}  dist {dist:.1f}  bld {builds / len(g0):.2f}"
              f"{util}{tag}\n"
              f"            pg {pg:+.4f}  v {vloss:.3f}  "
              f"evar {evar:+.2f} (e{evar_e:+.2f} l{evar_l:+.2f})  "
              f"{'dnp' if vec_backend else 'dlp0'} "
              f"{dnp if vec_backend else dlp0:.1e}  "
              f"klU {kl_u:.3f}/{kl_last:.3f}  klA {kl_a:.3f}  "
              f"beta {beta:.3f}  ent {ent:.2f}  gn {gn:.2f}  mb {nb}  "
              f"{time.time() - t0:.1f}s (roll {t_roll - t0:.1f}"
              f"{d2h} "
              f"pack {t_pack - t_roll:.1f} ing {t_ing - t_pack:.1f} "
              f"trn {time.time() - t_ing:.1f})", flush=True)

        if kill2:
            print(f"\n  KILL 2: evar under {args.warm_evar} for {low} consecutive "
                  f"iterations at stage {stage} (best seen {evar_max:+.3f}), so the "
                  f"policy is frozen and nothing downstream can move.", flush=True)
            # The same symptom means two different things and only stage 0
            # falsifies the hypothesis the curriculum was built on.
            if stage == 0:
                print(f"  At distance {dmin}{hi} the outcome is not predictable "
                      "from the state, so reward density was NEVER the problem "
                      "and no curriculum fixes it. Write it down in "
                      "docs/ml-log.md.", flush=True)
            else:
                print(f"  Stage 0 worked, so the critic fits SHORT games and not "
                      f"{turns:.0f}-turn ones. That is a critic problem, not a "
                      "refutation of the curriculum: lower --warm-evar, raise "
                      "--stage-cap so each stage consolidates, or use a "
                      "distributional value head.", flush=True)
            break

        # --- comp-eval: the ONLY progress number. Competition boards at every
        #     stage, so it is comparable across the run and against ml-log.md.
        if it % args.comp_eval_every == 0 or (it == args.iters - 1 and not args.probe):
            cur = {k: np.asarray(v) for k, v in theta.items()}
            # Round UP, so the forced final eval at `iters - 1` gets its own
            # block. Flooring gave 1199 // 50 == 1150 // 50 == 23, i.e. the run's
            # last progress number and its last checkpoint decision were made on
            # the same 200 boards used 49 iterations earlier -- correlated with
            # that result rather than independent of it. seed_span budgets +2
            # blocks, so this still cannot reach SP_GATE_SEED0.
            blk = it // args.comp_eval_every + (1 if it % args.comp_eval_every else 0)
            jobs = eval_jobs(args.comp_eval_games, rules.MIN_GENERALS_DISTANCE,
                             None, SP_COMP_SEED0 + blk * (args.comp_eval_games // 2))
            res = play(jobs, cur)
            score, _ = tally(res, specs)
            # stderr over what was PLAYED: `_rollout` returns None when the
            # opponent faults out, and quoting the requested count would both
            # understate the interval and tighten the guard below.
            se = stderr(len(res))
            mark = ""
            if score > best_score:
                best_score, best_iter, best_theta = score, it, cur
                publish(cur, out.with_suffix(".best.npz"))
                mark = "  <- kept"
            if base_score is None:
                base_score = score
            # 2 SE of the DIFFERENCE of two independent estimates, not 2 SE of
            # one. base_score is itself a 400-game number with the same error, so
            # `2 * se` was a 1.41 sigma test sold as 2 sigma: P = 0.079 per eval,
            # ~2 spurious rewinds in a 1200-iteration run, each one permanently
            # halving lr.
            band = 2.0 * se * 2 ** 0.5
            print(f"comp-eval {it:5d}  {score:.3f} +-{se:.3f} "
                  f"({len(res)}/{len(jobs)} games, dist 17+)  "
                  f"base {base_score:.3f} kill<{base_score - band:.3f}{mark}",
                  flush=True)

            # Collapse guard, ARMED ONLY AT STAGE >= 3. Below that the policy is
            # training at distance 4-13 and this number is measured at 17+: a
            # fall is the transfer gap the curriculum is buying time against, not
            # a collapse, and rewinding on it reverts to the iteration-0 clone
            # and halves lr for the rest of the run. In-distribution collapse has
            # its own detector on the stage-eval below, which is where it
            # belongs. KILL 6 was already gated this way; the rewind was not.
            if stage >= 3 and score < base_score - band:
                comp_low += 1
                if comp_low >= 2:
                    print(f"  KILL 6: two comp-evals below {base_score - band:.3f} "
                          f"at stage {stage}. The curriculum taught something that "
                          "does not transfer to distance 17. Stopping.", flush=True)
                    break
                if rewind(f"comp-eval {score:.3f} below base {base_score:.3f} "
                          f"by more than 2 se of the difference"):
                    break
            else:
                comp_low = 0
                if score < base_score - band:
                    print(f"  (below the baseline band, but stage {stage} trains at "
                          f"distance {dmin}{hi} and this is measured at 17+ -- "
                          "expected, not actionable)", flush=True)

        # --- stage-eval: promotion and NOTHING else
        if it % args.stage_eval_every == 0:
            off = (it // args.stage_eval_every) * (args.stage_eval_games // 2)
            jobs = eval_jobs(args.stage_eval_games, dmin, dmax,
                             SP_STAGE_SEED0 + off, stage_mode)
            res = play(jobs, {k: np.asarray(v) for k, v in theta.items()})
            wr, _ = tally(res, specs)
            forced = resident >= args.stage_cap
            new_stage, over = promote(stage, wr, evar, over, resident, args.stage_cap)
            note = (f"  -> STAGE {new_stage}{' (forced)' if forced else ''}"
                    if new_stage != stage else "")
            print(f"stage-eval {it:5d}  {wr:.3f} +-{stderr(len(res)):.3f} "
                  f"({len(res)}/{len(jobs)} games, dist {dmin}{hi}, vs stage-entry "
                  f"self)  {over}/{PROMOTE_OVER} over "
                  f"{STAGES[min(stage + 1, len(STAGES) - 1)][2]:.2f}{note}", flush=True)

            # THE IN-DISTRIBUTION COLLAPSE DETECTOR. 0.500 is a fixed point here,
            # not an estimate -- two argmax copies of the same weights play the
            # same deterministic game whichever way they are seated, so an
            # unchanged policy scores exactly 0.500 and STAGE_COLLAPSE is 3 x
            # stderr(200) below it. Unlike comp-eval this is measured on the
            # boards the policy is actually training on, so it is the number the
            # rewind is entitled to act on.
            if wr < STAGE_COLLAPSE and rewind(
                    f"stage-eval {wr:.3f} vs this policy's own stage-entry "
                    f"weights, under {STAGE_COLLAPSE:.2f} at distance {dmin}{hi}"):
                break

            if new_stage != stage:
                stage, resident, comp_low = new_stage, 0, 0
                # The gate re-anchors to HERE: the question at every stage is
                # "did it improve on the weights it entered this stage with",
                # which reads 0.500 on entry by construction and is invariant to
                # any opponent's miscalibration at any distance. It replaces a
                # gate against v16, whose enemy-general prior is built from
                # `dist >= MIN_GENERALS_DISTANCE` (bot/belief.py) and therefore
                # does not contain the true general at all below stage 4 -- that
                # gate was satisfied by the initialisation, before a single
                # policy step, and would have walked the curriculum to stage 4 by
                # iteration ~160 while measuring nothing but v16's blind spot.
                stage_theta = {k: np.asarray(v) for k, v in theta.items()}
                publish(stage_theta, stage_ref)
                # Nothing else needs resetting: episodes are whole games, so
                # there is no in-flight state on the old distribution. That is
                # the CPU pool's largest practical advantage over a vectorised
                # env, which has to regenerate a map pool, force a JIT retrace
                # and discard every running episode.

            save_resume(resume_path,
                        {**_flat("theta", {k: np.asarray(v) for k, v in theta.items()}),
                         **_flat("phi", {k: np.asarray(v) for k, v in phi.items()}),
                         **_flat("optp_m", {k: np.asarray(v[0]) for k, v in opt_p.items()}),
                         **_flat("optp_v", {k: np.asarray(v[1]) for k, v in opt_p.items()}),
                         **_flat("optv_m", {k: np.asarray(v[0]) for k, v in opt_v.items()}),
                         **_flat("optv_v", {k: np.asarray(v[1]) for k, v in opt_v.items()}),
                         **_flat("best", best_theta),
                         **_flat("stageref", stage_theta),
                         **_flat("init", theta_init)},
                        {"it": it, "stage": stage, "over": over,
                         "resident": resident, "beta": float(beta),
                         "lr_scale": float(lr_scale), "t_p": t_p, "t_v": t_v,
                         "warmed": bool(warmed), "low": low, "rewinds": rewinds,
                         "best_score": float(best_score), "best_iter": best_iter,
                         "base_score": None if base_score is None else float(base_score),
                         "comp_low": comp_low,
                         "rng_state": rng.bit_generator.state})

    if args.probe:
        pool.shutdown()
        ok = evar_max >= args.warm_evar
        print(f"\nPROBE over {args.iters} iterations at stage 0 "
              f"({STAGES[0][0]}-{STAGES[0][1]}): best evar {evar_max:+.3f}, "
              f"needs {args.warm_evar:+.3f}")
        print("VERDICT: " + ("the outcome IS predictable from the state at short "
                             "generals distance, so the critic has something to "
                             "fit and the overnight run is worth its night."
                             if ok else
                             "the critic explains nothing even at distance 2-6. "
                             "Reward density was NEVER the problem -- the "
                             "curriculum cannot fix this and the overnight run "
                             "would answer nothing. Record it in docs/ml-log.md "
                             "and go and look for a different cause."))
        print(f"({time.time() - started:.0f}s)")
        return

    # --- gate: paired against the initialisation on competition-distance boards
    # nothing was selected on. "Did this improve on its own starting point" is
    # the exact question failures 4 and 5 answered no to.
    gate = eval_jobs(args.gate_games, rules.MIN_GENERALS_DISTANCE, None, SP_GATE_SEED0)
    r_a = play(gate, best_theta)
    r_b = play(gate, theta_init)
    score, _ = tally(r_a, specs)
    init_score, _ = tally(r_b, specs)
    pool.shutdown()

    # 2 SE of the DIFFERENCE. Both sides are separate estimates, so sqrt(2) is
    # the independent-sample bound; the two runs share boards, which correlates
    # them positively and makes this conservative rather than wrong. `2 *
    # stderr(n)` was a 1.4 sigma test advertised as 2 sigma.
    margin = 2 * stderr(min(len(r_a), len(r_b))) * 2 ** 0.5
    accepted = bool(score - init_score > margin)
    publish(best_theta, out)
    publish({k: np.asarray(v) for k, v in theta.items()}, out.with_suffix(".last.npz"))
    out.with_suffix(".json").write_text(json.dumps(
        {"accepted": accepted, "score": round(score, 4),
         "init_score": round(init_score, 4), "margin": round(margin, 4),
         "base_score": None if base_score is None else round(base_score, 4),
         "gate_games": len(gate), "stage_reached": stage,
         "best_iter": best_iter, "best_comp_eval": round(best_score, 4),
         "opponent": specs[0]}, indent=2) + "\n")

    print(f"\ngate on {len(gate)} fresh competition-distance games: trained "
          f"{score:.3f} vs init {init_score:.3f}, needs +{margin:.3f} -> "
          f"{'ACCEPTED' if accepted else 'REJECTED'}   (reached stage {stage})")
    print(f"wrote {out}  ({time.time() - started:.0f}s)")
    print("Confirm it before believing it:")
    print(f"  python -m arena.runner --a clone:{out} --b ours:configs/v16.json "
          f"--games 400 --workers {args.workers}")


if __name__ == "__main__":
    main()
