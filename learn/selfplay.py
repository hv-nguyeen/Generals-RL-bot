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
          That is a LATE-GAME gradient-scale fix, where |V| is large; the
          midgame credit assignment this project's collapse is about runs at
          |V| ~ 0 where the factor is ~1.0 and the two losses have near
          identical gradients. Turn it on only if the evar bands say the critic
          fails to beat a board-blind predictor (see READING THE OUTPUT), and
          merge it only on an arena.runner verdict;
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
                pg -0.0132  v 0.214  evar 0.31 (e+0.04 m+0.22 l+0.88
                sc m+0.18 l+0.80)  dlp0 4.1e-04  klU 0.011/0.019
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

`evar` PRINTS FIVE NUMBERS AND NOT ONE OF THEM IS READABLE ON ITS OWN. With
terminal-only reward the critic's target is z at every state, so the optimal
critic V* is a martingale: Var(V*_t) rises from whatever the map alone
determines at t=0 to Var(z) at t=T. Every evar here is therefore capped by a
ceiling set by how fast games resolve and how often a loser is sniped through
fog -- NOT by the critic. Simulated over a bounded win-probability martingale
with z drawn consistently with V*, a PERFECT critic reads a last-decile evar
anywhere from 0.29 to 1.00 across plausible operating points. So no absolute
threshold on any of these fields is a claim, and an earlier version of this
docstring asserting `l >= 0.85` was the stage-eval 0.60 mistake again: a bar
that may be unreachable by construction (see docs/ml-log.md).

    e   first 10% of each episode's plies      m   frac 0.20-0.45, the MIDGAME,
    l   last 10%                                   where the collapse lives

`sc` IS THE ONLY CEILING-FREE READING and it is why the bands are printed at
all. It is the same evar for the same subset from an ordinary least squares on
the EIGHT BROADCAST SCALARS the observation already carries (turn/1200, turn%2,
(turn%50)/50, turn>=800, log1p of both army and both land totals) -- a
predictor that cannot see the board at all. Both face the identical ceiling, so
the ceiling CANCELS and only the difference means anything:

    m/l well above sc   the critic reads the board. A low absolute number there
                        is information the state does not contain, not a fitting
                        failure, and no value loss recovers it.
    m/l at or below sc  the critic is not extracting anything the clock does not
                        already give. THIS is the only reading that justifies
                        touching the value loss, and the midgame `m` gap is the
                        one the credit-assignment hypothesis is about.
    nan                 the buffer had no decided games (Var(z) = 0). At stage 5
                        an all-draw batch does this; it is not a critic result.

The measurement, ~30 iterations at competition distance ON THE POLICY WHOSE
CRITIC IS IN QUESTION. Starting fresh from `--init` fits a critic to
CLONE-vs-CLONE games and answers about a policy nobody is training, so resume --
and resume onto a COPY, because `--out` owns `.resume.npz` and `.stageref.npz`
and the diagnostic would otherwise overwrite the run it is diagnosing:

    cp /local/data/vng205/sp.resume.npz /local/data/vng205/ceil.resume.npz
    python -m learn.selfplay --backend gpu --resume \\
        --out /local/data/vng205/ceil.npz --iters $((N + 30)) \\
        --warm-evar 0.99 --comp-eval-every 999

`--iters` is an ABSOLUTE end, so pass the resumed iteration N plus 30, and drop
`--start-stage`: on a resume the stage comes from the checkpoint and the flag is
ignored. `--warm-evar 0.99` is never satisfied, so the policy re-freezes after
REFREEZE_AFTER iterations and the critic then fits a near-stationary
distribution; KILL 2 ends the run ~FROZEN_KILL iterations later, after printing
everything this needs. `--comp-eval-every 999` skips the 400-game CPU comp-evals
this run has no use for.

Then: `m` at or below `sc` -> `--value-head hlgauss` is the cheapest thing to
try (same command plus that flag). `m` well above `sc` -> the critic already
reads the board and the night belongs to the receptive field instead. Either
way the head merges on an arena.runner verdict, never on evar; and the honest
A/B needs a THIRD arm, scalar head at 5x `--critic-lr`, because CE and
tanh-MSE differ in effective critic learning rate by 0.4x to 12.6x across the
state space and a two-arm test is confounded with a critic-LR sweep. Price it
before starting: that is three matched training runs plus a 400-game gate, i.e.
two GPU-nights minimum, for a mechanism this docstring argues is narrow.

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
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from arena import agents
from bot import features, rules
from bot.memory import TemporalMemory
from bot.policy.net import Net, arch_of, arch_record
from learn.replay import GameBalancedReplay
from learn.league import stderr
from learn.netoracle import (ADV_CLIP, ANCHOR_HI, ANCHOR_LO, BETA0, BETA_MAX,
                             BETA_MIN, CHUNK, KL_STOP, LAM, MAX_REWINDS, WARMUP,
                             _log_softmax, adam, clip_grads, critic_ready, gae,
                             policy_keys, policy_loss, publish, readiness_step,
                             restore_actor_critic, tally)
from sim import engine, mapgen
# ONE definition of what a counterfactual build IS -- which cell it lands on and
# which stratum it belongs to -- shared with the probe that gates this run. If
# the two drifted, `tools/cfprobe`'s verdict would describe a different
# intervention from the one the trainer performs, and the gate would be theatre.
from tools import cfprobe

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
    "comp_low", "lam", "augment", "value_hidden", "rng_state"})


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
    shrink is a UNIFORM SCALE -- row 1 recovers 0 exactly and mixture means are
    linear, so the CE optimum is exactly kappa * V*(s) for the one constant
    kappa = row 2 @ centres (~1 - sigma*sqrt(2/pi), i.e. set by ABSOLUTE sigma,
    not by sigma/w).

    AND GAE IS NOT INVARIANT TO IT, which an earlier version of this docstring
    claimed. The terminal delta is z - V_{T-1} and z is NOT scaled by kappa, so
    adv_t(kappa V) = kappa * adv_t(V) + (1 - kappa) * lam^(T-1-t) * z. The
    kappa factor does wash out of the std-normalisation; the additive term does
    not, because normalisation is a scale and not a shift, and it is signed
    toward the winner over the last ~20 plies of every episode (measured: the
    terminal advantage is 1.33x the scalar head's at kappa = 0.9677). So
    `values_of_hl` DIVIDES the recovered mean by kappa, which restores
    adv(V_hl) == adv(V_scalar) to f32 round-off and makes evar directly
    comparable between the two heads. `selfcheck` pins all of that.
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
    cg = (np.asarray(params["context_global"], np.float32)
          if arch["context"] else None)
    cr = (np.asarray(params["context_region"], np.float32)
          if arch["context"] else None)
    strategy = ({k: np.asarray(params[k], np.float32)
                 for k in npnet.strategy_keys()}
                if arch.get("strategy", False) else None)
    out = []
    for row in np.asarray(x, dtype=np.float32):
        h = npnet._trunk(row, layers, arch["residual"])
        h = npnet._context_mix(h, row[features.VALID], cg, cr)
        h = npnet._strategy_mix(h, row[features.VALID], strategy)
        move = npnet._conv3x3(h, hw, hb)
        pass_logit = float(pw @ h.mean(axis=(1, 2)) + pb)
        out.append(np.concatenate(
            [np.transpose(move, (1, 2, 0)).reshape(-1), [pass_logit]]))
    return np.stack(out)


def _pack(buf: list, winner: int, turns: int, dist: int,
          cf: dict | None = None) -> list[dict] | None:
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

    A counterfactual payload rides as EXTRA KEYS on `out[0]`; the list is always
    length 2. Returning the fork as a third element would pass the ingest block's
    key check silently, land in `episodes`, be handed a GAE advantage and be
    TRAINED ON as if it were on-policy -- breaking the zero-sum pairing this
    function's None-return exists to protect, and inflating `bld`/`bldA`, the
    very metric the experiment is read by. It would also trip `selfcheck`'s
    `len(got) == 2`, which is the intended tripwire.
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
    if cf:
        out[0].update(cf)
    return out


# --------------------------------------------------------------------------
def build_jobs(it: int, games: int, stage: int, replay: float = 0.0,
               restart_frac: float = 0.0, restart_min: int = 80,
               restart_max: int = 180) -> list[tuple]:
    """Self-play training jobs. ONE job = one board = one game = TWO trajectories.

    There is no seat loop and no opponent index. Both seats are the same
    network on the same board, so seat bias cancels inside the game rather than
    across a pair of them, and the batch is exactly zero-sum.

    `replay` is the share of boards drawn from an ALREADY-CLEARED stage instead
    of the current one. Promotion is otherwise a hard switch -- enter stage 4 and
    every board is 11-17 forever, stage 3 never appears again -- and after a
    FORCED promotion that is the worst case: the policy is dropped into
    positions it loses whatever it does, terminal reward carries no gradient
    there, and it forgets the distances it had. sp8 is the evidence, comp-eval
    0.820 at iteration 200 and falling through both of its forced promotions.

    Mixing does not slow the curriculum down: the current stage still gets
    `1 - replay` of every batch and the promotion gate is unchanged. It only
    stops the earlier distances from vanishing.

    Board seeds are untouched by `replay` -- the same `it` and `b` give the same
    seed, only the distance band moves -- so the seed-block assertions in
    `selfcheck` hold and two runs at different `replay` remain comparable.
    """
    base = SP_TRAIN_SEED0 + it * games
    restart_rng = np.random.default_rng([SP_TRAIN_SEED0 + it, 97])

    def job(b, dmin, dmax):
        base_job = (base + b, dmin, dmax, 0, 0)
        if restart_frac <= 0 or restart_rng.random() >= restart_frac:
            return base_job
        return (*base_job, int(restart_rng.integers(restart_min, restart_max + 1)))

    if stage == 0 or replay <= 0.0:
        dmin, dmax, _ = STAGES[stage]
        return [job(b, dmin, dmax) for b in range(games)]

    rng = np.random.default_rng(SP_TRAIN_SEED0 + it)
    picks = np.where(rng.random(games) < replay,
                     rng.integers(0, stage, size=games), stage)
    jobs = []
    for b, s in enumerate(picks):
        dmin, dmax, _ = STAGES[int(s)]
        jobs.append(job(b, dmin, dmax))
    return jobs


def _short(spec: str) -> str:
    """`ours:configs/v16.json` -> `v16`, for the per-opponent breakdown.

    Only a path-shaped argument is shortened: `hunter:2` is a garrison setting,
    not a file, and labelling that column `2` would be worse than not shortening.

    The KIND is kept for anything but `clone:`, because two opponents built from
    the same weights differ only in their wrapper -- `clone:sp9-i500` and
    `snipe:sp9-i500` both shortened to `sp9-i500` and the breakdown gave no way
    to tell which column was the sniper.
    """
    kind, _, arg = spec.partition(":")
    if "/" not in arg:
        return spec
    stem = Path(arg).stem
    return stem if kind == "clone" else f"{kind}:{stem}"


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


def _init(live: str, specs: tuple, max_turns: int, cf_frac: float = 0.0) -> None:
    _CTX.update(live=live, specs=specs, max_turns=max_turns, cf_frac=cf_frac)


def _act(net: Net, obs, rng, memory: TemporalMemory | None = None):
    """(index, mask, log prob, probs) from the masked softmax. `rng` None means
    argmax.

    `probs` is returned so a counterfactual fork can pick its build cell out of
    the SAME forward the action came from. Calling `net.logits` a second time
    would be both wasted work and a second chance for the two to disagree.
    """
    if memory is not None:
        memory.update(obs)
    mask = features.legal_mask(obs)        # PASS is always legal: never empty
    lg = np.where(mask, net.logits(obs, memory=memory), -np.inf)
    lg -= lg.max()
    p = np.exp(lg)
    p /= p.sum()
    idx = int(np.argmax(p)) if rng is None else int(rng.choice(len(p), p=p))
    return idx, mask, float(np.log(p[idx])), p


def _continue(st, net: Net, seat: int, turn0: int, max_turns: int, forced, rng,
              memories: list[TemporalMemory]):
    """Play `st` to terminal from turn `turn0`.

    Returns (z for `seat`, final state, `seat`'s encoded observation one step
    after the fork). That third value is the successor state the critic term
    trains on; taking it here rather than replaying the fork turn a third time
    keeps one definition of "the board the forced build produced".

    `forced` overrides `seat`'s action on turn `turn0` only, and it is applied
    AFTER the sample rather than instead of it, so both branches of a fork
    consume identical draws on the fork turn. That is the whole of what common
    random numbers can buy here: from t+1 the branches occupy different states
    and drift apart on their own.
    """
    x1 = None
    memories = [m.copy() for m in memories]
    for turn in range(turn0, max_turns + 1):
        acts = [None, None]
        for s in (0, 1):
            idx, _, _, _ = _act(net, engine.observe(st, s), rng, memories[s])
            acts[s] = features.index_to_action(idx)
            if turn == turn0 and s == seat and forced is not None:
                acts[s] = forced
        over = engine.step(st, acts[0], acts[1])
        if turn == turn0:
            obs1 = engine.observe(st, seat)
            memories[seat].update(obs1)
            x1 = features.encode(obs1, memories[seat]).astype(np.float16)
        if over:
            break
    return outcome(st.winner, seat), st, x1


def _fork(seed: int, res: list, pick, net: Net, max_turns: int) -> dict | None:
    """Run the counterfactual pair for one game and return its training payload.

    `res` is the two per-stratum reservoirs filled during the parent game. A fair
    coin picks between them when both are populated -- see `tools/cfprobe` for
    why neither stratum alone is the right estimand.

    The control is a RE-ROLLED continuation from the fork snapshot, not the
    parent episode: both branches draw from one seed, so the pair is coupled at
    the fork instead of being two independent games that happen to share a
    prefix. The parent episode is untouched and enters PPO exactly as before.
    """
    have = [k for k in (0, 1) if res[k] is not None]
    if not have:
        return None
    k = have[0] if len(have) == 1 else int(pick.random() < 0.5)
    snap, memory_snap, x_fork, mask_fork, turn0, seat, r, c = res[k]

    forced = (rules.BUILD, r, c, 0, 0)
    # `x1` is the board the forced build produced, seen by the seat that paid for
    # it. It is the ONLY state the critic term trains on, and the one the critic
    # has essentially never seen -- at bld 0.2/game post-build states are ~0.1%
    # of the buffer. Keeping it to s' rather than the whole forced branch is
    # deliberate: five runs have died of a critic whose evar collapsed, and
    # off-distribution states are the direct way to cause that.
    zb, stb, x1 = _continue(snap.copy(), net, seat, turn0, max_turns, forced,
                            np.random.default_rng([seed, turn0]), memory_snap)
    zc, _, _ = _continue(snap.copy(), net, seat, turn0, max_turns, None,
                         np.random.default_rng([seed, turn0]), memory_snap)
    return {
        "cf_x_fork": x_fork,
        "cf_x": x1,
        "cf_mask": np.packbits(mask_fork[None], axis=1),
        "cf_idx": np.int32(((r * features.PAD + c) * features.PER_CELL)
                           + features.BUILD_OFFSET),
        # Halved to [-1, +1] so a weight of 0.05 means what it says next to a
        # log_pi of order 1. NOT centred: mean-centring would push half of all
        # builds up even where builds are uniformly bad, which is exactly the
        # property `tools/cfprobe` exists to gate.
        "cf_delta": np.float32((zb - zc) / 2.0),
        "cf_z": np.float32(zb),
        "cf_stratum": np.int8(k),
        "cf_turn": np.int32(turn0),
        "cf_captured": np.int8(bool(stb.own[1 - seat][r, c])),
    }


def _rollout(job):
    """One complete game. job = (seed, dmin, dmax, mode, seat).

    mode 0    -- self-play training: BOTH seats are this same network, both
                 sample, both trajectories are returned. 256 games therefore
                 yield 512 trajectories, twice `netoracle`'s samples per game.
    mode >= 1 -- measurement: the net plays `seat` with argmax against
                 `specs[mode - 1]`, and the result is tagged `opp = mode - 1`
                 so `netoracle.tally` scores it per instrument.
    """
    if len(job) == 5:
        seed, dmin, dmax, mode, seat = job
        record_from = 1
    elif len(job) == 6:
        seed, dmin, dmax, mode, seat, record_from = job
    else:
        raise ValueError(f"rollout job must have 5 or 6 fields, got {job}")
    grid = mapgen.generate(seed, dmin, dmax)
    dist = mapgen.generals_distance(grid)
    st = engine.from_grid(grid)
    net = Net(_CTX["live"])
    max_turns = _CTX["max_turns"]
    memories = [TemporalMemory(*grid.shape) for _ in range(2)]

    if mode:
        foe = agents.make(_CTX["specs"][mode - 1], 1 - seat, *grid.shape, seed)
        faults, turns = 0, 0
        for turns in range(1, max_turns + 1):
            idx, _, _, _ = _act(net, engine.observe(st, seat), None,
                                 memories[seat])
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
    # Whether this board is forked is a function of its SEED, not of a draw from
    # `rng`: the parent game then plays identically at any --cf-frac, so a
    # treatment and a control arm on the same seed differ in the aux weight and
    # in nothing else. `pick` is a third stream for the same reason -- reservoir
    # coin flips must not shift the game's own action sampling.
    do_cf = (_CTX.get("cf_frac", 0.0) > 0.0
             and np.random.default_rng([seed, 7]).random() < _CTX["cf_frac"])
    pick = np.random.default_rng([seed, 1])
    seen, res = [0, 0], [None, None]
    turns = 0
    for turns in range(1, max_turns + 1):
        acts = [None, None]
        snap = None
        memory_snap = [m.copy() for m in memories] if do_cf else None
        for s in (0, 1):
            obs = engine.observe(st, s)
            idx, mask, lp, probs = _act(net, obs, rng, memories[s])
            xs, ids, masks, lps = buf[s]
            enc = features.encode(obs, memories[s]).astype(np.float16)
            if turns >= record_from:
                xs.append(enc)
                ids.append(idx)
                masks.append(mask)
                lps.append(lp)
            # engine.observe and engine.step are positional by seat. Placing
            # these the wrong way round trains each seat on the other's rewards
            # and looks like slow noise; selfcheck pins both the sign (via
            # `outcome`) and the buffer-to-seat filing (via frame 0).
            acts[s] = features.index_to_action(idx)
            if do_cf:
                cells = cfprobe.build_cells(mask, obs.H, obs.W)
                if len(cells):
                    # One snapshot per turn, shared by both seats: `engine.step`
                    # has not run yet, so seat 1 sees the board seat 0 saw.
                    if snap is None:
                        snap = st.copy()
                    flat = ((cells[:, 0] * features.PAD + cells[:, 1])
                            * features.PER_CELL + features.BUILD_OFFSET)
                    r, c = (int(v) for v in cells[int(np.argmax(probs[flat]))])
                    mine = obs.owner_grid == rules.OWNER_ME
                    structs = mine & ((obs.type_grid == rules.T_GENERAL)
                                      | (obs.type_grid == rules.T_CASTLE))
                    cost = int(rules.build_cost_grid(structs)[r, c])
                    k = 0 if cfprobe.stratum_a(obs, r, c, turns, max_turns,
                                               cost) else 1
                    seen[k] += 1
                    if pick.random() < 1.0 / seen[k]:
                        res[k] = (snap, memory_snap, enc, mask, turns, s, r, c)
        if engine.step(st, acts[0], acts[1]):
            break

    cf = _fork(seed, res, pick, net, max_turns) if do_cf else None
    # Turn limit and mutual capture (engine.step sets winner -1) are both draws.
    return _pack(buf, st.winner, turns, dist, cf)


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
    value_schema = int(z["value_schema"]) if "value_schema" in z.files else 1
    if "__scalars__" not in z.files:
        return None                       # a pre-single-file checkpoint
    return {"arrays": {k: z[k] for k in z.files if k != "__scalars__"},
            "scalars": json.loads(str(z["__scalars__"]))}


def _flat(prefix: str, d: dict) -> dict:
    return {f"{prefix}__{k}": v for k, v in d.items()}


def _unflat(prefix: str, arrays: dict) -> dict:
    p = prefix + "__"
    return {k[len(p):]: v for k, v in arrays.items() if k.startswith(p)}


def load_critic(path: str, arch: dict, blank: dict) -> dict:
    """A trained critic out of a run's `.resume.npz`, checked against `arch`.

    Every run starts its critic from scratch, because `--init` reads `.best.npz`
    and that file carries the POLICY only. Relearning "what does a winning
    position look like" takes ~40 iterations of short games, and on long ones it
    often never converges at all -- the policy stays frozen behind the `evar`
    gate while it tries. That cost three runs on 2026-08-05: sp7 froze at stage 5,
    sp12 stalled at stage 4 at `[critic warmup 20/20]`, and sp9 paid the stage-3
    tax purely to warm one, learning nothing there (`stage-eval 0.495` after 400
    iterations).

    None of that was necessary. `save_resume` has been writing a trained `phi`
    all along and nothing ever read it back into a NEW run. This does.

    It is also what makes `--start-stage 5` viable: the reason a run cannot begin
    at the distance its policy actually plays is the cold critic, not the policy.

    The critic belongs to the policy it was trained on, so a mismatched pair is
    worse than a blank one -- every shape is checked and anything unexpected
    raises here rather than producing quiet nonsense 200 iterations in.
    """
    z = np.load(path)
    value_schema = int(z["value_schema"]) if "value_schema" in z.files else 1
    got = _unflat("phi", {k: z[k] for k in z.files})
    standalone = not got
    if not got:
        # A run's `.resume.npz` stores the critic under `phi__*`; `learn.valuetrain`
        # writes a STANDALONE one with bare keys via `train.save`. Both are
        # critics and both should load — the second is the whole point of
        # `learn/valuedata.py`, whose docstring names our exact problem: "an
        # evaluation function grounded in games where REAL opponents did the
        # punishing. Our local opponents never punish over-commitment."
        #
        # Until `--init-critic` existed there was no way to get such a model into
        # a run, which is presumably why `/local/data/vng205/val` was built and
        # then went unused.
        got = {k: z[k] for k in z.files}
    if not got:
        raise SystemExit(f"{path} is empty")
    if standalone and value_schema >= 2:
        sidecar = Path(path).with_suffix(".json")
        if not sidecar.is_file():
            raise SystemExit(f"schema-{value_schema} critic {path} is missing its "
                             f"validation sidecar {sidecar}")
        meta = json.loads(sidecar.read_text())
        if meta.get("gate_passed") is not True:
            raise SystemExit(f"schema-{value_schema} critic {path} did not pass "
                             "its held-out board-signal and calibration gate")
    try:
        a = arch_of(got)
    except ValueError as e:
        raise SystemExit(
            f"{path} has no critic in it ({e}). Expected either a run's "
            f".resume.npz (critic under phi__*) or a standalone value model "
            f"from `learn.valuetrain` (bare conv*/v_w keys).") from e
    for k in ("layers", "channels", "residual", "context"):
        if a[k] != arch[k]:
            raise SystemExit(
                f"critic in {path} is {a['layers']}x{a['channels']} "
                f"(residual={a['residual']}), this run is {arch['layers']}x"
                f"{arch['channels']} (residual={arch['residual']}). Grow or "
                f"retrain it; a critic for a different trunk is not loadable.")
    critic_dense = bool(a["context"] and np.ndim(got["context_global"]) == 2)
    if critic_dense != bool(arch.get("dense_context", False)):
        raise SystemExit(f"critic in {path} has dense_context={critic_dense}, "
                         f"this run has dense_context="
                         f"{bool(arch.get('dense_context', False))}. Grow or retrain it.")
    critic_strategy = int(a.get("strategy_hidden", 0))
    policy_strategy = int(arch.get("strategy_hidden", 0))
    if critic_strategy != policy_strategy:
        raise SystemExit(f"critic in {path} has strategy_hidden={critic_strategy}, "
                         f"this run has strategy_hidden={policy_strategy}. "
                         "Migrate the critic with tools.grow or retrain it.")
    # `arch_record` keys are METADATA, not parameters, and `arch_of` above has
    # already used them for the cross-check that makes a mislabelled file fail
    # loudly. `tools.grow` writes them into everything it produces, so a migrated
    # critic arrives with three keys the freshly built one does not have, and an
    # exact key match rejected it.
    metadata = set(arch_record(got)) | {"value_schema", "value_pool", "value_hidden"}
    got = {k: v for k, v in got.items() if k not in metadata}
    # Migrate the historical mean-only head into the v2 pooled head. The first
    # block of ``valuetrain.pooled`` is the same global mean, so zero-filling the
    # remaining max/regional weights preserves the function exactly.
    if ("v_w" in got and "v_w" in blank
            and np.shape(got["v_w"]) != np.shape(blank["v_w"])
            and np.ndim(got["v_w"]) == np.ndim(blank["v_w"]) == 1
            and len(blank["v_w"]) > len(got["v_w"])):
        migrated = np.zeros_like(blank["v_w"], dtype=np.float32)
        migrated[:len(got["v_w"])] = got["v_w"]
        got["v_w"] = migrated
    # Add the residual value head without moving the historical function. Its
    # final layer is exactly zero in `valuetrain.init_params`; copying all four
    # fresh parameters therefore preserves every legacy prediction bit-for-bit
    # while making the branch trainable from the next update.
    residual_keys = {"v_res1_w", "v_res1_b", "v_res2_w", "v_res2_b"}
    if not (residual_keys & set(got)) and residual_keys <= set(blank):
        got.update({k: np.asarray(blank[k]) for k in residual_keys})
    # Grow conv0's INPUT channels when the observation width changed (a new input
    # plane was added). Zero-pad the new channels so the critic's value function
    # is preserved bit-for-bit and learns the new plane from the next update --
    # the same migration the policy stem gets in netoracle, and the reason the
    # prior width lives in features.LEGACY_INPUT_CHANNELS. Without this a
    # champion critic one plane behind cannot warm-start a wider-input run, and
    # --oracle net refuses to start without a matching critic.
    if ("conv0_w" in got and "conv0_w" in blank
            and np.shape(got["conv0_w"])[1] != np.shape(blank["conv0_w"])[1]
            and np.shape(got["conv0_w"])[1] in features.LEGACY_INPUT_CHANNELS
            and np.shape(blank["conv0_w"])[1] == features.C):
        old = np.asarray(got["conv0_w"], dtype=np.float32)
        grown = np.zeros_like(blank["conv0_w"], dtype=np.float32)
        grown[:, :old.shape[1]] = old
        got["conv0_w"] = grown
    if set(got) != set(blank):
        missing = sorted(set(blank) - set(got)) or None
        extra = sorted(set(got) - set(blank)) or None
        raise SystemExit(f"critic keys do not match: missing {missing}, extra {extra}")
    for k, v in blank.items():
        if tuple(np.shape(got[k])) != tuple(np.shape(v)):
            raise SystemExit(f"critic {k} is {np.shape(got[k])}, expected {np.shape(v)}")
    got = {k: np.asarray(got[k], np.float32) for k in blank}
    if standalone and value_schema < 2:
        # HALVE THE HEAD. `learn/valuetrain.py` fits the SAME `forward` under
        # `logaddexp(0, l) - y*l`, so its output is a SIGMOID logit: P(win) is
        # sigma(l). This trainer reads that function through `tanh(l)` against
        # targets in {-1, 0, +1}. The correct conversion is
        #     2*P - 1 = 2*sigma(l) - 1 = tanh(l / 2),
        # so a pretrained critic arrives with logits exactly TWICE too large.
        #
        # It is not a small error. At P(win) = 0.84 (`tools/calibrate`'s reading
        # on 2026-08-08) l is 1.658: the value should be 0.68 and `tanh(1.658)`
        # is 0.93 -- saturated, where tanh' is ~0.14 and the critic can barely
        # gradient its way back. This is the most likely mechanism behind
        # 547da6e, "value24 does not transfer": 0.759 BCE on field positions
        # buying NEGATIVE evar on self-play returns. The fit may have been fine
        # and the scaling threw it away.
        #
        # Only the standalone path is rescaled. A run's own `.resume.npz` was
        # trained through this same tanh and is already in these units.
        got["v_w"] = got["v_w"] * 0.5
        got["v_b"] = got["v_b"] * 0.5
    return got


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
    # --- the critic round-trips out of a resume file, and refuses a wrong one
    with tempfile.TemporaryDirectory() as _d:
        _rng = np.random.default_rng(3)
        _phi = {}
        for _i in range(3):
            _cin = features.C if _i == 0 else 8
            _phi[f"conv{_i}_w"] = _rng.normal(0, .2, (8, _cin, 3, 3)).astype(np.float32)
            _phi[f"conv{_i}_b"] = _rng.normal(0, .1, (8,)).astype(np.float32)
        _phi["v_w"] = _rng.normal(0, .2, (8,)).astype(np.float32)
        _phi["v_b"] = np.float32(0.0)
        _p = Path(_d) / "r.npz"
        # WITH the arch record, because `tools.grow` writes one into every file
        # it migrates and an exact key match rejected exactly that.
        np.savez(_p, **_flat("phi", {**_phi, **arch_record(_phi)}), it=0)
        _arch = arch_of(_phi)
        _got = load_critic(str(_p), _arch, _phi)

        # A STANDALONE critic too: `learn.valuetrain` writes bare keys via
        # `train.save`, and that file is the ladder-grounded value model the
        # whole `valuedata` pipeline exists to produce.
        # ...and its HEAD IS HALVED on the way in, because `valuetrain` fits the
        # same `forward` under `logaddexp(0, l) - y*l` -- a SIGMOID logit -- while
        # this trainer reads it through `tanh(l)` against targets in {-1, 0, +1}.
        # 2*sigma(l) - 1 == tanh(l/2), so an unscaled load arrives with logits
        # twice too large: at P(win) 0.84 the value reads 0.93 instead of 0.68,
        # saturated where tanh' is 0.14 and the critic cannot gradient back.
        # That is the likely mechanism behind 547da6e, "value24 does not
        # transfer" -- 0.759 BCE buying NEGATIVE evar. The trunk carries no units
        # and is untouched.
        _b = Path(_d) / "value.npz"
        np.savez(_b, **{**_phi, **arch_record(_phi)})
        _bare = load_critic(str(_b), _arch, _phi)
        for _k in _phi:
            _want = _phi[_k] * 0.5 if _k in ("v_w", "v_b") else _phi[_k]
            assert np.allclose(_bare[_k], _want), f"bare critic {_k}"
        # the identity the rescale rests on, so nobody "simplifies" it back
        _l = np.array([-3.0, -0.5, 0.0, 1.658, 3.0])
        assert np.allclose(2.0 / (1.0 + np.exp(-_l)) - 1.0, np.tanh(_l / 2.0))
        assert set(_got) == set(_phi)
        for _k in _phi:
            assert np.array_equal(_got[_k], _phi[_k]), _k
        # a critic for a different trunk must RAISE, not silently half-load: it
        # belongs to the policy it was trained on and a mismatched pair is worse
        # than a blank one.
        try:
            load_critic(str(_p), {**_arch, "channels": 32}, _phi)
            raise AssertionError("accepted a critic with the wrong width")
        except SystemExit:
            pass
        np.savez(Path(_d) / "nope.npz", it=0)
        try:
            load_critic(str(Path(_d) / "nope.npz"), _arch, _phi)
            raise AssertionError("accepted a file with no critic in it")
        except SystemExit:
            pass

    jt = build_jobs(1199, 256, 0)
    assert len(jt) == 256 and len({j[0] for j in jt}) == 256
    assert max(j[0] for j in jt) < SP_STAGE_SEED0
    assert all(j[3] == 0 for j in jt)

    # stage replay: same board seeds, only the distance band moves, and the
    # current stage still supplies the majority. replay=0 must be bit-identical
    # to every run before it existed.
    assert build_jobs(7, 256, 3, 0.0) == build_jobs(7, 256, 3)
    mixed = build_jobs(7, 256, 3, 0.25)
    assert [j[0] for j in mixed] == [j[0] for j in build_jobs(7, 256, 3)]
    cur = sum(1 for j in mixed if (j[1], j[2]) == STAGES[3][:2])
    assert 0.6 < cur / len(mixed) < 0.9, f"current stage got {cur}/{len(mixed)}"
    assert all((j[1], j[2]) in [s[:2] for s in STAGES[:4]] for j in mixed)
    assert build_jobs(7, 256, 0, 0.25) == build_jobs(7, 256, 0)   # stage 0 has no past
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
    arch = {"layers": 4, "channels": DEFAULT_CHANNELS, "residual": False,
            "context": False}
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
        i, lm, logp, _ = _act(net, obs, None)
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
            assert np.any(r["x"][:, features.BASE_C:] != 0), (
                "self-play rollout zeroed every temporal plane")
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
        fresh_memory = [TemporalMemory(*fresh.armies.shape) for _ in range(2)]
        for r in got:
            obs0 = engine.observe(fresh, r["seat"])
            fresh_memory[r["seat"]].update(obs0)
            want = features.encode(obs0, fresh_memory[r["seat"]]).astype(np.float16)
            assert np.array_equal(r["x"][0], want), r["seat"]

        # --- counterfactual forks. Three properties, all of which fail SILENTLY
        #     in a way the run cannot show you.
        #
        #     1. cf_frac 0 must change nothing. The control arm and every run
        #        that predates this feature depend on it.
        base = _rollout((seed, 2, 6, 0, 0))
        assert [set(r) for r in base] == [set(r) for r in got]
        for a, b in zip(base, got):
            assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["idx"], b["idx"])

        #     2. The payload rides on out[0] and the list stays length 2. A third
        #        element would be trained on as if it were on-policy.
        cf = {"cf_idx": np.int32(5), "cf_delta": np.float32(0.5),
              "cf_z": np.float32(1.0), "cf_stratum": np.int8(0),
              "cf_turn": np.int32(9), "cf_captured": np.int8(0),
              "cf_x": np.zeros((features.C, features.PAD, features.PAD), np.float16),
              "cf_x_fork": np.zeros((features.C, features.PAD, features.PAD), np.float16),
              "cf_mask": np.packbits(np.zeros((1, features.N_ACTIONS), bool), axis=1)}
        stub = [([np.zeros(1)], [0], [np.zeros(features.N_ACTIONS, bool)], [0.0])
                for _ in range(2)]
        packed2 = _pack(stub, 0, 3, 4, cf)
        assert len(packed2) == 2, len(packed2)
        assert "cf_idx" in packed2[0] and "cf_idx" not in packed2[1]
        assert [r["seat"] for r in packed2] == [0, 1]
        # the harvest predicate the trainer uses, and the seat-0 game count it
        # must NOT disturb
        assert len([r for r in packed2 if "cf_idx" in r]) == 1
        assert len([r for r in packed2 if r["seat"] == 0]) == 1

        #     3. A forked game plays the SAME parent game as an unforked one.
        #        The fork must be a read of the policy, not a perturbation of the
        #        data PPO trains on -- otherwise treatment and control differ in
        #        their trajectories too and the pairing is worthless. `do_cf` is
        #        keyed on the seed and the reservoir draws from its own stream
        #        precisely so this holds.
        _init(str(path), (), 3, 1.0)
        forked = _rollout((seed, 2, 6, 0, 0))
        assert forked is not None and len(forked) == 2
        for a, b in zip(forked, base):
            assert np.array_equal(a["x"], b["x"]), "forking moved the parent game"
            assert np.array_equal(a["idx"], b["idx"])
            assert a["z"] == b["z"] and a["turns"] == b["turns"]
        # A 3-turn stub can never afford 35 army, so there is no fork to take and
        # the payload must be ABSENT rather than empty or zero-filled.
        assert "cf_idx" not in forked[0], "a 3-turn game cannot afford a castle"
        _CTX.clear()

        # --- resume round-trips: arrays exact, scalars exact, RNG stream intact
        r = np.random.default_rng(7)
        r.random(3)
        st = {"it": 41, "stage": 2, "over": 1, "resident": 12, "beta": 0.05,
              "lr_scale": 0.5, "t_p": 100, "t_v": 130, "warmed": True, "low": 0,
              "rewinds": 1, "best_score": 0.31, "best_iter": 40,
              "base_score": 0.19, "comp_low": 1, "lam": LAM,
              "augment": False, "value_hidden": 0,
              "rng_state": r.bit_generator.state}
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
    # The BOUNDARY does not round-trip on its own, and by exactly the
    # half-normal mean: the Gaussian at z = +-1 is centred on the edge, so half
    # its mass is renormalised away. `values_of_hl` divides by that constant, so
    # what the trainer sees IS +-1 -- and this is the assert that fails if
    # someone drops the rescale, or "fixes" the shrink by clipping z instead.
    kappa = float(T[2] @ c)
    assert abs(kappa - (1 - 0.04 * math.sqrt(2 / math.pi))) < w
    assert abs(float(T[2] @ c) / kappa - 1.0) < 1e-6
    assert abs(float(T[1] @ c)) < 1e-7                       # draw recovers 0 exactly
    # And the reason the rescale is not cosmetic: GAE's terminal delta is
    # z - V_{T-1} with z UNSCALED, so a kappa-shrunk critic is not a uniform
    # rescale of the advantage and std-normalisation cannot remove the residue.
    V = np.array([0.5, 0.8, 0.9], dtype=np.float32)
    shrunk = gae(1.0, (kappa * V).astype(np.float32))
    assert abs(shrunk[-1] - kappa * gae(1.0, V)[-1]) > 0.02      # measured 0.032
    assert np.abs(gae(1.0, (kappa * V / kappa).astype(np.float32))
                  - gae(1.0, V)).max() < 1e-6
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
    v = np.exp(_log_softmax(rng.normal(size=(7, 128)) * 5, np)) @ c / kappa
    assert np.all(np.abs(v) <= abs(float(c[0])) / kappa) and np.all(np.isfinite(v))
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
    ap.add_argument("--critic-lr", type=float, default=1e-4,
                    help="critic Adam step. 1e-3 memorised the on-policy buffer "
                         "in measured full-distance runs; 1e-4 is the safe "
                         "transfer default")
    ap.add_argument("--critic-replay-games", type=int, default=0,
                    help="historical complete games retained for critic-only, "
                         "game-balanced replay; 0 reproduces on-policy fitting")
    ap.add_argument("--critic-replay-frac", type=float, default=0.5,
                    help="share of each critic minibatch drawn from replay once non-empty")
    ap.add_argument("--lam", type=float, default=LAM,
                    help="GAE TD/Monte-Carlo interpolation. 0.95 is historical; "
                         "1.0 gives z-V(s) and relies least on critic bootstraps. "
                         "The value is part of the resume identity")
    ap.add_argument("--augment", action="store_true",
                    help="one random dihedral transform per PPO minibatch; "
                         "actions/masks are relabelled and old/ref log-probs "
                         "are recomputed under frozen snapshots")
    ap.add_argument("--value-head", choices=("scalar", "hlgauss"), default="scalar",
                    help="scalar is tanh+MSE and is THE DEFAULT. hlgauss is a "
                         "distributional head; with our 3-atom return support it "
                         "is a 3-way softmax with fixed smoothing and adds no "
                         "distributional information, so the ONLY reason to use "
                         "it is that `ret` sits ON the tanh asymptote, where the "
                         "MSE gradient carries a (1 - tanh^2) factor that "
                         "vanishes exactly where the critic is confidently "
                         "wrong. Do not merge it on an evar number: run "
                         "arena.runner on the two trained policies -- and run a "
                         "THIRD arm, scalar at 5x --critic-lr, because CE and "
                         "tanh-MSE differ in effective critic step by 0.4x-12.6x "
                         "across the state space and a two-arm test cannot tell "
                         "the loss function from a critic-LR sweep")
    ap.add_argument("--value-hidden", type=int, default=64,
                    help="scalar critic residual-head width; 0 is the exact "
                         "linear-head control. Ignored by hlgauss")
    ap.add_argument("--hl-bins", type=int, default=128,
                    help="AverageJoe's, lifted whole; not load-bearing, and "
                         "the arithmetic says so: the head is 32x128 = 4k "
                         "parameters against the trunk's ~33M MACs a sample "
                         "(0.013%%) and the (4096, 128) logits+target pair is "
                         "4.2 MB against ~231 MB for ONE trunk activation")
    ap.add_argument("--hl-sigma", type=float, default=0.04,
                    help="sigma/w = 2.56 at 128 bins over [-1, 1]. The RATIO "
                         "governs the smoothing: Farebrother's flat band is "
                         "0.5..2 and the failure modes (one-hot label, no "
                         "ordinal structure, NaN in the density form) are all at "
                         "ratios well under 1, so 2.56 is the safe side. "
                         "ABSOLUTE sigma governs something else -- the boundary "
                         "shrink kappa = 1 - sigma*sqrt(2/pi), 0.968 here and "
                         "0.871 at --hl-bins 32 --hl-sigma 0.16 despite the "
                         "identical ratio -- which values_of_hl divides out")
    ap.add_argument("--warm-evar", type=float, default=PROMOTE_EVAR,
                    help="hold the policy frozen until the critic explains this "
                         "much of the return, and re-freeze if it stops")
    ap.add_argument("--opp", default="ours:configs/v16.json",
                    help="comma-separated opponents for comp-eval, the only "
                         "progress number; they never generate a training "
                         "gradient and no longer decide promotion. Several is "
                         "the point: this project's own first rule is that no "
                         "single fixed opponent measures general strength, and "
                         "for the whole life of the run that is what selected "
                         "every checkpoint. --comp-eval-games is SPLIT across "
                         "them on the same boards, so adding one costs games "
                         "and not wall clock. Default is the single v16 so a "
                         "run without this flag reproduces the old instrument")
    ap.add_argument("--init-critic", default=None, metavar="RESUME.NPZ",
                    help="warm-start the critic from a previous run's "
                         ".resume.npz. --init reads .best.npz, which carries the "
                         "POLICY only, so every run has been relearning 'what "
                         "does a winning position look like' from scratch -- ~40 "
                         "iterations on short games, and often never on long "
                         "ones. That cost three runs on 2026-08-05 and is the "
                         "only reason --start-stage cannot be set to where the "
                         "policy actually plays. The trunk must match; every "
                         "shape is checked")
    ap.add_argument("--stage-eval-every", type=int, default=10)
    ap.add_argument("--stage-eval-games", type=int, default=200)
    ap.add_argument("--comp-eval-every", type=int, default=50)
    ap.add_argument("--comp-eval-games", type=int, default=400)
    ap.add_argument("--gate-games", type=int, default=400)
    ap.add_argument("--stage-cap", type=int, default=200,
                    help="iterations at one stage before promotion is FORCED. "
                         "5 x this must be under --iters or the run never "
                         "reaches the competition distribution it is gated on")
    ap.add_argument("--stage-replay", type=float, default=0.25,
                    help="share of training boards drawn from already-cleared "
                         "stages instead of the current one. Keeps the forced "
                         "promotion (the run must reach competition distance) "
                         "without letting the earlier distances vanish, which "
                         "is what sp8 lost after each of its two forced ones. "
                         "0.0 reproduces every run before 2026-08-05")
    ap.add_argument("--restart-frac", type=float, default=0.0,
                    help="CPU rollout share whose first recorded on-policy state "
                         "comes after an exact policy/memory burn-in; evaluation "
                         "always starts at turn 1")
    ap.add_argument("--restart-min-turn", type=int, default=80)
    ap.add_argument("--restart-max-turn", type=int, default=180)
    ap.add_argument("--frozen-kill", type=int, default=FROZEN_KILL, metavar="N",
                    help="consecutive low-evar iterations that END the run, or 0 "
                         "to never end it. The default exists because a critic "
                         "that never fits leaves the POLICY FROZEN -- do_policy "
                         "is `warmed` -- so the run prints [critic warmup] until "
                         "morning and answers nothing; that cost three runs on "
                         "2026-08-05. Raise it only to ask whether a critic "
                         "recovers given longer, and watch that `pg` is still "
                         "0.0000 the whole time it is climbing")
    ap.add_argument("--dist-tail", type=float, default=0.0,
                    help="the same idea UPWARD: share of training boards drawn "
                         "from the final stage with distance beyond this "
                         "stage's dmax. Stage 4 is (17,24) and the ladder is "
                         "17+ unbounded -- measured over 600 real games, 69%% "
                         "fall in 17-24 and 31%% above, so stage 4 never sees a "
                         "third of what it is evaluated on. 0.31 reconstructs "
                         "the real histogram. Before reaching for that, note "
                         "the two measured dose points: 0.0 produced the only "
                         "checkpoint that ever beat the champion (+35.7), and "
                         "stage 5, which IS this distribution, measured about "
                         "-18. No-op at the final stage, which has no beyond")
    # ---- counterfactual build supervision ---------------------------------
    # See docs/design-counterfactual-build.md. Gated by `tools/cfprobe`: the
    # preference term multiplies an UNCENTRED delta, so if E[delta] <= 0 it
    # pushes the build probability DOWN and the whole thing is a sign flip.
    # Run the probe before setting these to anything but zero.
    ap.add_argument("--cf-frac", type=float, default=0.0, metavar="F",
                    help="share of training games that also run a counterfactual "
                         "build fork: one legal build opportunity is sampled, the "
                         "build is FORCED in one branch and not in the other, and "
                         "both are played to terminal under common random "
                         "numbers. Data only -- forks never enter the PPO buffer. "
                         "0 disables the whole mechanism. Requires --backend cpu")
    ap.add_argument("--cf-pref-weight", type=float, default=0.0, metavar="W",
                    help="weight on -mean(delta * log_pi(build|s_fork)). Set 0 "
                         "with a nonzero --cf-frac for the CONTROL arm: forks are "
                         "still generated and logged, so the arm measures delta "
                         "on a policy the term never touched")
    ap.add_argument("--cf-value-weight", type=float, default=0.0, metavar="W",
                    help="weight on the successor-value term, MSE of the critic "
                         "against the forced branch's outcome at s' only. Drop it "
                         "to 0 if `tools/vprobe` says this critic already prices "
                         "a castle -- that term's premise is that it does not")
    ap.add_argument("--cf-iters", type=int, default=200, metavar="N",
                    help="both weights decay linearly to zero over N iterations, "
                         "after which the run is pure PPO. The term exists to "
                         "break the policy out of a mode, not to stay forever")
    ap.add_argument("--cf-buffer", type=int, default=4096, metavar="ROWS")
    ap.add_argument("--cf-minibatch", type=int, default=256, metavar="ROWS",
                    help="forks resampled WITH REPLACEMENT per gradient step. "
                         "Fixed size so XLA does not retrace on a ragged tail")
    ap.add_argument("--max-turns", type=int, default=rules.TURN_LIMIT)
    ap.add_argument("--probe", type=int, default=0, metavar="ITERS",
                    help="CHEAP HYPOTHESIS TEST, run this before the night. Pins "
                         "--start-stage, disables promotion, runs ITERS "
                         "iterations and prints whether the critic ever reaches "
                         "--warm-evar on that distance distribution. Use stage "
                         "0 to test reward density or stage 5 to test a critic's "
                         "full-competition transfer. ~10 min at 60 workers")
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
    if not 0.0 < args.lam <= 1.0:
        raise SystemExit(f"--lam must be in (0, 1], got {args.lam}")
    if args.critic_replay_games < 0:
        raise SystemExit("--critic-replay-games must be non-negative")
    if not 0.0 <= args.critic_replay_frac < 1.0:
        raise SystemExit("--critic-replay-frac must be in [0, 1)")
    if args.value_hidden < 0:
        raise SystemExit("--value-hidden must be zero or positive")
    effective_value_hidden = args.value_hidden if args.value_head == "scalar" else 0
    if not 0 <= args.start_stage < len(STAGES):
        raise SystemExit(f"--start-stage {args.start_stage} is not a stage "
                         f"(0..{len(STAGES) - 1})")
    if args.start_stage >= 4 and args.stage_replay > 0.0:
        # Not fatal -- a curriculum run that CLIMBS to stage 4 wants replay, and
        # that is what the 0.25 default is for. But a run that STARTS at 4 or
        # above is pinned at competition distance on purpose, and replay feeds it
        # short boards where castles correctly do not pay. Eight runs unlearned
        # castle-building that way before anyone noticed, and castle rate tracks
        # Elo across every measured checkpoint.
        print(f"WARNING --start-stage {args.start_stage} with --stage-replay "
              f"{args.stage_replay}: {args.stage_replay:.0%} of every batch will "
              f"be SHORT boards from stages below {args.start_stage}. The only "
              f"checkpoint that ever beat the champion used --stage-replay 0. "
              f"Pass it explicitly if this is deliberate.", flush=True)
    if not 0.0 <= args.dist_tail < 1.0:
        raise SystemExit(f"--dist-tail {args.dist_tail} must be in [0, 1)")
    if args.dist_tail + args.stage_replay >= 1.0:
        # `mix` slices `n - k - kt` and a negative stop silently keeps boards
        # instead of raising, so the pool would be wrong rather than absent.
        raise SystemExit(f"--dist-tail {args.dist_tail} + --stage-replay "
                         f"{args.stage_replay} must be under 1.0")
    if args.dist_tail and args.backend == "cpu":
        # `build_jobs` on the cpu path generates boards from a seed rather than
        # reading the mixed pool, so the flag would be silently ignored there.
        raise SystemExit("--dist-tail needs --backend gpu or scan; the cpu path "
                         "generates boards per job and never reads the pool")
    if not 0.0 <= args.restart_frac <= 1.0:
        raise SystemExit("--restart-frac must be in [0, 1]")
    if not 1 <= args.restart_min_turn <= args.restart_max_turn <= args.max_turns:
        raise SystemExit("restart turns must satisfy 1 <= min <= max <= max-turns")
    if args.restart_frac and args.backend != "cpu":
        raise SystemExit("--restart-frac currently needs --backend cpu so the "
                         "simulator and TemporalMemory burn-in stay exact")
    if not 0.0 <= args.cf_frac <= 1.0:
        raise SystemExit(f"--cf-frac {args.cf_frac} must be in [0, 1]")
    if args.augment and args.cf_frac:
        raise SystemExit("--augment and --cf-frac are separate experimental arms; "
                         "combining them would leave fork samples untransformed "
                         "and confound the symmetry result")
    if args.restart_frac and args.cf_frac:
        raise SystemExit("--restart-frac and counterfactual forks are separate arms")
    if args.cf_frac and args.backend != "cpu":
        # THE failure this check exists for. Forks are generated in `_rollout`
        # mode 0, which is the `else` arm of the backend dispatch below; under
        # gpu/scan the training rollout is `vecroll` and that branch never runs.
        # The aux buffer would stay empty, both weights would multiply nothing,
        # and a treatment arm would be byte-identical to its control -- three
        # weeks producing two copies of the same answer with nothing in the log
        # saying why. docs/CLUSTER.md's launch line says --backend gpu, so this
        # is the default mistake, not an exotic one.
        raise SystemExit("--cf-frac needs --backend cpu: forks are generated in "
                         "_rollout, which only runs training games on the cpu "
                         "path. Under gpu/scan the fork buffer stays EMPTY and "
                         "the treatment arm silently becomes its own control.")
    if (args.cf_pref_weight or args.cf_value_weight) and not args.cf_frac:
        raise SystemExit("--cf-pref-weight/--cf-value-weight do nothing without "
                         "--cf-frac; no forks would be generated to weight.")
    if args.cf_value_weight and args.value_head != "scalar":
        # The term is written as MSE against tanh(V), matching `v_step`. Under
        # hlgauss there is no tanh -- `values_of_hl` is a mean-of-categorical
        # over `hl_kappa` -- so the same expression would train the critic
        # against a quantity it does not emit.
        raise SystemExit(f"--cf-value-weight needs --value-head scalar, not "
                         f"{args.value_head}: the successor term is MSE against "
                         f"tanh(V) and the hlgauss head has no tanh.")
    if args.cf_frac and args.stage_replay:
        # Stage 3 `bldA` is -0.9 to -3.0 and the design agrees builds are wrong
        # there, so replayed boards would pool a known-negative stratum into the
        # measurement. `cf_stage` is asserted below for the same reason.
        print(f"WARNING: --cf-frac with --stage-replay {args.stage_replay} mixes "
              f"forks from earlier stages, where builds are correctly negative "
              f"(stage 3 bldA -0.9 to -3.0). Use --stage-replay 0.", flush=True)
    # Fail here, not on the first rollout: a partial last chunk would step past
    # the turn limit, and the turn limit IS the draw rule.
    if args.backend == "scan" and (args.scan_chunk < 1
                                   or args.max_turns % args.scan_chunk):
        raise SystemExit(f"--scan-chunk {args.scan_chunk} must be a positive "
                         f"divisor of --max-turns {args.max_turns}")
    if args.probe:
        # Everything the probe does is subtraction: no promotion (so the chosen
        # --start-stage is pinned without a second code path), no comp-eval past the iteration-0
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
    # specs[:n_opp] are the comp-eval instruments; specs[n_opp] is the promotion
    # gate's opponent, which is a FILE this run rewrites on every stage entry
    # (see stage_ref below) rather than an external bot.
    opp_specs = [s.strip() for s in args.opp.split(",") if s.strip()]
    if not opp_specs:
        raise SystemExit("--opp is empty; comp-eval needs at least one opponent")
    if len(set(opp_specs)) != len(opp_specs):
        raise SystemExit(f"--opp repeats an opponent: {opp_specs}. Duplicates "
                         "double that opponent's weight in the pooled score "
                         "without saying so.")
    per_opp = args.comp_eval_games // len(opp_specs) // 2 * 2
    if per_opp < 2:
        raise SystemExit(f"--comp-eval-games {args.comp_eval_games} over "
                         f"{len(opp_specs)} opponents leaves {per_opp} games each")
    for spec in opp_specs:
        name, _, arg = spec.partition(":")
        if name in ("ours", "clone") and not Path(arg).exists():
            raise SystemExit(f"opponent {spec} points at a missing file")
        try:
            # Build each once here rather than discovering a typo inside 60
            # spawned workers, where `_rollout` would raise per game and the
            # eval would come back empty with no line saying why.
            agents.make(spec, 0, 18, 18, 0)
        except Exception as e:                       # noqa: BLE001
            raise SystemExit(f"opponent {spec} does not construct: {e}") from e

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

    raw0 = np.load(args.init)
    arch = bc.resolve_arch(args.init, None, None, None)
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
    phi = vt.init_params(jax.random.PRNGKey(args.seed), arch,
                         effective_value_hidden)   # critic, same trunk
    if args.value_head == "hlgauss":
        # SAME KEYS, wider. `arch_of` run-length-scans conv*/res* only, so
        # reshaping v_w is invisible to it; `opt_v`, `_flat("phi", ...)` and the
        # resume are comprehensions over phi.items() and need no change; and
        # `vt.forward` -- (B, ch * VALUE_POOL) @ (ch * VALUE_POOL, m) + (m,)
        # -- returns (B, m) logits with
        # no edit at all, so valuetrain.main() and tools/calibrate keep the
        # scalar head they were written against.
        phi["v_w"] = jax.random.normal(jax.random.fold_in(
            jax.random.PRNGKey(args.seed), 8),
            (ch * vt.VALUE_POOL, args.hl_bins)) * 0.01
        phi["v_b"] = jnp.zeros((args.hl_bins,))
        for key in ("v_res1_w", "v_res1_b", "v_res2_w", "v_res2_b"):
            phi.pop(key, None)       # residual scalar head is not an HL-Gauss head
    if args.init_critic:
        # AFTER the hlgauss reshape, so the shape check compares against the head
        # this run will actually use rather than the scalar one it started from.
        warm = load_critic(args.init_critic, arch, {k: np.asarray(v) for k, v in phi.items()})
        phi = {k: jnp.asarray(v) for k, v in warm.items()}
        print(f"critic: warm start from {args.init_critic} "
              f"({arch['layers']}x{arch['channels']}) -- no warmup tax, and "
              f"--start-stage is not bounded by what a blank critic can fit",
              flush=True)
    opt_p = {k: (jnp.zeros_like(v), jnp.zeros_like(v)) for k, v in theta.items()}
    opt_v = {k: (jnp.zeros_like(v), jnp.zeros_like(v)) for k, v in phi.items()}

    def p_objective(p, beta, x, mask, idx, old_logp, ref_logp, adv,
                    cf_x, cf_mask, cf_idx, cf_delta, cf_w):
        loss, aux = policy_loss(bc.forward(p, x), mask, idx, old_logp, ref_logp,
                                adv, beta, xp=jnp)
        # The counterfactual preference term. It is a SHAPING term, not an
        # estimator: the build was forced rather than sampled, so this is
        # off-policy and biased, and the unbiased correction would divide by
        # pi(build|s) -- which is ~0.003 and is the entire problem. It survives
        # only by being small, which is why the weight is 0.05 and decays out.
        #
        # delta is uncentred by design, so the SIGN of E[delta] is the sign of
        # this gradient. `tools/cfprobe` measures that before the run and is a
        # hard gate: at E[delta] <= 0 this term pushes builds DOWN.
        lp = _log_softmax(jnp.where(cf_mask, bc.forward(p, cf_x), -1e9), jnp)
        pref = -jnp.mean(cf_delta * lp[jnp.arange(cf_idx.shape[0]), cf_idx])
        return loss + cf_w * pref, aux

    @jax.jit
    def p_step(p, opt, t, beta, lr, batch):
        (loss, aux), g = jax.value_and_grad(p_objective, has_aux=True)(p, beta, *batch)
        g, norm = clip_grads(g, 0.5)
        p, opt = adam(p, opt, g, t, lr)
        return p, opt, loss, aux, norm

    @jax.jit
    def v_step(q, opt, t, x, ret, cf_x, cf_z, cf_w):
        def lo(qq):
            main = jnp.mean((jnp.tanh(vt.forward(qq, x)) - ret) ** 2)
            # s' ONLY -- the single state the forced build produced, against that
            # branch's own outcome. The control successor is already in `x`: it
            # is the next state of a real episode with a real return, so the only
            # genuinely new data here is the post-build board.
            succ = jnp.mean((jnp.tanh(vt.forward(qq, cf_x)) - cf_z) ** 2)
            return main + cf_w * succ
        loss, g = jax.value_and_grad(lo)(q)
        g, _ = clip_grads(g, 1.0)
        q, opt = adam(q, opt, g, t, args.critic_lr)
        return q, opt, loss

    # tanh: with gamma=1 the return range is exactly [-1, 1], so the critic reads
    # as 2*P(win) - 1 and cannot run away from a terminal-only signal.
    @jax.jit
    def values_of(q, x):
        return jnp.tanh(vt.forward(q, x))

    # Everything hlgauss is INSIDE the flag, table included. Built unconditionally
    # it let an hlgauss-only knob kill the DEFAULT critic: `--hl-sigma 0` divides
    # by zero inside the erf and, because the edges are a numpy array, produces a
    # silent all-NaN table rather than raising.
    if args.value_head == "hlgauss":
        if args.hl_bins < 2 or args.hl_sigma <= 0:
            raise SystemExit(f"--hl-bins {args.hl_bins} must be >= 2 and "
                             f"--hl-sigma {args.hl_sigma} > 0; the label table "
                             f"is otherwise empty or silently NaN.")
        hl_c, hl_t = hl_table(args.hl_bins, args.hl_sigma)
        # Half the z = +-1 Gaussian is truncated away, so the CE optimum is
        # kappa * V*. Divided out in `values_of_hl`; see `hl_table` for why GAE
        # does NOT hide it.
        hl_kappa = float(hl_t[2] @ hl_c)
        hl_c, hl_t = jnp.asarray(hl_c), jnp.asarray(hl_t)

    @jax.jit
    def v_step_hl(q, opt, t, x, ret, cf_x, cf_z, cf_w):
        # cf_* are accepted and IGNORED: this head has no tanh, so the successor
        # term's expression does not apply to it. Argument parsing refuses
        # --cf-value-weight unless --value-head is scalar, so a nonzero weight
        # cannot reach here silently; taking the arguments keeps ONE call site
        # in the training loop rather than a branch that could drift.
        del cf_x, cf_z, cf_w
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
        # Bounded by [c_0, c_-1] / kappa for any logits, garbage included.
        return (jax.nn.softmax(vt.forward(q, x), axis=-1) @ hl_c) / hl_kappa

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
    specs = [*opp_specs, f"clone:{stage_ref}"]
    stage_mode = len(opp_specs) + 1

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
    # The critic selected on the SAME comp-eval as best_theta. A run's ordinary
    # resume carries the latest phi, which may be dozens of updates newer than
    # the policy written to --out; exporting that pair as if it matched made the
    # next run's warm start an uncontrolled distribution shift.
    best_phi = {k: np.asarray(v) for k, v in phi.items()}
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
        stored_phi = _unflat("phi", a)
        # Exact schema migration for pre-residual scalar critics. The current
        # template's W2 is zero, so these additions cannot move V(s).
        if effective_value_hidden and "v_res1_w" not in stored_phi:
            for key in ("v_res1_w", "v_res1_b", "v_res2_w", "v_res2_b"):
                stored_phi[key] = np.asarray(phi[key])
        phi = {k: jnp.asarray(v) for k, v in stored_phi.items()}
        # A --value-head mismatch here is a shape error deep inside a jitted
        # v_step, hours after the preemption this resume exists to survive.
        want_v = ((ch * vt.VALUE_POOL, args.hl_bins)
                  if args.value_head == "hlgauss"
                  else (ch * vt.VALUE_POOL,))
        if phi["v_w"].shape != want_v:
            raise SystemExit(f"--resume: this checkpoint's critic head is "
                             f"{tuple(phi['v_w'].shape)} but --value-head "
                             f"{args.value_head} wants {want_v}.")
        opt_p = {k: (jnp.asarray(v), jnp.asarray(_unflat("optp_v", a)[k]))
                 for k, v in _unflat("optp_m", a).items()}
        opt_v = {k: (jnp.asarray(v), jnp.asarray(_unflat("optv_v", a)[k]))
                 for k, v in _unflat("optv_m", a).items()}
        for k in set(phi) - set(opt_v):
            opt_v[k] = (jnp.zeros_like(phi[k]), jnp.zeros_like(phi[k]))
        best_theta = _unflat("best", a)
        stored_best_phi = _unflat("bestphi", a)
        if stored_best_phi:
            if effective_value_hidden and "v_res1_w" not in stored_best_phi:
                for key in ("v_res1_w", "v_res1_b", "v_res2_w", "v_res2_b"):
                    stored_best_phi[key] = np.asarray(phi[key])
            best_phi = stored_best_phi
        else:
            # Backward compatibility for old runs. The final transfer probe is
            # still mandatory, so this approximation cannot silently enter a
            # league; new checkpoints always carry bestphi.
            best_phi = {k: np.asarray(v) for k, v in phi.items()}
            print("WARNING resume predates matched best-critic storage; using "
                  "its latest critic and requiring the transfer probe", flush=True)
        # Restore the promotion gate's opponent too: without it the gate silently
        # re-anchors to wherever the policy happened to be at the restart, which
        # makes every post-preemption promotion decision incomparable with the
        # ones before it.
        stage_theta = _unflat("stageref", a)
        publish(stage_theta, stage_ref)
        missing = RESUME_SCALARS - set(s)
        # Compatibility with existing default-lambda runs. A non-default lambda
        # may never be grafted onto an old resume whose credit setting is unknown.
        if "lam" in missing and args.lam == LAM:
            s["lam"] = LAM
            missing.remove("lam")
        # Symmetry augmentation changes the sampling distribution. Historical
        # resumes are unambiguously the false/default arm and may migrate only
        # when the caller also requests that arm.
        if "augment" in missing and not args.augment:
            s["augment"] = False
            missing.remove("augment")
        if "value_hidden" in missing and effective_value_hidden == 0:
            s["value_hidden"] = 0
            missing.remove("value_hidden")
        if missing:
            raise SystemExit(f"--resume: {resume_path} predates this build, "
                             f"missing scalars {sorted(missing)}. Start fresh.")
        if float(s["lam"]) != float(args.lam):
            raise SystemExit(f"--resume: checkpoint used --lam {s['lam']}, you "
                             f"passed {args.lam}. Use the original value or a new --out.")
        if bool(s["augment"]) != bool(args.augment):
            raise SystemExit(f"--resume: checkpoint used --augment={s['augment']}, "
                             f"you passed {args.augment}. Use the original setting "
                             "or a new --out.")
        if int(s["value_hidden"]) != int(effective_value_hidden):
            raise SystemExit(f"--resume: checkpoint used value_hidden="
                             f"{s['value_hidden']}, this run requests "
                             f"{effective_value_hidden}. Use the original setting "
                             "or a new --out.")
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
        initializer=_init,
        initargs=(str(live), tuple(specs), args.max_turns, args.cf_frac))

    # The counterfactual ring. Fork rows accumulate across iterations and each
    # gradient step resamples a fixed minibatch WITH REPLACEMENT from whatever is
    # in it. Training only on the ~64 forks an iteration produced is the failure
    # `tools/buildprior` records: 3,711 frames at 320 exposures each into a small
    # net memorised them and moved p(build) not at all. By iteration 200 this arm
    # has drawn from ~12,800 distinct forks instead.
    cf_ring: dict[str, np.ndarray] = {}
    cf_n, cf_seen = 0, 0          # rows currently valid, rows ever written
    cf_last: list[dict] = []      # this iteration's forks, for the log line
    value_replay = GameBalancedReplay(2 * args.critic_replay_games)

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
            host = pools.mix(args.pool_dir, stage, args.stage_replay,
                             tail=args.dist_tail)
            vec.update(stage=stage, pool=vecroll.device_pool(host))
            dmax = STAGES[stage][1]
            far = float((host["dist"] > dmax).mean()) if dmax is not None else 0.0
            print(f"  pool: stage {stage}, {len(host['dist'])} boards, "
                  f"mean dist {host['dist'].mean():.1f}"
                  f"{f', {args.stage_replay:.0%} replay of stages 0-{stage - 1}' if stage and args.stage_replay else ''}"
                  # Printed as MEASURED, not as requested: the mean dist and this
                  # share are the only visible proof the pool is what the flags
                  # said, and a stage header has lied about its boards before.
                  f"{f', {far:.0%} tail beyond dist {dmax}' if far else ''}",
                  flush=True)
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
        nonlocal rewinds, theta, phi, opt_p, opt_v, t_p, t_v
        nonlocal warmed, low, lr_scale, beta
        rewinds += 1
        print(f"  COLLAPSE {why}; rewinding to iter {best_iter}, halving lr "
              f"(rewind {rewinds}/{MAX_REWINDS})", flush=True)
        if rewinds > MAX_REWINDS:
            print("  giving up: the schedule is wrong, not unlucky. "
                  "Lower --lr or raise --warm-evar.", flush=True)
            return True
        theta, phi, opt_p, opt_v, t_p, t_v = restore_actor_critic(
            best_theta, best_phi, jnp)
        # The selected policy and critic are a matched snapshot. Resetting the
        # value optimiser avoids replaying moments learned on the abandoned
        # actor distribution; readiness is then re-established on fresh games.
        warmed, low = False, 0
        lr_scale *= 0.5
        beta = min(beta * 2.0, BETA_MAX)
        return False

    print(f"opponent: comp-eval {', '.join(opp_specs)} "
          f"({per_opp} games each, progress), "
          "stage-eval this policy's own stage-entry weights (promotion only)")
    if args.probe:
        pdmin, pdmax, _ = STAGES[stage]
        distance_label = "+" if pdmax is None else f"-{pdmax}"
        print(f"PROBE: stage {stage} only, {args.iters} iterations, no promotion, "
              f"no gate. The question is whether evar reaches "
              f"{args.warm_evar:.2f} at distance {pdmin}{distance_label}.", flush=True)
    started = time.time()
    evar_max = float("-inf")

    for it in range(start_it, args.iters):
        t0 = time.time()
        dmin, dmax, _ = STAGES[stage]
        snapshot = {k: np.asarray(v) for k, v in theta.items()}
        theta_old = ({k: jnp.asarray(v) for k, v in snapshot.items()}
                     if args.augment else None)
        results = (play_train(it, stage) if vec_backend
                   else play(build_jobs(
                       it, args.games, stage, args.stage_replay,
                       args.restart_frac, args.restart_min_turn,
                       args.restart_max_turn),
                             snapshot))
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

        # References survive the destructive packing below. They are copied
        # into the critic-only ring after this iteration's updates, preventing
        # one trajectory from appearing as both current and historical data in
        # the same minibatch.
        replay_pending = [(r["x"], float(r["z"]), f"stage-{stage}")
                          for r in results if "x" in r]

        # Harvest the forks BEFORE the ingest block pops `x` and `mask`. A fork
        # rides as extra keys on a seat-0 trajectory (see `_pack`), so the test
        # is a key, not a seat or a list position.
        cf_last = [r for r in results if "cf_idx" in r]
        if args.cf_frac:
            # The A0 abort. An empty harvest means fork generation is not
            # running -- the backend check should have caught it, but this is the
            # one that fires on the first iteration rather than after a night of
            # a treatment arm quietly being its own control.
            if not cf_last:
                raise SystemExit(
                    f"--cf-frac {args.cf_frac} produced NO forks at iteration "
                    f"{it}. The aux terms would multiply an empty batch and the "
                    f"arm would be its own control. Check --backend cpu.")
            for r in cf_last:
                if not cf_ring:
                    cf_ring = {
                        "x": np.empty((args.cf_buffer, features.C, features.PAD,
                                       features.PAD), np.float16),
                        "x_fork": np.empty((args.cf_buffer, features.C,
                                            features.PAD, features.PAD), np.float16),
                        "mask": np.empty((args.cf_buffer,
                                          r["cf_mask"].shape[1]), np.uint8),
                        "idx": np.empty(args.cf_buffer, np.int32),
                        "delta": np.empty(args.cf_buffer, np.float32),
                        "z": np.empty(args.cf_buffer, np.float32),
                    }
                j = cf_seen % args.cf_buffer
                cf_ring["x"][j] = r["cf_x"]
                cf_ring["x_fork"][j] = r["cf_x_fork"]
                cf_ring["mask"][j] = r["cf_mask"][0]
                cf_ring["idx"][j] = r["cf_idx"]
                cf_ring["delta"][j] = r["cf_delta"]
                cf_ring["z"][j] = r["cf_z"]
                cf_seen += 1
            cf_n = min(cf_seen, args.cf_buffer)
        # Linear decay to zero over --cf-iters. Past it the run is pure PPO, and
        # every later iteration is directly comparable with a run that never had
        # the term at all.
        cf_ramp = max(0.0, 1.0 - it / args.cf_iters) if args.cf_iters else 0.0
        # jnp scalars, not Python floats: a float is a STATIC argument to a jitted
        # step, and this one changes every iteration, so it would recompile both
        # kernels 200 times. `beta` and `lr` get away with it because they only
        # ever take a handful of distinct values.
        cf_pw = jnp.float32(args.cf_pref_weight * cf_ramp)
        cf_vw = jnp.float32(args.cf_value_weight * cf_ramp)

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
            adv[k:k + length] = gae(z, vals[k:k + length], lam=args.lam)
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
        def _evar(m, v=None):
            """1 - Var(z - V)/Var(z) over a subset of the buffer.

            NO ABSOLUTE VALUE HERE IS A QUALITY SCORE. V* is a martingale, so
            its explainable share of Var(z) rises from map-determined at t=0 to
            1.0 at t=T, and every subset carries a ceiling set by how fast games
            resolve and how often a loser is sniped through fog -- simulated at
            0.29..1.00 for a PERFECT critic in the last decile alone. What is
            readable is the GAP to `sc`, the same quantity for a predictor that
            cannot see the board (see below): the ceiling is common to both and
            cancels. Read the module docstring before reading these numbers.
            """
            r = ret[m]
            v = vals[m] if v is None else v
            rv = float(r.var())
            return float(1.0 - ((r - v).var() / rv)) if rv > 1e-9 else float("nan")

        # The ceiling-free control: least squares on ALL the broadcast scalar
        # planes, which are constant over the grid. The temporal suffix also
        # contains spatial planes, so the scalar indices are explicit. A
        # predictor with no board at all, fit
        # on the same subset, facing the same martingale ceiling. `evar_l - sc_l
        # <= 0` says the critic extracts nothing the clock does not already give;
        # both low and equal says the state does not contain it and no value loss
        # recovers it.
        #
        # `features.C - 8` until 2026-08-05, which was right at C=20 and silently
        # WRONG the moment the encoder grew: at C=22 it selected channels 14-21
        # and dropped CLOCK and PARITY, at C=24 it also dropped GROW_PHASE and
        # DEATHTOUCH -- so the control lost the clock, the one scalar this
        # module's own docstring calls mechanically necessary for predicting a
        # draw, and every sc number printed after the migration was a different
        # instrument under the old name. Anchored to the first scalar channel
        # instead of counted back from the end, so it survives the next one.
        scalar_idx = list(range(features.CLOCK, features.BASE_C)) + [
            features.DELTA_MY_ARMY, features.DELTA_OPP_ARMY,
            features.DELTA_LAND_ADV]
        sc = np.c_[xs[:, scalar_idx, 0, 0].astype(np.float32),
                   np.ones(n, np.float32)]

        def _evar_scalars(m):
            a, r = sc[m], ret[m]
            return _evar(m, a @ np.linalg.lstsq(a, r, rcond=None)[0])

        evar = _evar(slice(None))
        mid = (frac >= 0.2) & (frac < 0.45)      # the midgame collapse window
        evar_e, evar_m, evar_l = _evar(frac < 0.1), _evar(mid), _evar(frac >= 0.9)
        sc_m, sc_l = _evar_scalars(mid), _evar_scalars(frac >= 0.9)
        if evar == evar:
            evar_max = max(evar_max, evar)
        t_ing = time.time()      # GAE is a host python loop; it counts as ingest

        # Warm up on the CRITIC'S SCORE, not on a fixed iteration count -- the
        # 1500-iteration run released the policy after 3 iterations with evar
        # still at +0.02 and eval fell 0.175 -> 0.000 by iteration 160. And the
        # gate is STANDING, not a latch: in self-play the critic chases the
        # outcome of a game between two policies that are both moving, so evar
        # can be healthy and then decay, and a permanent latch would not notice.
        ready = critic_ready(it, evar, args.warm_evar)
        warmed, low, refroze = readiness_step(
            warmed, low, ready, REFREEZE_AFTER)
        if refroze:
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
        kill2 = (args.frozen_kill > 0 and not warmed
                 and it > WARMUP * 10 and low >= args.frozen_kill)

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
                if args.augment:
                    host_mask = np.unpackbits(
                        packed[sel], axis=1)[:, :features.N_ACTIONS].astype(bool)
                    host_x, host_mask, host_idx = bc.augment_ppo(
                        xs[sel], host_mask, idxs[sel], int(rng.integers(8)))
                    xb = to_x(host_x)
                    maskb, idxb = jnp.asarray(host_mask), jnp.asarray(host_idx)
                    # A transformed action has a different density under a
                    # non-equivariant net. Reusing rollout log-probs poisons the
                    # PPO ratio, so recompute both frozen-policy densities.
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
                # ONE call shape whether or not the mechanism is on: with no ring
                # the aux batch is a single row of the main batch at weight 0.0,
                # which costs one extra row's forward and removes a second code
                # path that could drift away from the one under test.
                if cf_n:
                    # WITH REPLACEMENT and always --cf-minibatch wide: a batch
                    # that grew with the ring would retrace XLA on every
                    # iteration until the ring filled.
                    csel = rng.integers(0, cf_n, size=args.cf_minibatch)
                    cf_b = (to_x(cf_ring["x_fork"][csel]),
                            to_mask(cf_ring["mask"][csel]),
                            jnp.asarray(cf_ring["idx"][csel]),
                            jnp.asarray(cf_ring["delta"][csel]))
                    cf_vb = (to_x(cf_ring["x"][csel]), jnp.asarray(cf_ring["z"][csel]))
                else:
                    cf_b = (xb[:1], maskb[:1], idxb[:1],
                            jnp.zeros(1, jnp.float32))
                    cf_vb = (xb[:1], jnp.zeros(1, jnp.float32))
                if do_policy:
                    batch = (xb, maskb, idxb, oldb, refb,
                             jnp.asarray(adv[sel]), *cf_b, cf_pw)
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
                    rx, rz = value_replay.sample(rng, nr)
                    vxb = jnp.concatenate([xb[:mb - nr], to_x(rx)], axis=0)
                    vret = np.concatenate([vret[:mb - nr], rz])
                t_v += 1
                phi, opt_v, vl = v_step(phi, opt_v, t_v, vxb, jnp.asarray(vret),
                                        *cf_vb, cf_vw)
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

        for replay_x, replay_z, replay_source in replay_pending:
            value_replay.add(replay_x, replay_z, replay_source)

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
        bmask = (idxs % features.PER_CELL) == features.BUILD_OFFSET
        builds = int(bmask.sum())
        # `bldA` is the mean NORMALISED advantage PPO actually applied to those
        # build actions -- the gradient itself, not an inference about it.
        #
        # It exists because three measurements could not all be true at once.
        # `tools/vprobe` says this critic values a castle at +0.28 in z units,
        # within 20% of the +111 Elo the heuristic measures for castles; the 35
        # army costs it only -0.004; and yet PPO drove `bld` from 0.57 to 0.08 in
        # 31 iterations with `--stage-replay 0`, so the replay boards are not the
        # cause either. Under the straightforward GAE story the delta at a build
        # is V(after) - V(before) > 0 and the rate should RISE.
        #
        # bldA > 0 -> the gradient does favour building and something downstream
        #             suppresses it anyway. Look at the ratio clip and the KL term.
        # bldA < 0 -> the critic's castle valuation does not survive at the states
        #             where builds really happen, vprobe's rear-tile perturbation
        #             is unrepresentative, and its conclusion does not hold.
        #
        # Normalised, so read the SIGN and the size relative to ADV_CLIP; it is
        # not comparable across iterations in absolute units.
        bld_adv = float(adv[bmask].mean()) if builds else 0.0
        # The freeze counter is in the tag, not just the fact of it: a frozen
        # policy is a legitimate state for a few iterations and a dead run after
        # FROZEN_KILL, and the line has to say which one you are looking at.
        tag = (f"  [critic warmup {low}/{args.frozen_kill or 'off'}]"
               if not do_policy else "")
        # The A3 abort, on the line rather than in a notebook. `cfA` is the
        # stratum-A mean of the SAME uncentred delta the preference term
        # multiplies, so its sign is the sign of that gradient: `tools/cfprobe`
        # measures it on frozen weights before the run, and this is the same
        # number under training. Strata are never pooled -- B is every other
        # build-legal moment and is expected to be negative.
        cfs = ""
        if args.cf_frac and cf_last:
            da = [float(r["cf_delta"]) for r in cf_last if int(r["cf_stratum"]) == 0]
            db = [float(r["cf_delta"]) for r in cf_last if int(r["cf_stratum"]) == 1]
            cfs = (f"\n            cf {len(cf_last):3d} fork  ring {cf_n:5d}  "
                   f"cfA {np.mean(da):+.3f}/{len(da):3d}  "
                   f"cfB {np.mean(db):+.3f}/{len(db):3d}  "
                   f"w {float(cf_pw):.4f}/{float(cf_vw):.4f}  "
                   f"capt {np.mean([int(r['cf_captured']) for r in cf_last]):.2f}"
                   if da and db else
                   f"\n            cf {len(cf_last):3d} fork  ring {cf_n:5d}  "
                   f"one stratum only (A {len(da)} B {len(db)})")
        print(f"iter {it:5d}  stage {stage} ({dmin}{hi})  games {len(g0)}  "
              f"W/D/L {w}/{d}/{len(g0) - w - d}  samp {n // 1000:3d}k  "
              f"turns {turns:.0f}  dist {dist:.1f}  bld {builds / len(g0):.2f}"
              f" bldA {bld_adv:+.2f}"
              f"{util}{tag}\n"
              # `v` is MSE on the scalar head and KL nats on hlgauss -- two
              # different units, so the field is NAMED after the head rather
              # than letting a 0.21 and a 0.21 look like the same number.
              f"            pg {pg:+.4f}  "
              f"{'vkl' if args.value_head == 'hlgauss' else 'v'} {vloss:.3f}  "
              f"evar {evar:+.2f} (e{evar_e:+.2f} m{evar_m:+.2f} l{evar_l:+.2f} "
              f"sc m{sc_m:+.2f} l{sc_l:+.2f})  "
              f"{'dnp' if vec_backend else 'dlp0'} "
              f"{dnp if vec_backend else dlp0:.1e}  "
              f"klU {kl_u:.3f}/{kl_last:.3f}  klA {kl_a:.3f}  "
              f"beta {beta:.3f}  ent {ent:.2f}  gn {gn:.2f}  mb {nb}  "
              f"{time.time() - t0:.1f}s (roll {t_roll - t0:.1f}"
              f"{d2h} "
              f"pack {t_pack - t_roll:.1f} ing {t_ing - t_pack:.1f} "
              f"trn {time.time() - t_ing:.1f})"
              f"{cfs}", flush=True)

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
            # Every opponent plays the SAME boards, so the per-opponent numbers
            # are paired and a hard block of maps cannot look like one opponent
            # getting stronger. Splitting a fixed budget also means adding an
            # instrument costs games rather than wall clock.
            jobs = [j for m in range(1, len(opp_specs) + 1)
                    for j in eval_jobs(per_opp, rules.MIN_GENERALS_DISTANCE, None,
                                       SP_COMP_SEED0 + blk * (args.comp_eval_games // 2), m)]
            res = play(jobs, cur)
            score, per = tally(res, specs)
            # stderr over what was PLAYED: `_rollout` returns None when the
            # opponent faults out, and quoting the requested count would both
            # understate the interval and tighten the guard below.
            se = stderr(len(res))
            mark = ""
            if score > best_score:
                best_score, best_iter, best_theta = score, it, cur
                best_phi = {k: np.asarray(v) for k, v in phi.items()}
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
            # The breakdown is appended, never inserted: field 3 stays the score
            # so `grep comp-eval | sort -k3 -g` keeps working on old and new logs.
            # An opponent sitting at 1.000 is saturated -- it is spending games
            # to say nothing and should come out of --opp.
            detail = ("  [" + " ".join(f"{_short(s)} {per[s]:.2f}"
                                       for s in opp_specs if s in per) + "]"
                      if len(opp_specs) > 1 else "")
            print(f"comp-eval {it:5d}  {score:.3f} +-{se:.3f} "
                  f"({len(res)}/{len(jobs)} games, dist 17+)  "
                  f"base {base_score:.3f} kill<{base_score - band:.3f}{mark}{detail}",
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
                # A stage transition changes the board/episode distribution.
                # Do not spend up to REFREEZE_AFTER actor updates before the
                # standing gate notices that the old-stage critic is stale.
                warmed, low = False, 0
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
                         **_flat("bestphi", best_phi),
                         **_flat("stageref", stage_theta),
                         **_flat("init", theta_init)},
                        {"it": it, "stage": stage, "over": over,
                         "resident": resident, "beta": float(beta),
                         "lr_scale": float(lr_scale), "t_p": t_p, "t_v": t_v,
                         "warmed": bool(warmed), "low": low, "rewinds": rewinds,
                         "best_score": float(best_score), "best_iter": best_iter,
                         "base_score": None if base_score is None else float(base_score),
                         "comp_low": comp_low,
                         "lam": float(args.lam),
                         "augment": bool(args.augment),
                         "value_hidden": int(effective_value_hidden),
                         "rng_state": rng.bit_generator.state})

    if args.probe:
        pool.shutdown()
        ok = evar_max >= args.warm_evar
        pdmin, pdmax, _ = STAGES[stage]
        distance_label = "+" if pdmax is None else f"-{pdmax}"
        print(f"\nPROBE over {args.iters} iterations at stage {stage} "
              f"(distance {pdmin}{distance_label}): best evar {evar_max:+.3f}, "
              f"needs {args.warm_evar:+.3f}")
        print("VERDICT: " + ("the critic DOES explain the on-policy return at "
                             "this distance, so the overnight run has cleared "
                             "its value-transfer precondition."
                             if ok else
                             "the critic does not explain the on-policy return at "
                             "this distance. Do not launch the overnight run; "
                             "refit or change the critic and repeat this probe."))
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
    # A load_critic-compatible, policy-matched warm start. It intentionally uses
    # the run/resume `phi__*` layout rather than masquerading as a standalone
    # schema-2 offline critic with a validation sidecar it does not have.
    critic_out = out.with_suffix(".critic.npz")
    critic_tmp = critic_out.with_suffix(".tmp.npz")
    np.savez(critic_tmp, **_flat("phi", best_phi))
    os.replace(critic_tmp, critic_out)
    publish({k: np.asarray(v) for k, v in theta.items()}, out.with_suffix(".last.npz"))
    out.with_suffix(".json").write_text(json.dumps(
        {"accepted": accepted, "score": round(score, 4),
         "init_score": round(init_score, 4), "margin": round(margin, 4),
         "base_score": None if base_score is None else round(base_score, 4),
         "gate_games": len(gate), "stage_reached": stage,
         "best_iter": best_iter, "best_comp_eval": round(best_score, 4),
         "opponent": opp_specs}, indent=2) + "\n")

    print(f"\ngate on {len(gate)} fresh competition-distance games: trained "
          f"{score:.3f} vs init {init_score:.3f}, needs +{margin:.3f} -> "
          f"{'ACCEPTED' if accepted else 'REJECTED'}   (reached stage {stage})")
    print(f"wrote {out} with matched critic {critic_out}  "
          f"({time.time() - started:.0f}s)")
    from tools import manifest
    inputs = [args.init, out, critic_out, out.with_suffix(".last.npz")]
    if args.init_critic:
        inputs.append(args.init_critic)
    manifest.write(out.with_suffix(".manifest.json"), command=sys.argv,
                   artifacts=inputs,
                   extra={"kind": "curriculum-ppo", "args": vars(args),
                          "gate": {"accepted": accepted, "score": score,
                                   "init_score": init_score, "margin": margin},
                          "stage_reached": stage, "opponents": opp_specs})
    print("Confirm it before believing it:")
    print(f"  python -m arena.runner --a clone:{out} --b ours:configs/v16.json "
          f"--games 400 --workers {args.workers}")


if __name__ == "__main__":
    main()
