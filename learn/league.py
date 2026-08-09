"""PSRO league over the config space: the one local scoreboard that grades itself.

Eight local instruments (arena vs hand-written bots, self-play SPRT, a behaviour
clone, a field-trained win-probability model, the hunter gauntlet, ...) have all
disagreed with the ladder in the same direction: they reward committing army,
because no opponent any of them contains punishes over-commitment. v12 beat the
local arena by a wide margin and lost 241 Elo on the ladder. Every one of those
instruments needed an external yardstick to be valid and none of them had one.

A double-oracle league needs no yardstick. The only question it asks is "does
this config beat the archive", which is objectively answerable, and the archive
is grown by the search itself:

    archive P = { seeds }
    repeat:  sigma <- Nash(P)                     # meta-solve
             c     <- CEM best response to sigma  # oracle
             P     <- P + {c}; extend the payoff matrix

The oracle step is the whole point. The diagnosis is "no constructible opponent
punishes over-commitment"; constructing that opponent is literally the oracle's
job. It searches CONFIG space, not weights, so every candidate already plays
competently and only has to find the exploit -- learning the game from scratch
over a 3529-action space is what killed the neural exploiter.

READING THE OUTPUT
------------------
One row per archive member is printed after every iteration:

    mean      its win rate averaged over the archive. IGNORE THIS NUMBER. It is
              exactly the quantity that made v12 look good: whoever farms a
              cluster of similar weak members hardest wins the mean.
    min       its worst win rate against any single archive member, printed with
              the per-entry standard error. THIS IS THE REPORTED NUMBER, and
              max-min is the row you would submit -- the member for which the
              league contains no exploiter.
    worst-vs  which member holds it down. Read this column together with `min`.
    nash      weight in the current mixture (see fictitious_play for why Nash
              and not uniform).

Three lines under the table are the ones that catch this instrument failing the
way the previous eight did:

    support   how many members carry Nash weight. Config variants of one
              controller are mostly transitive, and on a transitive matrix
              fictitious play returns a point mass -- support 1 means the oracle
              is a best response to a SINGLE opponent, which is precisely the
              setup that produced v12. Read every result at support 1 as a
              single-opponent tune, not as a league.
    worst-vs  a tally over that column. If one member is nearly everyone's
              worst-vs, max-min has collapsed into that member's gauntlet. The
              hunter gauntlet is already-failed instrument #5; if `hunter:1` owns
              this tally, the league is reproducing it under a new name.
    falsify   v12's min minus v16's min, with its standard error, and whether
              v12's worst-vs is an oracle member.

That last line is the falsification test, and it lives in the min column, never
in the mean. Over-commitment IS locally punishable if, once oracle members enter
the archive, `ours:configs/v12.json` (the known over-committer, -241 ladder Elo)
shows a lower min than `ours:configs/v16.json` by MORE THAN TWO STANDARD ERRORS
AND its worst-vs is an oracle member rather than a seed. At --pair-games 48 one
standard error is 0.10, so a gap under 0.20 is noise and means nothing. If after
four or more iterations v12's min still sits at or above v16's, then no config
in this search space punishes over-commitment: the diagnosis survives, and the
honest negative result is that the exploiter -- if it exists at all -- does not
live in config space.

WHY THE NUMBERS ARE MEASURED THE WAY THEY ARE
---------------------------------------------
Boards are deterministic. Every agent here ignores its seed (only `random` uses
it), so an agent pair plus a board plus a seating IS one game, always the same
game. A pool of L boards therefore holds 2L distinct games and not one more:
asking for more replays identical games and inflates n with zero information.
The run refuses to start when a game count exceeds what its slice of the pool
can supply.

Three things must not share boards, and with --maps the pool is split in three
so they cannot. `pool_grid` indexes `pool[seed % L]`, so "disjoint seed windows"
is a fiction on a finite pool -- on runs/realmaps.json (36 boards) the windows
this file used to declare overlapped on 12 of 24. The split is structural:

    oracle boards    what the CEM selects on
    matrix boards    what the payoff matrix grades on
    runoff boards    fresh boards for the final re-measurement

The runoff exists because max-min is a selection. `min` over n-1 noisy entries
is biased down, and argmax over those minima picks whichever member's noise was
kindest; selecting and reporting on the same games is the v12 mistake with more
steps. The top three by `min` are replayed against the whole archive on the
runoff boards and the winner of those fresh numbers is what gets written.

Confirm the winner with an SPRT run before submitting it; a league is a search,
not a significance test.

THREE ORACLES (--oracle)
------------------------
    config    CEM over ~24 config knobs. The default, and the only one that
              produces a shippable answer.
    net       PPO over network weights (learn/netoracle.py), because a config
              can only reweight behaviours `bot/policy/controller.py` already
              implements and PSRO's prescription when the oracle class stops
              finding best responses is to widen it. Needs --nn-init.
    both      Append both members per iteration. Grows the archive fastest.

The net oracle's GATE is not "is this good": it is "did PPO beat its OWN
initialisation, paired, on boards no checkpoint was selected on". A net that
fails it never enters the archive and the CEM oracle runs for that iteration
instead, so an iteration always appends something. Two lines are printed per
call -- the gate verdict and its per-opponent bracket -- and both come from
netoracle's stdout, which is worth reading live: it prints a kill checklist.

An `oracle-nn-*` row in the table is a NETWORK. If it wins max-min, this writes
`<out>.npz`. It becomes a submission candidate only after the promotion suite
passes and that exact file is copied to `bot/weights.npz` for hash-locked
packaging.

    # Does this deserve a night on the cluster? One oracle call against v12
    # alone, plus a control against v16. ~1 hour.
    python -m learn.league --probe --dir runs/probe \
        --pop 20 --gens 4 --games 48 --maps runs/bigmaps.json --workers 60

    # The league.
    python -m learn.league --dir runs/league --out runs/league/best.json \
        --iters 6 --gens 4 --pop 20 --elite 5 --games 48 --pair-games 48 \
        --maps runs/bigmaps.json --workers 60

    # The league with the wider oracle class. Each --oracle net iteration costs
    # ~1 h of GPU on top of the CEM budget.
    python -m learn.league --dir runs/league-nn --out runs/league-nn/best.json \
        --oracle both --nn-init runs/top3-v2/incumbent-c40-context.npz \
        --nn-init-critic runs/top3-v2/incumbent-value.npz --nn-iters 200 \
        --iters 6 --gens 4 --pop 20 --elite 5 --games 48 --pair-games 48 \
        --maps runs/bigmaps.json --workers 60

Both default to --group commit: 24 knobs, because pop*gens is ~80 samples and CEM
resolution goes as samples per dimension -- the full 114-axis space at that
budget is a random walk dressed as a search. --group all is honest only with
roughly ten times the games.

--games 48 needs 24 boards per slice, so --maps needs a pool of at least 72:

    python -m analysis.official maps runs/official6 --player <handle> \
        --only all --out runs/bigmaps.json

runs/realmaps.json holds 36 and caps both counts at 24 (standard error 0.14,
which cannot resolve anything this search is looking for). Without --maps the
boards are generated and there is no cap.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from arena.runner import run_many, run_match, tally
from bot import rules
from bot.config import Config

# Seed origins for generated boards. The oracle must not be scored on the boards
# the payoff matrix is built from, or it overfits the very games that decide
# whether it gets archived, and the final runoff must see neither. On generated
# boards these ranges are disjoint because the seed IS the board; on a map pool
# `pool_grid` takes the seed modulo the pool length and disjointness has to come
# from splitting the pool instead (see split_pool).
MATRIX_SEED0 = 50_000
ORACLE_SEED0 = 200_000
HOLDOUT_SEED0 = 900_000
NN_REWARD_MODES = ("terminal", "tempo")
NN_TEMPO_EPS = 0.03
PFSP_WEIGHTINGS = ("variance", "linear", "squared")


# --------------------------------------------------------------------------
# search space
#
# Ranges are priced against the feature each weight multiplies (see
# bot/policy/features.py): indicators are 0/1, `progress` and `toward_enemy` are
# +-1 BFS steps, `army` is log1p(moved), the rest arrive normalised into [0,1].
# So a weight's range is "how many captures is one unit of this worth".
_WEIGHT = {
    "cap_neutral": (0.0, 30.0),
    "reveal": (0.0, 12.0),
    "cap_enemy": (0.0, 30.0),
    "cap_enemy_army": (0.0, 10.0),
    "cap_castle": (0.0, 80.0),      # +0.5 army/turn forever, the biggest bonus
    "to_own": (-5.0, 5.0),
    "progress": (0.0, 40.0),
    "army": (0.0, 10.0),
    "from_general": (-20.0, 2.0),   # positive means spending the base
    "keep_home": (-2.0, 8.0),
    "failed_capture": (-150.0, -5.0),   # a wasted turn; must dominate progress
    "contact": (-8.0, 12.0),        # seeking and avoiding contact are both plausible
    "frontier": (0.0, 12.0),
    "stack_break": (-20.0, 4.0),    # really the split penalty
    "toward_enemy": (-6.0, 8.0),
}
# Where a mode genuinely wants a different range: GATHER exists to move one big
# stack (own-tile moves are the point, splitting it is the failure mode), ATTACK
# exists to resist the neutral-nibbling distraction.
_WEIGHT_OVERRIDE = {
    "gather.to_own": (-5.0, 8.0),
    "gather.army": (0.0, 12.0),
    "gather.stack_break": (-25.0, 2.0),
    "attack.cap_neutral": (0.0, 20.0),
}

# Scalar knobs. Types come from the Config default, so this table cannot drift
# out of sync with the dataclass. Deliberately absent: castle_max_cost (35 is the
# engine's surcharge cliff, not a preference), pass_score and time_budget_ms
# (safety, not strategy), deathtouch_prep_turn / deathtouch_guard_turn (turn 730+
# is reached in a small minority of games, so samples spent there are wasted).
_SCALARS = {
    "first_expand_turn": (5, 90),
    "lock_enabled": (0, 1),
    "lock_radius": (0, 12),
    "general_block_radius": (0, 10),
    "general_block_ratio": (0.0, 3.0),
    "castle_enabled": (0, 1),
    "castle_min_turn": (20, 300),
    "castle_max_turn": (200, 900),
    "max_castles": (0, 16),
    "castle_min_land": (0, 60),
    "castle_safe_dist": (0, 20),
    "castle_keep": (1, 12),
    "castle_gather_min_army": (50, 900),   # highest-leverage number in the file
    "defend_margin": (0.6, 3.0),
    "defend_horizon": (4, 40),
    "defend_hold": (0, 30),
    "garrison_frac": (0.0, 0.3),
    "guard_radius": (0, 10),
    "garrison_cap": (0, 200),
    "garrison_from_turn": (0, 300),
    "corridor_defence": (0, 1),
    "corridor_max_dist": (2, 20),
    "attack_margin": (0.8, 3.0),
    "attack_margin_unsure": (1.0, 5.0),
    "probe_turn": (40, 500),
    "located_candidates": (1, 20),
    "attack_defense_frac": (0.0, 1.0),
    "attack_trust_sight": (0, 1),
    "gather_start_turn": (10, 300),
    "gather_army_ratio": (0.5, 3.0),
    "expand_floor": (0, 20),
    # The over-commitment knobs. These are the reason the league exists: lowering
    # thrust_min_army / thrust_army_ratio won two local instruments and lost 241
    # ladder Elo. If a punisher exists, it is found by moving these.
    "thrust_enabled": (0, 1),
    "thrust_min_turn": (20, 400),
    "thrust_min_army": (4, 120),
    "thrust_army_ratio": (0.0, 0.6),
    "thrust_abort_army": (1, 40),
    "thrust_home_safe": (0, 30),
    "mass_enabled": (0, 1),
    "mass_min_army": (10, 300),
}


def _build_space() -> list[tuple[str, float, float, type]]:
    cfg = Config()
    space = []
    for block in ("expand", "gather", "attack", "defend", "deathtouch"):
        for attr, rng in _WEIGHT.items():
            name = f"{block}.{attr}"
            lo, hi = _WEIGHT_OVERRIDE.get(name, rng)
            space.append((name, float(lo), float(hi), float))
    for name, (lo, hi) in _SCALARS.items():
        space.append((name, float(lo), float(hi), type(getattr(cfg, name))))
    return space


SPACE = _build_space()

# CEM resolution goes as samples-per-dimension, and the default budget is
# pop*gens = 80 samples for 114 axes: the elite mean random-walks and the oracle
# emits perturbations of its own start config instead of an exploiter.
# tools/tune.py learned this already ("searching every knob at once wastes
# samples on things that barely matter") and searches named groups. `commit` is
# the subset the diagnosis is about: what leaves home, and what happens to the
# side that sent it. Search `all` only with a budget to match.
GROUPS = {
    "commit": [
        "thrust_enabled", "thrust_min_turn", "thrust_min_army", "thrust_army_ratio",
        "thrust_abort_army", "thrust_home_safe", "mass_enabled", "mass_min_army",
        "attack_margin", "attack_margin_unsure", "attack_defense_frac",
        "defend_margin", "defend_horizon", "defend_hold",
        "garrison_frac", "garrison_cap", "guard_radius",
        "gather_start_turn", "gather_army_ratio", "expand_floor",
        "expand.from_general", "expand.keep_home",
        "gather.from_general", "attack.from_general",
    ],
    "all": [n for n, _, _, _ in SPACE],
}


def select_space(group: str, params: str | None) -> list[tuple[str, float, float, type]]:
    if params:
        names = [p.strip() for p in params.split(",") if p.strip()]
    elif group in GROUPS:
        names = GROUPS[group]
    else:
        raise SystemExit(f"unknown --group {group!r}; have {sorted(GROUPS)}")
    keep = {n: None for n in names}
    space = [e for e in SPACE if e[0] in keep]
    missing = keep.keys() - {e[0] for e in space}
    if missing:
        raise SystemExit(f"not searchable: {sorted(missing)}")
    return space


def _get(cfg: Config, name: str) -> float:
    if "." in name:
        block, attr = name.split(".")
        return float(getattr(getattr(cfg, block), attr))
    return float(getattr(cfg, name))


def encode(cfg: Config, space: list | None = None) -> np.ndarray:
    """Config -> normalised [0,1] coordinates. Out-of-range values clip."""
    v = np.array([(_get(cfg, n) - lo) / (hi - lo) for n, lo, hi, _ in (space or SPACE)])
    return np.clip(v, 0.0, 1.0)


def decode(v: np.ndarray, base: Config, space: list | None = None) -> Config:
    """Normalised coordinates -> Config, on top of `base` for unsearched fields.

    `with_vector` does the type work: ints round, bools take bool(round(x)), so a
    bool knob is just a [0,1] axis with a threshold at the midpoint.
    """
    space = space or SPACE
    vals = [lo + float(np.clip(x, 0.0, 1.0)) * (hi - lo)
            for x, (_, lo, hi, _) in zip(v, space)]
    return base.with_vector([n for n, _, _, _ in space], vals)


# --------------------------------------------------------------------------
# meta-solver
def fictitious_play(payoff: np.ndarray, iters: int = 20_000) -> np.ndarray:
    """Nash mixture of a symmetric zero-sum payoff matrix, by fictitious play.

    Uniform is the lazy meta-solver and it is wrong here. The archive fills up
    with near-identical members -- every oracle output is a variation on the last
    one -- and uniform hands that cluster most of the weight, so the oracle
    spends its next generation re-beating a config it has already beaten. Nash
    puts weight only on members that are genuinely hard to beat, which is what
    makes the best response informative.

    Both players share one empirical distribution (the game is symmetric): each
    round, best-respond to the play so far and add that response to the history.
    """
    n = len(payoff)
    counts = np.zeros(n)
    total = np.zeros(n)              # total[k] = k's score against the history
    for _ in range(iters):
        i = int(np.argmax(total))    # best response to the empirical mixture
        counts[i] += 1
        total += payoff[:, i]
    return counts / counts.sum()


def pfsp(win_rates: np.ndarray, weighting: str = "variance",
         exclude: int | None = None) -> np.ndarray:
    """Prioritised fictitious self-play distribution from matchup win rates.

    ``variance`` targets informative near-50% opponents for the main agent;
    ``linear`` and ``squared`` increasingly focus exploiters on weaknesses.
    """
    x = np.clip(np.asarray(win_rates, dtype=float), 0.0, 1.0)
    if weighting == "variance":
        w = x * (1.0 - x)
    elif weighting == "linear":
        w = 1.0 - x
    elif weighting == "squared":
        w = (1.0 - x) ** 2
    else:
        raise ValueError(f"unknown PFSP weighting {weighting!r}")
    if exclude is not None:
        w[int(exclude)] = 0.0
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones_like(x)
        if exclude is not None and len(w) > 1:
            w[int(exclude)] = 0.0
    return w / w.sum()


# --------------------------------------------------------------------------
# payoff matrix
def set_pair(P: list[list], i: int, j: int, score: float) -> None:
    """Record i's score against j, and the mirror entry it implies."""
    P[i][j] = float(score)
    P[j][i] = 1.0 - float(score)


def check_antisymmetry(P: list[list]) -> None:
    for i, row in enumerate(P):
        for j, v in enumerate(row):
            if v is None:
                continue
            assert abs(v + P[j][i] - 1.0) < 1e-9, f"P[{i}][{j}] breaks antisymmetry"


def dense(P: list[list]) -> np.ndarray:
    """Matrix with the diagonal filled in at 0.5 (a config mirrors itself)."""
    n = len(P)
    M = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(n):
            if i != j and P[i][j] is not None:
                M[i][j] = P[i][j]
    return M


def score_of(results: list[dict]) -> float:
    w, d, _ = tally(results)
    return (w + 0.5 * d) / max(len(results), 1)


def stderr(games: int) -> float:
    """Standard error of a win rate over `games` games.

    Conservative on purpose: the two games of a seed pair are the same board with
    the colours swapped, so they are correlated and the effective sample size is
    the board count, not the game count. Quoting 0.5/sqrt(games) instead would
    understate every interval in the output by 40%.
    """
    return 0.5 / max(games / 2, 1) ** 0.5


def split_pool(maps: str | None, out: Path, pair_games: int, games: int) -> tuple:
    """Cut a board pool into (oracle, matrix, runoff) files with no board in common.

    Returns three map-pool paths, or (None, None, None) when boards are generated
    and scarcity does not apply. Splitting the FILE rather than choosing seed
    offsets is the only way to get this right: `pool_grid` is `pool[seed % L]`, so
    two "disjoint" seed windows collide as soon as they wrap, and on
    runs/realmaps.json (36 boards) the windows this file used to declare shared 12
    of their 24 boards. Then the CEM selects on boards the matrix grades on, which
    is the v12 mistake rebuilt inside the instrument meant to prevent it.
    """
    if not maps:
        return None, None, None
    grids = json.loads(Path(maps).read_text())["grids"]
    third = len(grids) // 3
    need = max(pair_games, games) // 2
    if third < need:
        raise SystemExit(
            f"{maps} has {len(grids)} boards -> {third} per slice, but --pair-games "
            f"{pair_games} / --games {games} need {need}. Boards are deterministic, "
            f"so anything beyond {2 * third} games per slice replays identical games. "
            f"Build a pool of >= {3 * need} boards with `python -m analysis.official "
            f"maps`, lower the game counts, or drop --maps.")
    paths = []
    for tag, part in (("oracle", grids[:third]),
                      ("matrix", grids[third:2 * third]),
                      ("runoff", grids[2 * third:])):
        p = out / f"{tag}_maps.json"
        p.write_text(json.dumps({"grids": part}))
        paths.append(str(p.resolve()))
    print(f"board pool {maps}: {len(grids)} boards -> "
          f"{third}/{third}/{len(grids) - 2 * third} oracle/matrix/runoff")
    return tuple(paths)


# --------------------------------------------------------------------------
# oracle
def allocate(sigma: np.ndarray, games: int) -> list[int]:
    """Split `games` across archive members in proportion to sigma, in even chunks.

    run_match plays each seed twice with the colours swapped, so an odd count
    leaves one game unpaired and lets seat bias in. Largest-remainder on pairs.
    """
    pairs_total = games // 2
    raw = np.asarray(sigma, dtype=float) * pairs_total
    pairs = np.floor(raw).astype(int)
    for k in np.argsort(-(raw - pairs))[:max(pairs_total - int(pairs.sum()), 0)]:
        pairs[k] += 1
    return (pairs * 2).tolist()


def play_mixture(cfg: Config, specs: list[str], alloc: list[int], seed0: int,
                 workers: int, max_turns: int, maps: str | None,
                 tmpdir: Path, tag: str) -> float:
    """Score one config against the meta-mixture. Pooled over all games, which
    equals the sigma-weighted score because `alloc` is proportional to sigma."""
    path = tmpdir / f"cand_{tag}.json"
    cfg.save(path)
    spec = f"ours:{path.resolve()}"
    # One pool for the whole mixture: played per opponent, only alloc[i] games are
    # ever in flight, which is a quarter of a 60-core box on a five-member support.
    # ponytail: candidates are still evaluated one at a time; batch the generation
    # too if `games` ever drops below the worker count.
    batch = run_many([(spec, opp, g, seed0) for opp, g in zip(specs, alloc) if g > 0],
                     workers, max_turns, maps=maps)
    w = d = n = 0
    for res in batch:
        ww, dd, _ = tally(res)
        w, d, n = w + ww, d + dd, n + len(res)
    return (w + 0.5 * d) / max(n, 1)


def oracle(base: Config, specs: list[str], sigma: np.ndarray, *, space: list, pop: int,
           elite: int, gens: int, games: int, workers: int, max_turns: int,
           maps: str | None, tmpdir: Path, seed: int, spread: float, floor: float,
           it: int = 0, state: dict | None = None, on_gen=None) -> tuple[Config, float]:
    """Cross-entropy search for a best response to `sigma`.

    Returns (elite mean of the last generation, its search score). The elite mean
    and not the best individual ever seen: at pop*gens evaluations of `games`
    games each, the running argmax is an unpaired maximum over ~100 noisy
    estimates taken on DIFFERENT boards, worth about +2 standard errors of pure
    winner's curse, and it makes a lucky generation-0 draw impossible for a
    genuinely better generation-3 config to beat. Averaging the elite is what CEM
    is for. The returned score is a search statistic, not a measurement -- the
    payoff matrix re-measures the member on held-out boards and that is the number
    to quote.

    `state` is a mutable dict carrying {gen, mean, std}; pass the one from the
    checkpoint to resume a killed oracle mid-search, and `on_gen()` is called
    after each generation so the caller can persist it.
    """
    if state is None:
        state = {}
    dim = len(space)
    mean = np.array(state.get("mean") or encode(base, space))
    std = np.array(state.get("std") or np.full(dim, spread))
    alloc = allocate(sigma, games)
    se = stderr(games)
    score = float("nan")

    for g in range(int(state.get("gen", 0)), gens):
        t0 = time.time()
        # Same seeds for every candidate in the generation: candidates are then
        # compared on identical boards, which is worth more than doubling games.
        # The block moves with the PSRO iteration as well as the generation --
        # without `it` every iteration re-searched the same boards and the whole
        # run was decided by gens*games/2 of them (tools/tune.py:133 rotates for
        # exactly this reason).
        seed0 = ORACLE_SEED0 + (it * gens + g) * (games // 2)
        rng = np.random.default_rng(seed + g)     # reproducible after a resume
        X = np.clip(rng.normal(mean, std, size=(pop, dim)), 0.0, 1.0)
        X[0] = mean                               # always score the incumbent
        scored = []
        for k, x in enumerate(X):
            s = play_mixture(decode(x, base, space), specs, alloc, seed0, workers,
                             max_turns, maps, tmpdir, f"g{g}_{k}")
            scored.append((s, x))
            print(f"    gen{g} cand{k:02d}  {s:.3f}")
        scored.sort(key=lambda t: -t[0])
        E = np.array([x for _, x in scored[:elite]])
        mean = E.mean(axis=0)
        # Variance floor. An elite of 5 in a high-dimensional space has a sample
        # std near zero on most axes, so without this the search collapses onto
        # its own first elite set within two generations and stops exploring.
        std = np.maximum(E.std(axis=0), floor)
        score = float(np.mean([s for s, _ in scored[:elite]]))
        state.update(gen=g + 1, mean=mean.tolist(), std=std.tolist())
        print(f"  gen{g} elite mean {score:.3f} +-{se / elite ** 0.5:.3f}  "
              f"(top {scored[0][0]:.3f}, {time.time() - t0:.0f}s)")
        if on_gen:
            on_gen()

    return decode(mean, base, space), score


# --------------------------------------------------------------------------
# archive + checkpoint
def net_oracle_command(args, npz: Path, ckpt: Path,
                       maps: str | None, it: int) -> list[str]:
    """Materialise the neural subprocess command, including every train knob.

    Keeping this pure makes the forwarding contract testable without importing
    JAX or launching a multi-hour oracle. These values also live in the league
    resume identity below, so a resumed payoff matrix cannot silently change
    the optimiser that produced its members.
    """
    cmd = [sys.executable, "-m", "learn.netoracle",
           "--league", str(ckpt), "--init", args.nn_init, "--out", str(npz),
           "--workers", str(args.workers), "--iters", str(args.nn_iters),
           "--games", str(args.nn_games), "--max-turns", str(args.max_turns),
           "--epochs", str(args.nn_epochs),
           "--minibatch", str(args.nn_minibatch),
           "--lr", str(args.nn_lr),
           "--critic-lr", str(args.nn_critic_lr),
           "--critic-replay-games", str(getattr(args, "nn_critic_replay_games", 0)),
           "--critic-replay-frac", str(getattr(args, "nn_critic_replay_frac", 0.5)),
           "--critic-replay-source-floor",
           str(getattr(args, "nn_critic_replay_source_floor", 0.25)),
           "--value-hidden", str(args.nn_value_hidden),
           "--lam", str(args.nn_lam),
           "--reward-mode", str(args.nn_reward_mode),
           "--tempo-eps", str(args.nn_tempo_eps),
           "--warm-evar", str(args.nn_warm_evar),
           "--sigma-floor", str(args.nn_sigma_floor),
           "--role", str(getattr(args, "nn_role", "league-exploiter")),
           "--sampling", str(getattr(args, "nn_sampling", "nash")),
           "--pfsp-weighting", str(getattr(args, "nn_pfsp_weighting", "variance")),
           "--seed", str(args.seed + 1000 * it)]
    if args.nn_init_critic:
        cmd += ["--init-critic", args.nn_init_critic]
    if args.nn_augment:
        cmd += ["--augment"]
    if maps:
        cmd += ["--maps", maps]
    return cmd


def net_oracle(args, out: Path, ckpt: Path, maps: str | None, it: int) -> tuple[Path, dict]:
    """Widen the oracle class: PPO best response to sigma. Returns (weights, gate).

    The config oracle can only reweight behaviours `bot/policy/controller.py`
    already implements, and PSRO's prescription when the oracle class stops
    finding best responses is to widen it. See learn/netoracle.py.

    A SUBPROCESS, not an import, and that is not a style choice: `arena/runner`
    builds its pools with the default fork start method, so the moment this
    process imports jax every later matrix fork inherits an initialised,
    multithreaded runtime and deadlocks a worker (commit d683bdb). league.py must
    never import jax.
    """
    npz = (out / "nn" / f"oracle-nn-{it}.npz").resolve()
    npz.parent.mkdir(parents=True, exist_ok=True)
    cmd = net_oracle_command(args, npz, ckpt, maps, it)
    print(f"\n  neural oracle: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # A dead neural oracle must not take a multi-hour league with it. The
        # CEM branch below runs for this iteration instead; the best weights the
        # PPO run reached are still on disk as <out>.best.npz.
        print(f"  neural oracle exited {e.returncode}; falling back to the CEM "
              f"oracle for iteration {it}", flush=True)
        return npz, {"accepted": False, "score": float("nan"),
                     "init_score": float("nan")}
    gate = json.loads(npz.with_suffix(".json").read_text())
    critic = npz.with_suffix(".critic.npz")
    if gate.get("accepted") and not critic.is_file():
        print(f"  neural oracle claimed acceptance but its matched critic is "
              f"missing: {critic}; rejecting it", flush=True)
        gate["accepted"] = False
    return npz, gate


def save_state(path: Path, state: dict) -> None:
    """Atomic: this run gets killed, and a half-written checkpoint is a lost run."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)


def materialise(members: list[dict], cfgdir: Path) -> None:
    """Rewrite inline configs to disk and point each member's spec at its file.

    Oracle members live in the checkpoint, not in the working tree, so a resume
    on a clean box reconstructs the files rather than trusting they survived.
    """
    cfgdir.mkdir(parents=True, exist_ok=True)
    for m in members:
        if m.get("config") is not None:
            path = (cfgdir / f"{m['name']}.json").resolve()
            Config.from_dict(m["config"]).save(path)
            m["spec"] = f"ours:{path}"


def grow(P: list[list], n: int) -> list[list]:
    """Extend the matrix to n members, keeping every cached entry."""
    for row in P:
        row.extend([None] * (n - len(row)))
    while len(P) < n:
        P.append([None] * n)
    return P


def complete_matrix(state: dict, games: int, workers: int, max_turns: int,
                    maps: str | None, ckpt: Path) -> None:
    """Play every pair that has no cached entry yet, checkpointing after each."""
    members, P = state["archive"], grow(state["payoff"], len(state["archive"]))
    n = len(members)
    for i in range(n):
        for j in range(i + 1, n):
            if P[i][j] is not None:
                continue
            t0 = time.time()
            res = run_match(members[i]["spec"], members[j]["spec"], games,
                            MATRIX_SEED0, workers, max_turns, maps=maps)
            set_pair(P, i, j, score_of(res))
            print(f"  {members[i]['name']} vs {members[j]['name']}: "
                  f"{P[i][j]:.3f}  ({time.time() - t0:.0f}s)")
            save_state(ckpt, state)


def report(state: dict, sigma: np.ndarray, pair_games: int) -> tuple[int, np.ndarray]:
    """Print the archive table and the lines that catch the league failing.

    Returns (max-min index, mins). The three diagnostics under the table are not
    decoration: a point-mass mixture, a single member owning the worst-vs column,
    and a v12-v16 gap inside its own error bar are the three ways this instrument
    degenerates into one of the eight that already disagreed with the ladder, and
    all three are invisible in the table itself.
    """
    members = state["archive"]
    names = [m["name"] for m in members]
    M = dense(state["payoff"])
    n = len(members)
    off = ~np.eye(n, dtype=bool)
    means = np.array([M[i][off[i]].mean() for i in range(n)])
    mins = np.array([M[i][off[i]].min() for i in range(n)])
    worst = [int(np.arange(n)[off[i]][np.argmin(M[i][off[i]])]) for i in range(n)]
    se = stderr(pair_games)

    print(f"\n  {'member':<34}{'mean':>7}{'min':>7}  {'worst-vs':<26}{'nash':>7}")
    for i in np.argsort(-mins):
        print(f"  {names[i]:<34}{means[i]:>7.3f}{mins[i]:>7.3f}  "
              f"{names[worst[i]]:<26}{sigma[i]:>7.3f}")
    print(f"  every entry +-{se:.3f} (1 se on {pair_games} games); `min` is the "
          f"smallest of {n - 1} such entries, so it reads low by roughly {se:.2f}")

    support = [i for i in range(n) if sigma[i] > 0.01]
    print(f"  support   {len(support)}/{n}: {', '.join(names[i] for i in support)}")
    if len(support) == 1:
        print("            WARNING point mass. Config variants of one controller are "
              "transitive, and the oracle is now a best response to a SINGLE "
              "opponent -- that is the v12 setup, not a league.")

    hits = Counter(names[w] for w in worst)
    top, cnt = hits.most_common(1)[0]
    print(f"  worst-vs  {dict(hits)}")
    if cnt >= max(3, int(0.6 * n)):
        print(f"            WARNING {top} is the worst-vs of {cnt}/{n} members. "
              f"max-min is that member's gauntlet; a hunter gauntlet was already "
              f"tried and disagreed with the ladder.")

    v12 = next((i for i, nm in enumerate(names) if "v12" in nm), None)
    v16 = next((i for i, nm in enumerate(names) if "v16" in nm), None)
    if v12 is not None and v16 is not None:
        gap, dse = mins[v12] - mins[v16], se * 2 ** 0.5
        by_oracle = names[worst[v12]].startswith("oracle")
        print(f"  falsify   v12.min - v16.min = {gap:+.3f} +-{dse:.3f}; v12 held down "
              f"by {names[worst[v12]]} ({'oracle' if by_oracle else 'seed'})")
        if gap < -2 * dse and by_oracle:
            print("            over-commitment IS punishable in config space: an oracle "
                  "member exploits v12 harder than anything exploits v16.")
        else:
            print("            no punisher: gap is inside noise or v12's tormentor is a "
                  "seed the ladder already disagrees with.")
    return int(np.argmax(mins)), mins


def runoff(state: dict, mins: np.ndarray, top: int, games: int, workers: int,
           max_turns: int, maps: str | None) -> tuple[int, dict]:
    """Re-measure the leaders' worst case on boards nothing was selected on.

    max-min is a selection over ~n noisy minima, so the printed winner is partly
    whoever's noise was kindest -- reporting the selection and its selecting
    sample as one number is exactly what made v12 look good. This replays the top
    `top` members against the whole archive on the runoff board slice and returns
    the winner of the fresh numbers. Cost is top*(n-1)*games, a few percent of the
    run.
    """
    members = state["archive"]
    n = len(members)
    order = [int(i) for i in np.argsort(-mins)[:min(top, n)]]
    print(f"\n  runoff: top {len(order)} by min, {games} fresh games against each of "
          f"the other {n - 1} members")
    fresh = {}
    for i in order:
        others = [j for j in range(n) if j != i]
        batch = run_many([(members[i]["spec"], members[j]["spec"], games, HOLDOUT_SEED0)
                          for j in others], workers, max_turns, maps=maps)
        scores = [score_of(r) for r in batch]
        k = int(np.argmin(scores))
        fresh[i] = float(scores[k])
        print(f"    {members[i]['name']:<34}{fresh[i]:>7.3f} +-{stderr(games):.3f}  "
              f"worst-vs {members[others[k]]['name']}")
    return max(fresh, key=fresh.get), fresh


def serialise_runoff(fresh: dict[int, float]) -> dict[str, float]:
    """JSON-safe runoff scores keyed by archive index.

    ``runoff`` returns a sparse dict because only the finalists are replayed.
    Treating it like a NumPy array (`fresh.tolist()`) crashed both terminal
    league paths after all games had finished and before the manifest landed.
    """
    return {str(int(i)): float(score) for i, score in sorted(fresh.items())}


def migrate_resume_params(saved: dict | None, current: dict) -> dict | None:
    """Give old checkpoints only their unambiguous historical identities."""
    if not isinstance(saved, dict):
        return saved
    out = saved
    if "nn_lam" not in out and current.get("nn_lam") == 0.95:
        out = {**out, "nn_lam": 0.95}
    if "nn_value_hidden" not in out and current.get("nn_value_hidden") == 0:
        out = {**out, "nn_value_hidden": 0}
    if "nn_augment" not in out and current.get("nn_augment") is False:
        out = {**out, "nn_augment": False}
    if "nn_reward_mode" not in out and current.get("nn_reward_mode") == "terminal":
        out = {**out, "nn_reward_mode": "terminal"}
    if "nn_tempo_eps" not in out and current.get("nn_tempo_eps") == NN_TEMPO_EPS:
        out = {**out, "nn_tempo_eps": NN_TEMPO_EPS}
    for key, default in (("nn_critic_replay_games", 0),
                         ("nn_critic_replay_frac", 0.5),
                         ("nn_critic_replay_source_floor", 0.25)):
        if key not in out and current.get(key) == default:
            out = {**out, key: default}
    for key, default in (("nn_role", "league-exploiter"),
                         ("nn_sampling", "nash"),
                         ("nn_pfsp_weighting", "variance")):
        if key not in out and current.get(key) == default:
            out = {**out, key: default}
    return out


# --------------------------------------------------------------------------
def selfcheck() -> None:
    # Rock-paper-scissors: the unique Nash is uniform.
    rps = np.array([[0.5, 1.0, 0.0], [0.0, 0.5, 1.0], [1.0, 0.0, 0.5]])
    w = fictitious_play(rps, 20_000)
    assert abs(w.sum() - 1.0) < 1e-9, w
    assert np.abs(w - 1 / 3).max() < 0.02, w

    # Same game plus a strategy that loses to everything: it must get ~no weight.
    dom = np.full((4, 4), 0.5)
    dom[:3, :3] = rps
    dom[:3, 3], dom[3, :3] = 0.9, 0.1
    w = fictitious_play(dom, 20_000)
    assert w[3] < 0.01, w
    assert np.abs(w[:3] - 1 / 3).max() < 0.03, w

    near = pfsp(np.array([0.1, 0.5, 0.9]), "variance")
    assert int(np.argmax(near)) == 1 and near[1] > near[0]
    weak = pfsp(np.array([0.1, 0.5, 0.9]), "squared")
    assert weak[0] > weak[1] > weak[2]
    excluded = pfsp(np.array([0.5, 0.5, 0.5]), exclude=1)
    assert excluded[1] == 0 and abs(excluded.sum() - 1.0) < 1e-12

    record = serialise_runoff({3: np.float64(0.625), 1: 0.5})
    assert record == {"1": 0.5, "3": 0.625}
    assert json.loads(json.dumps(record)) == record

    # Every searched field's default lies inside its range, so encode() does not
    # silently clip and the round trip below is testing something real.
    cfg = Config()
    for name, lo, hi, _ in SPACE:
        assert lo <= _get(cfg, name) <= hi, (name, _get(cfg, name), lo, hi)

    # Config -> [0,1] -> Config is the identity.
    back = decode(encode(cfg), cfg)
    n0, v0 = cfg.flatten()
    n1, v1 = back.flatten()
    assert n0 == n1
    bad = [(n, a, b) for n, a, b in zip(n0, v0, v1) if abs(a - b) > 1e-9]
    assert not bad, bad
    for name, _, _, kind in SPACE:
        if kind in (int, bool):
            assert getattr(back, name) == getattr(cfg, name), name
            assert type(getattr(back, name)) is kind, name

    # Payoff bookkeeping.
    P = [[None] * 3 for _ in range(3)]
    set_pair(P, 0, 1, 0.7)
    set_pair(P, 0, 2, 0.5)
    check_antisymmetry(P)
    assert abs(P[1][0] - 0.3) < 1e-9
    M = dense(P)
    assert np.allclose(np.diag(M), 0.5)
    assert np.allclose(M + M.T, 1.0)
    grow(P, 4)                                       # a new member joins
    assert [len(r) for r in P] == [4, 4, 4, 4]
    assert P[0][1] == 0.7 and P[0][3] is None        # cached entries survive
    check_antisymmetry(P)

    alloc = allocate(np.array([0.5, 0.3, 0.2]), 48)
    assert sum(alloc) == 48 and all(g % 2 == 0 for g in alloc), alloc
    assert allocate(np.array([1.0, 0.0]), 8) == [8, 0]

    # A group is a strict subspace and round-trips the same way the full one does.
    sub = select_space("commit", None)
    assert len(sub) == len(GROUPS["commit"]) < len(SPACE)
    _, v2 = decode(encode(cfg, sub), cfg, sub).flatten()
    assert not [n for n, a, b in zip(n0, v0, v2) if abs(a - b) > 1e-9]

    # The pool split IS the hold-out guarantee: no board may reach two slices, and
    # a pool too small to supply the requested games must refuse to run.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        pool = d / "pool.json"
        pool.write_text(json.dumps({"grids": [[[i]] for i in range(30)]}))
        slices = [{tuple(g[0]) for g in json.loads(Path(p).read_text())["grids"]}
                  for p in split_pool(str(pool), d, 20, 20)]
        assert sum(len(s) for s in slices) == 30, slices
        assert not (slices[0] & slices[1] or slices[0] & slices[2] or slices[1] & slices[2])
        try:
            split_pool(str(pool), d, 48, 48)     # 24 boards wanted, 10 per slice
        except SystemExit:
            pass
        else:
            raise AssertionError("split_pool accepted a pool too small to split")

    print(f"selfcheck ok: {len(SPACE)} searched fields, {len(sub)} in --group commit")


def probe(args, out: Path, space: list, omaps: str | None, rmaps: str | None) -> None:
    """The cheap version of the whole question, before anyone books a night.

    One oracle call whose mixture is a point mass on the known over-committer,
    then the exploiter it built is measured on runoff boards against that target
    AND against a control that does not over-commit. Only the DIFFERENCE means
    anything: a config that beats v12 and v16 equally is just a better config, and
    a league built on it will grade config quality while saying nothing about the
    diagnosis. Cost is one oracle call, roughly an hour at the default budget,
    against ~31k games for the full run.
    """
    for spec in (args.probe, args.control):
        path = spec.partition(":")[2]
        if spec.startswith("ours:") and not Path(path).exists():
            raise SystemExit(f"config not found: {path}")
    base = Config.load(args.base) if args.base else Config()
    print(f"probe: best response to {args.probe} over {len(space)} knobs")
    cand, score = oracle(base, [args.probe], np.array([1.0]), space=space,
                         pop=args.pop, elite=args.elite, gens=args.gens,
                         games=args.games, workers=args.workers,
                         max_turns=args.max_turns, maps=omaps,
                         tmpdir=out / "candidates", seed=args.seed,
                         spread=args.spread, floor=args.floor)
    dest = out / "probe.json"
    cand.save(dest)

    g = args.runoff_games
    a, b = (score_of(r) for r in run_many(
        [(f"ours:{dest.resolve()}", args.probe, g, HOLDOUT_SEED0),
         (f"ours:{dest.resolve()}", args.control, g, HOLDOUT_SEED0)],
        args.workers, args.max_turns, maps=rmaps))
    se, dse = stderr(g), stderr(g) * 2 ** 0.5
    print(f"\n  search score {score:.3f} (not a measurement)")
    print(f"  exploiter vs {args.probe:<28}{a:.3f} +-{se:.3f}")
    print(f"  exploiter vs {args.control:<28}{b:.3f} +-{se:.3f}")
    print(f"  gap {a - b:+.3f} +-{dse:.3f}")
    if a - b > 2 * dse and a > 0.5:
        print("  -> config space contains a punisher of the over-committer "
              "SPECIFICALLY. The league is worth the night.")
    elif a > 0.55:
        print("  -> beats both about equally: a better config, not a punisher. The "
              "league would measure config quality, which the ladder already "
              "disagrees with. Fix the seed archive or the search space first.")
    else:
        print("  -> the oracle beat nothing. It cannot search this space at this "
              "budget, so a league built on it is an expensive uniform sampler. "
              "Raise --pop/--gens/--games or shrink --group before spending a night.")
    print(f"wrote {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="runs/league", help="run directory (checkpoint lives here)")
    ap.add_argument("--out", default=None, help="where to write the max-min config")
    ap.add_argument("--resume", action="store_true", help="continue from the checkpoint")
    ap.add_argument("--seeds", default="hunter:1,hunter:3,greedy,expander,"
                                       "ours:configs/v16.json,ours:configs/v12.json,"
                                       "ours:configs/v6.json",
                    help="seed archive; keep non-ours members so the league is not internal")
    ap.add_argument("--clone", default=None, help="weights .npz to add as clone:<path>")
    ap.add_argument("--base", default=None, help="config the oracle starts from "
                                                 "(default: the current max-min member)")
    ap.add_argument("--iters", type=int, default=6, help="PSRO iterations")
    ap.add_argument("--gens", type=int, default=4, help="CEM generations per oracle call")
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--elite", type=int, default=5)
    ap.add_argument("--games", type=int, default=48, help="games per candidate, split across sigma")
    ap.add_argument("--pair-games", type=int, default=48, help="games per payoff-matrix entry")
    ap.add_argument("--runoff-games", type=int, default=48,
                    help="fresh games per pairing in the final runoff; what makes "
                         "it unbiased is the fresh boards, not the count, and "
                         "raising it above --pair-games needs a bigger --maps pool")
    ap.add_argument("--top", type=int, default=3, help="members entering the runoff")
    ap.add_argument("--oracle", default="config", choices=("config", "net", "both"),
                    help="which best response to search for. config: CEM over the "
                         "config space, unchanged. net: PPO over network weights "
                         "(learn/netoracle.py), falling back to the CEM oracle for "
                         "that iteration if the net fails its gate, so the archive "
                         "always grows. both: append both. net/both need --nn-init")
    ap.add_argument("--nn-init", default=None,
                    help="trained policy .npz the neural oracle starts from and "
                         "anchors to; use the incumbent/accepted curriculum "
                         "candidate, never random weights")
    ap.add_argument("--nn-init-critic", default=None,
                    help="gated critic passed to every neural oracle warm start")
    ap.add_argument("--nn-iters", type=int, default=200)
    ap.add_argument("--nn-games", type=int, default=256, help="rollout games per PPO iteration")
    ap.add_argument("--nn-epochs", type=int, default=1,
                    help="passes over each neural-oracle rollout buffer")
    ap.add_argument("--nn-minibatch", type=int, default=4096)
    ap.add_argument("--nn-lr", type=float, default=1e-4,
                    help="neural-oracle policy learning rate")
    ap.add_argument("--nn-critic-lr", type=float, default=1e-4,
                    help="neural-oracle critic learning rate. 1e-3 memorised "
                         "the buffer in measured full-distance runs")
    ap.add_argument("--nn-critic-replay-games", type=int, default=0,
                    help="historical neural-oracle episodes retained by its critic")
    ap.add_argument("--nn-critic-replay-frac", type=float, default=0.5)
    ap.add_argument("--nn-critic-replay-source-floor", type=float, default=0.25)
    ap.add_argument("--nn-value-hidden", type=int, default=64,
                    help="neural-oracle residual critic-head width; 0 is linear")
    ap.add_argument("--nn-lam", type=float, default=0.95,
                    help="GAE lambda forwarded to the neural oracle")
    ap.add_argument("--nn-reward-mode", choices=NN_REWARD_MODES, default="terminal",
                    help="neural-oracle reward: terminal for the ladder objective; "
                         "tempo for an explicit fast-win/slow-loss exploiter")
    ap.add_argument("--nn-tempo-eps", type=float, default=NN_TEMPO_EPS,
                    help="tempo tie-break strength forwarded to netoracle")
    ap.add_argument("--nn-augment", action="store_true",
                    help="enable the neural oracle's PPO dihedral arm")
    ap.add_argument("--nn-warm-evar", type=float, default=0.10,
                    help="keep the neural-oracle policy frozen until its critic "
                         "reaches this explained variance")
    ap.add_argument("--nn-sigma-floor", type=float, default=0.15,
                    help="uniform mass mixed into sigma for the neural oracle's "
                         "TRAINING opponents. `support 1` in the table above means "
                         "the default still sends 85%% of its games to one bot, "
                         "which is frozen-opponent PPO; netoracle prints the "
                         "effective opponent count at startup")
    ap.add_argument("--nn-role",
                    choices=("main", "main-exploiter", "league-exploiter"),
                    default="league-exploiter")
    ap.add_argument("--nn-sampling", choices=("nash", "pfsp"), default="nash")
    ap.add_argument("--nn-pfsp-weighting", choices=PFSP_WEIGHTINGS,
                    default="variance")
    ap.add_argument("--nn-no-fallback", action="store_true",
                    help="if a neural oracle is rejected, do not spend this "
                         "iteration on the config CEM fallback; use for a pure "
                         "neural pilot whose hypothesis excludes config tuning")
    ap.add_argument("--group", default="commit", choices=sorted(GROUPS),
                    help="which knobs the oracle searches (see GROUPS)")
    ap.add_argument("--params", default=None, help="explicit comma-separated knob list")
    ap.add_argument("--spread", type=float, default=0.25, help="initial CEM std, normalised units")
    ap.add_argument("--floor", type=float, default=0.05, help="CEM std floor")
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--max-turns", type=int, default=rules.TURN_LIMIT)
    ap.add_argument("--maps", default=None, help="real-board pool from `analysis.official maps`")
    ap.add_argument("--probe", nargs="?", const="ours:configs/v12.json", default=None,
                    help="skip the league: one oracle call against this spec alone, "
                         "to find out whether a night on the cluster is worth it")
    ap.add_argument("--control", default="ours:configs/v16.json",
                    help="--probe control: the exploiter must beat --probe by MORE "
                         "than it beats this, or it is just a better config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true", help="verify the solver and codec, no games")
    args = ap.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    if args.nn_value_hidden < 0:
        raise SystemExit("--nn-value-hidden must be zero or positive")
    if not 0.0 <= args.nn_tempo_eps < 1.0:
        raise SystemExit(f"--nn-tempo-eps must be in [0, 1), got {args.nn_tempo_eps}")

    if args.oracle != "config":
        if not args.nn_init:
            raise SystemExit(f"--oracle {args.oracle} needs --nn-init <policy.npz>")
        if not args.nn_init_critic:
            raise SystemExit(f"--oracle {args.oracle} needs --nn-init-critic "
                             "matching --nn-init; policy-only continuation is "
                             "not a safe PPO warm start")
        if not Path(args.nn_init).exists():
            raise SystemExit(f"--nn-init not found: {args.nn_init}")
        if args.nn_init_critic and not Path(args.nn_init_critic).exists():
            raise SystemExit(f"--nn-init-critic not found: {args.nn_init_critic}")

    out = Path(args.dir)
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    ckpt = out / "league.json"
    space = select_space(args.group, args.params)
    omaps, mmaps, rmaps = split_pool(args.maps, out,
                                     0 if args.probe else args.pair_games,   # no matrix
                                     max(args.games, args.runoff_games))

    if args.probe:
        probe(args, out, space, omaps, rmaps)
        return

    # What a payoff entry MEANS is fixed by these; a resume that changes one of
    # them mixes entries measured on different boards or budgets into one matrix,
    # and check_antisymmetry cannot see it because each entry is self-consistent.
    params = {k: getattr(args, k) for k in (
        "pair_games", "games", "runoff_games", "maps", "max_turns",
        "oracle", "group", "params", "gens", "pop", "elite", "spread",
        "floor", "base", "seed", "nn_init", "nn_init_critic", "nn_iters",
        "nn_games", "nn_epochs", "nn_minibatch", "nn_lr",
        "nn_critic_lr", "nn_critic_replay_games", "nn_critic_replay_frac",
        "nn_critic_replay_source_floor", "nn_value_hidden", "nn_lam", "nn_reward_mode",
        "nn_tempo_eps", "nn_augment",
        "nn_warm_evar", "nn_sigma_floor",
        "nn_role", "nn_sampling", "nn_pfsp_weighting",
        "nn_no_fallback")}

    if args.resume:
        if not ckpt.exists():
            raise SystemExit(f"no checkpoint at {ckpt}")
        state = json.loads(ckpt.read_text())
        check_antisymmetry(state["payoff"])
        original_params = state.get("params")
        saved_params = migrate_resume_params(original_params, params)
        # `nn_lam` was historically a hardcoded 0.95 inside the neural oracle.
        # Permit an old checkpoint to acquire only that exact explicit identity;
        # a non-default value is a different experiment and must start a new dir.
        if saved_params != params:
            raise SystemExit(f"checkpoint was made with {original_params}, "
                             f"you passed {params}; the payoff entries are not "
                             f"comparable. Use the original flags or a new --dir.")
        if saved_params != original_params:
            state["params"] = saved_params
        print(f"resumed at iteration {state['iter']} with {len(state['archive'])} members")
    else:
        specs = [s for s in args.seeds.split(",") if s]
        if args.clone:
            specs.append(f"clone:{args.clone}")
        if len(specs) < 2:
            raise SystemExit("--seeds needs at least two members: with one, every "
                             "member's `min` is over an empty set")
        for s in specs:
            name, _, arg = s.partition(":")
            # `clone:` was never checked here, so a missing .npz used to surface
            # as a worker exception hours into the run.
            if name in ("ours", "clone") and arg and not Path(arg).exists():
                raise SystemExit(f"seed {name} file not found: {arg}")
        state = {"iter": 0, "pending": None, "params": params,
                 "archive": [{"name": s, "spec": s, "config": None} for s in specs],
                 "payoff": [[None] * len(specs) for _ in specs]}
    materialise(state["archive"], out / "archive")
    save_state(ckpt, state)

    base = Config.load(args.base) if args.base else None

    while state["iter"] < args.iters:
        it = state["iter"]
        print(f"\n=== iteration {it} ===")
        complete_matrix(state, args.pair_games, args.workers, args.max_turns,
                        mmaps, ckpt)
        sigma = fictitious_play(dense(state["payoff"]))
        champ, _ = report(state, sigma, args.pair_games)

        # Start the oracle from a config that already plays well: the current
        # max-min member if it has one, otherwise the defaults.
        start = base
        if start is None:
            cfg_of = state["archive"][champ].get("config")
            if cfg_of is not None:
                start = Config.from_dict(cfg_of)
            elif state["archive"][champ]["spec"].startswith("ours:"):
                start = Config.load(state["archive"][champ]["spec"][5:])
            else:
                start = Config()

        # Snapshot before any append: sigma is indexed by THIS list, and in
        # `both` mode the neural member lands in the archive first.
        specs_now = [m["spec"] for m in state["archive"]]
        # In `both` mode the CEM phase runs after the append and checkpoints
        # inside itself, so a kill there leaves oracle-nn-{it} on disk with
        # `iter` still at {it}. Re-running the net oracle would then overwrite
        # the very .npz the matrix entries were just measured against and append
        # a second member with the same name -- individually self-consistent, so
        # check_antisymmetry cannot see it, which is the failure class the
        # `params` guard above exists to prevent.
        done = {m["name"] for m in state["archive"]}
        accepted = f"oracle-nn-{it}" in done
        if args.oracle in ("net", "both") and not accepted:
            npz, gate = net_oracle(args, out, ckpt, omaps, it)
            accepted = bool(gate["accepted"])
            if accepted:
                # config None, so materialise skips it; complete_matrix, report and
                # runoff read only `name` and `spec`. The name still starts with
                # "oracle", which is what report's falsify line tests.
                state["archive"].append({"name": f"oracle-nn-{it}",
                                         "spec": f"clone:{npz}", "config": None,
                                         "critic": str(npz.with_suffix('.critic.npz'))})
                save_state(ckpt, state)      # member and weights land together
                print(f"  oracle-nn-{it} accepted: {gate['score']:.3f} vs its own "
                      f"initialisation {gate['init_score']:.3f} on gate boards")
            else:
                print(f"  oracle-nn-{it} REJECTED at the gate ({gate['score']:.3f} vs "
                      f"init {gate['init_score']:.3f}); PPO did not beat the policy it "
                      f"started from, so it is not a best response")

        # `net` normally falls back to CEM when rejected. A hypothesis-isolation
        # pilot may disable that explicitly so config tuning cannot masquerade
        # as evidence for the neural opponent-training experiment.
        run_config = (args.oracle in ("config", "both")
                      or (not accepted and not args.nn_no_fallback))
        if run_config:
            if state["pending"] is None:
                state["pending"] = {}
                save_state(ckpt, state)
            print(f"\n  oracle: best response to sigma over {len(specs_now)} members")
            cand, score = oracle(
                start, specs_now, sigma, space=space,
                pop=args.pop, elite=args.elite, gens=args.gens, games=args.games,
                workers=args.workers, max_turns=args.max_turns, maps=omaps,
                tmpdir=out / "candidates", seed=args.seed + 1000 * it,
                spread=args.spread, floor=args.floor, it=it,
                state=state["pending"], on_gen=lambda: save_state(ckpt, state))

            name = f"oracle-{it}"
            print(f"  {name} search score {score:.3f} -- a selection statistic, not a "
                  f"measurement; the matrix re-plays it on held-out boards next")
            state["archive"].append({"name": name, "spec": "", "config": asdict(cand)})
        elif not accepted:
            print("  config fallback disabled; this iteration appends no member")
        materialise(state["archive"], out / "archive")
        state["iter"] = it + 1
        state["pending"] = None
        save_state(ckpt, state)

    complete_matrix(state, args.pair_games, args.workers, args.max_turns, mmaps, ckpt)
    sigma = fictitious_play(dense(state["payoff"]))
    champ, mins = report(state, sigma, args.pair_games)
    champ, fresh = runoff(state, mins, args.top, args.runoff_games, args.workers,
                          args.max_turns, rmaps)
    winner = state["archive"][champ]
    print(f"\nmax-min: {winner['name']}  worst case {fresh[champ]:.3f} "
          f"+-{stderr(args.runoff_games):.3f} on runoff boards "
          f"(matrix said {mins[champ]:.3f})")

    dest = Path(args.out) if args.out else out / "best.json"
    dest.parent.mkdir(parents=True, exist_ok=True)

    def write_manifest(result: Path) -> None:
        from tools import manifest
        inputs = [ckpt, result]
        for item in (args.nn_init, args.nn_init_critic, args.maps):
            if item:
                inputs.append(item)
        manifest.write(result.with_suffix(".manifest.json"), command=sys.argv,
                       artifacts=inputs,
                       extra={"kind": "psro-league", "args": vars(args),
                              "winner": winner,
                              "fresh_min": serialise_runoff(fresh),
                              "sigma": sigma.tolist(), "params": params})

    if winner.get("config") is not None:
        Config.from_dict(winner["config"]).save(dest)
    elif winner["spec"].startswith("ours:"):
        Config.load(winner["spec"][5:]).save(dest)
    elif winner["spec"].startswith("clone:"):
        # Without this a neural max-min fell through to the NOTE branch below and
        # wrote a DIFFERENT member's config as the answer.
        wdest = dest.with_suffix(".npz")
        shutil.copy(winner["spec"][6:], wdest)
        print(f"wrote {wdest} -- the max-min member is a network, not a config. "
              f"Copy it to bot/weights.npz and package it only after the "
              f"promotion suite passes.")
        print("Confirm it first:")
        print(f"  python -m arena.runner --a clone:{wdest} --b ours:configs/v16.json "
              f"--games 400 --workers {args.workers}")
        write_manifest(wdest)
        return
    else:
        # A hand-written baseline with the best worst case is itself a result:
        # nothing the oracle built survives contact with the whole archive.
        best_ours = max((i for i, m in enumerate(state["archive"])
                         if m.get("config") is not None or m["spec"].startswith("ours:")),
                        key=lambda i: mins[i], default=None)
        if best_ours is None:
            raise SystemExit(f"max-min is {winner['name']} and no config member exists")
        m = state["archive"][best_ours]
        print(f"NOTE: max-min is {winner['name']}, not a config. Writing the best "
              f"config member instead: {m['name']} (min {mins[best_ours]:.3f})")
        (Config.from_dict(m["config"]) if m.get("config")
         else Config.load(m["spec"][5:])).save(dest)
    print(f"wrote {dest}")
    write_manifest(dest)
    print("Confirm before submitting:")
    print(f"  python -m arena.runner --a ours:{dest} --b ours:configs/v16.json "
          f"--games 400 --workers {args.workers}")


if __name__ == "__main__":
    main()
