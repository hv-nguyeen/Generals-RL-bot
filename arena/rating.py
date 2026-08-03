"""Elo and SPRT.

The point of this module is to stop us celebrating noise. A heuristic tweak that
"looks better over 40 games" is usually nothing: the standard error on a 40-game
match is around 80 Elo. `sprt` answers the only question that matters — is B at
least `elo1` better than A — and tells you to keep playing when it cannot decide.
"""

from __future__ import annotations

import math


def score_to_elo(score: float) -> float:
    score = min(max(score, 1e-9), 1 - 1e-9)
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def summary(wins: int, draws: int, losses: int) -> dict:
    n = wins + draws + losses
    if n == 0:
        return {"n": 0, "score": 0.0, "elo": 0.0, "err": float("inf"),
                "wins": 0, "draws": 0, "losses": 0}
    score = (wins + 0.5 * draws) / n
    # Per-game variance of the score, then the 95% interval on the Elo scale.
    w, d = wins / n, draws / n
    var = w + d / 4.0 - score * score
    stderr = math.sqrt(max(var, 1e-12) / n)
    lo = score_to_elo(min(max(score - 1.96 * stderr, 1e-9), 1 - 1e-9))
    hi = score_to_elo(min(max(score + 1.96 * stderr, 1e-9), 1 - 1e-9))
    return {
        "n": n, "wins": wins, "draws": draws, "losses": losses,
        "score": score, "elo": score_to_elo(score),
        "elo_lo": lo, "elo_hi": hi, "err": (hi - lo) / 2.0,
    }


def llr(wins: int, draws: int, losses: int, elo0: float, elo1: float) -> float:
    """Log-likelihood ratio for H1(elo1) against H0(elo0), 3-outcome model."""
    n = wins + draws + losses
    if n == 0 or wins == 0 or losses == 0:
        return 0.0
    w, d = wins / n, draws / n
    score = w + d / 2.0
    var = w + d / 4.0 - score * score
    if var <= 0:
        return 0.0
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return n * (s1 - s0) * (2.0 * score - s0 - s1) / (2.0 * var)


def sprt(wins: int, draws: int, losses: int, elo0: float = 0.0, elo1: float = 12.0,
         alpha: float = 0.05, beta: float = 0.05) -> dict:
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    value = llr(wins, draws, losses, elo0, elo1)
    verdict = "continue"
    if value >= upper:
        verdict = "accept H1 (B is better)"
    elif value <= lower:
        verdict = "accept H0 (no improvement)"
    return {"llr": value, "lower": lower, "upper": upper, "verdict": verdict,
            "elo0": elo0, "elo1": elo1}
