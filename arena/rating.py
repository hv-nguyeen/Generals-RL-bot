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


def pair_scores(rows: list[dict]) -> list[float]:
    """One independent board score from each swapped-seat game pair.

    The runner emits adjacent rows with the same seed and opposite A seats.
    Treating those two correlated games as independent produces invalid
    intervals in either direction, depending on the board/seat correlation.
    """
    if len(rows) % 2:
        raise ValueError("paired evaluation needs an even number of rows")
    value = {"win": 1.0, "draw": 0.5, "loss": 0.0}
    out = []
    for i in range(0, len(rows), 2):
        a, b = rows[i], rows[i + 1]
        if a.get("seed") != b.get("seed"):
            raise ValueError(f"rows {i}/{i + 1} do not share a board seed")
        if {a.get("a_seat"), b.get("a_seat")} != {0, 1}:
            raise ValueError(f"rows {i}/{i + 1} are not opposite A seats")
        try:
            out.append((value[a["a_result"]] + value[b["a_result"]]) / 2.0)
        except KeyError as e:
            raise ValueError(f"row has invalid a_result: {e}") from e
    return out


def paired_summary(rows: list[dict], alpha: float = 0.05) -> dict:
    """W/D/L point estimate with uncertainty clustered by board pair."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    wins = sum(r.get("a_result") == "win" for r in rows)
    draws = sum(r.get("a_result") == "draw" for r in rows)
    losses = len(rows) - wins - draws
    base = summary(wins, draws, losses)
    scores = pair_scores(rows)
    boards = len(scores)
    if not boards:
        return {**base, "boards": 0, "pair_stderr": float("inf")}
    score = sum(scores) / boards
    if abs(score - base["score"]) > 1e-12:
        raise AssertionError("paired and per-game score estimates disagree")
    if boards < 2:
        stderr = float("inf")
        radius = 1.0
        lo_score, hi_score = 1e-9, 1.0 - 1e-9
    else:
        var = sum((x - score) ** 2 for x in scores) / (boards - 1)
        stderr = math.sqrt(max(var, 0.0) / boards)
        # Empirical Bernstein interval for independent values in [0, 1]. Unlike
        # a plain cluster-normal interval it does not claim zero uncertainty
        # after a finite all-win sweep or perfectly anti-correlated WL pairs.
        log_term = math.log(3.0 / alpha)
        radius = (math.sqrt(2.0 * max(var, 0.0) * log_term / boards)
                  + 3.0 * log_term / boards)
        lo_score = min(max(score - radius, 1e-9), 1.0 - 1e-9)
        hi_score = min(max(score + radius, 1e-9), 1.0 - 1e-9)
    lo, hi = score_to_elo(lo_score), score_to_elo(hi_score)
    return {**base, "boards": boards, "elo_lo": lo, "elo_hi": hi,
            "err": (hi - lo) / 2.0, "pair_stderr": stderr,
            "score_radius": radius,
            "alpha": alpha,
            "confidence": 1.0 - alpha,
            "interval": f"{100.0 * (1.0 - alpha):g}% empirical Bernstein over board pairs"}


def paired_test(rows: list[dict], elo0: float = 0.0, elo1: float = 12.0,
                alpha: float = 0.05) -> dict:
    """Fixed-sample 95% decision using the board-clustered interval.

    This deliberately is not called an SPRT: the existing LLR assumes
    independent games, while evaluation uses fixed paired boards.
    """
    s = paired_summary(rows, alpha)
    lo, hi = elo_to_score(s["elo_lo"]), elo_to_score(s["elo_hi"])
    verdict = "continue"
    if lo >= elo_to_score(elo1):
        verdict = "accept H1 (A is better)"
    elif hi <= elo_to_score(elo0):
        verdict = "accept H0 (no improvement)"
    return {"method": "paired fixed-sample empirical Bernstein CI",
            "verdict": verdict,
            "elo0": elo0, "elo1": elo1, "boards": s["boards"],
            "alpha": alpha, "confidence": 1.0 - alpha,
            "score_lo": lo, "score_hi": hi}


def llr(wins: int, draws: int, losses: int, elo0: float, elo1: float) -> float:
    """Log-likelihood ratio for H1(elo1) against H0(elo0), 3-outcome model."""
    n = wins + draws + losses
    if n == 0:
        return 0.0
    w, d = wins / n, draws / n
    score = w + d / 2.0
    var = w + d / 4.0 - score * score
    if wins == 0 or losses == 0 or var <= 0:
        # The normal approximation needs BOTH outcomes present: it divides by the
        # sample variance, which goes to zero in a shutout and makes the ratio
        # explode. 39W-1D-0L read llr 54 that way -- "overwhelming", from a
        # variance of 0.006.
        #
        # Returning 0.0 instead, as this did until 2026-08-05, is the opposite
        # error: a 40-0 sweep produced no SPRT evidence at all.
        #
        # Count exactly instead. Each decisive game is one log-odds increment
        # between the hypotheses, and the honest answer for 40-0 against
        # H1 = 12 Elo is llr ~1.3 -- because 12 Elo predicts a 51.7% win rate,
        # under which a sweep is nearly as surprising as it is under 50%. A
        # shutout says the opponent is much weaker; it says little about whether
        # the edge is 12 Elo or 500.
        s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
        s0 = min(max(s0, 1e-9), 1.0 - 1e-9)
        s1 = min(max(s1, 1e-9), 1.0 - 1e-9)
        return wins * math.log(s1 / s0) + losses * math.log((1.0 - s1) / (1.0 - s0))
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return n * (s1 - s0) * (2.0 * score - s0 - s1) / (2.0 * var)


def sprt(wins: int, draws: int, losses: int, elo0: float = 0.0, elo1: float = 12.0,
         alpha: float = 0.05, beta: float = 0.05) -> dict:
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    value = llr(wins, draws, losses, elo0, elo1)
    verdict = "continue"
    if value >= upper:
        verdict = "accept H1 (A is better)"
    elif value <= lower:
        verdict = "accept H0 (no improvement)"
    return {"llr": value, "lower": lower, "upper": upper, "verdict": verdict,
            "elo0": elo0, "elo1": elo1}
