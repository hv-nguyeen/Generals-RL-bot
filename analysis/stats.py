"""Turning a pile of games into "what should I fix next".

The classifier is the point. A win rate tells you that you are worse; a loss
histogram tells you *how*, and that maps to a specific knob:

    early_rush     -> defend_margin / defend_horizon / first_expand_turn
    out_expanded   -> expand weights, castle_gather_min_army (too greedy)
    out_gathered   -> castle economy, gather_start_turn
    blundered      -> defence and the deathtouch guard: we were ahead and died
    timeout        -> time_budget_ms, or the move loop got slow
    draw           -> deathtouch_prep_turn, attack_margin (too timid)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from bot import rules


def load_results(path: str | Path) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        p = p / "results.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def classify(r: dict) -> str:
    """Why this game ended, from agent A's point of view."""
    a = r["a_seat"]
    b = 1 - a
    if r["faults"][a] >= rules.MAX_FAULTS:
        return "timeout"
    if r["a_result"] == "win":
        return "win"
    if r["a_result"] == "draw":
        return "draw"

    land_a, land_b = r["land"][a], r["land"][b]
    army_a, army_b = r["army"][a], r["army"][b]
    # Land/army at the end are post-transfer (the winner inherits), so read the
    # last sample before the game ended instead.
    if len(r["series"]) >= 2:
        s = r["series"][-2]
        land_a, land_b = (s[1], s[3]) if a == 0 else (s[3], s[1])
        army_a, army_b = (s[2], s[4]) if a == 0 else (s[4], s[2])

    if r["turns"] < 150:
        return "early_rush"
    if land_a < 0.6 * max(land_b, 1):
        return "out_expanded"
    if army_a < 0.6 * max(army_b, 1):
        return "out_gathered"
    return "blundered"


def aggregate(results: list[dict]) -> dict:
    causes = Counter(classify(r) for r in results)
    wins = causes["win"]
    draws = causes["draw"]
    losses = len(results) - wins - draws

    def seat_val(r, key):
        return r[key][r["a_seat"]]

    first_castles = [seat_val(r, "first_castle") for r in results]
    first_castles = [t for t in first_castles if t]
    turns = [r["turns"] for r in results]

    return {
        "games": len(results),
        "wins": wins, "draws": draws, "losses": losses,
        "causes": dict(causes),
        "mean_turns": sum(turns) / max(len(turns), 1),
        "mean_ms": sum(seat_val(r, "mean_ms") for r in results) / max(len(results), 1),
        "max_ms": max((max(r["max_ms"]) for r in results), default=0.0),
        "faults": sum(seat_val(r, "faults") for r in results),
        "castles": sum(seat_val(r, "castles") for r in results) / max(len(results), 1),
        "first_castle_turn": sum(first_castles) / max(len(first_castles), 1) if first_castles else None,
        "castle_games": len(first_castles),
        "by_size": _by_size(results),
        "curve": _mean_curve(results),
    }


def _by_size(results: list[dict]) -> dict:
    out: dict[str, list[int]] = {}
    for r in results:
        key = f"{r['h']}x{r['w']}"
        w = 1 if r["a_result"] == "win" else 0
        out.setdefault(key, [0, 0])
        out[key][0] += w
        out[key][1] += 1
    return {k: {"wins": v[0], "games": v[1], "rate": v[0] / v[1]} for k, v in sorted(out.items())}


def _mean_curve(results: list[dict]) -> list[list[float]]:
    """Mean [turn, our_land, their_land, our_army, their_army] over all games.

    Games that end early stop contributing, so late points average only the
    games that got there — which is what you want when reading an expansion
    curve, otherwise a fast win looks like a collapse.
    """
    acc: dict[int, list[float]] = {}
    for r in results:
        a = r["a_seat"]
        for s in r["series"]:
            t = s[0]
            # The runner also samples on the final turn, whatever number that is.
            # Mixing those one-off turns into the curve makes it jump; keep the
            # regular grid only.
            if t % 10:
                continue
            mine = (s[1], s[2]) if a == 0 else (s[3], s[4])
            theirs = (s[3], s[4]) if a == 0 else (s[1], s[2])
            row = acc.setdefault(t, [0.0, 0.0, 0.0, 0.0, 0.0])
            row[0] += mine[0]
            row[1] += theirs[0]
            row[2] += mine[1]
            row[3] += theirs[1]
            row[4] += 1
    return [[t, row[0] / row[4], row[1] / row[4], row[2] / row[4], row[3] / row[4]]
            for t, row in sorted(acc.items())]
