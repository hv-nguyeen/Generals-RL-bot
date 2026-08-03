"""Where our bot disagrees with strong players, on the same position.

Every other tool here measures outcomes. This measures decisions: replay a field
game, and at each tick hand our controller the exact fogged observation the
strong player had, then compare what it chose to what they chose.

That turns 2,700 replays into a ranked list of systematic differences — "in this
situation they capture and we reinforce" — which is directly actionable, needs
no ladder, and has no noise floor to fight.

    python -m analysis.disagree /local/data/vng205/field --players ResBot,Kubic --limit 60
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np

from analysis.actions import _states_from, infer
from analysis.official import read_replay, replay_files
from bot import rules
from bot.board import DIRS
from bot.config import Config
from bot.obs import Obs
from bot.policy.controller import Controller
from sim import engine


def classify(obs: Obs, action, general) -> str:
    """A coarse label for what a move is *for*."""
    kind = int(action[0])
    if kind == rules.BUILD:
        return "build"
    if kind != rules.MOVE:
        return "pass"
    r, c, d = int(action[1]), int(action[2]), int(action[3]) % 4
    nr, nc = r + DIRS[d][0], c + DIRS[d][1]
    if not (0 <= nr < obs.H and 0 <= nc < obs.W):
        return "illegal"
    owner = int(obs.owner_grid[nr, nc])
    from_gen = (r, c) == general
    if owner == rules.OWNER_OPP:
        base = "attack"
    elif owner == rules.OWNER_ME:
        base = "reinforce"
    else:
        base = "expand"
    return base + ("_from_general" if from_gen else "")


def phase(turn: int) -> str:
    if turn < 50:
        return "0-50"
    if turn < 100:
        return "50-100"
    if turn < 200:
        return "100-200"
    if turn < 400:
        return "200-400"
    return "400+"


# Action inference is by far the expensive part and does not depend on the
# config, so keep it: sweeping weights against the agreement metric then costs
# only the controller's own decisions.
_CACHE: dict = {}


def positions(src: Path, players: set[str] | None, limit: int):
    key = (str(src), tuple(sorted(players)) if players else None, limit)
    if key in _CACHE:
        return _CACHE[key]
    out = []
    games = 0
    for f in replay_files(src):
        if limit and games >= limit:
            break
        rep = read_replay(f)
        seats = [i for i, n in enumerate(rep["players"])
                 if players is None or n in players]
        if not seats:
            continue
        acts, _ = infer(rep)
        states, _ = _states_from(rep)
        games += 1
        for seat in seats:
            general = tuple(int(v) for v in rep["generals"][seat])
            frames = []
            for t, pair in enumerate(acts):
                if pair is None or t + 1 >= len(states):
                    continue
                frames.append((t, engine.observe(states[t], seat), pair[seat]))
            out.append((seat, general, states[0].armies.shape, frames))
    _CACHE[key] = (out, games)
    return _CACHE[key]


def run(src: Path, players: set[str] | None, limit: int, cfg: Config) -> None:
    exact = same_src = same_dst = same_kind = total = 0
    by_phase: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    confusion = collections.Counter()
    theirs_when_differ = collections.Counter()
    ours_when_differ = collections.Counter()
    mode_all = collections.Counter()
    mode_bad = collections.Counter()
    tracks, games = positions(src, players, limit)

    for seat, general, (h, w), frames in tracks:
            ctrl = Controller(seat, h, w, cfg)
            for t, obs, expert in frames:
                try:
                    mine = ctrl.act(obs)
                except Exception:                        # noqa: BLE001
                    continue
                total += 1
                mode = (getattr(ctrl, "last_debug", {}) or {}).get("mode", "?")
                mode_all[mode] += 1
                ph = phase(t)
                by_phase[ph][1] += 1

                em = tuple(int(v) for v in expert)
                om = tuple(int(v) for v in mine)
                if em == om:
                    exact += 1
                    by_phase[ph][0] += 1
                else:
                    mode_bad[mode] += 1
                if em[0] == om[0] == rules.MOVE:
                    if em[1:3] == om[1:3]:
                        same_src += 1
                    ed = (em[1] + DIRS[em[3] % 4][0], em[2] + DIRS[em[3] % 4][1])
                    od = (om[1] + DIRS[om[3] % 4][0], om[2] + DIRS[om[3] % 4][1])
                    if ed == od:
                        same_dst += 1
                ek = classify(obs, expert, general)
                ok = classify(obs, mine, general)
                if ek == ok:
                    same_kind += 1
                else:
                    confusion[(ek, ok)] += 1
                    theirs_when_differ[ek] += 1
                    ours_when_differ[ok] += 1

    if not total:
        raise SystemExit("no comparable positions — check --players")

    print(f"{games} games, {total} positions where the expert move was recovered\n")
    print(f"  exact same action        {100.0 * exact / total:5.1f}%")
    print(f"  same source tile         {100.0 * same_src / total:5.1f}%")
    print(f"  same destination tile    {100.0 * same_dst / total:5.1f}%")
    print(f"  same kind of move        {100.0 * same_kind / total:5.1f}%")

    print("\nagreement by phase")
    for ph in ("0-50", "50-100", "100-200", "200-400", "400+"):
        hit, n = by_phase[ph]
        if n:
            print(f"  turn {ph:<9} {100.0 * hit / n:5.1f}%   ({n} positions)")

    print("\nwhich of our modes was deciding")
    print(f"  {'mode':<14}{'share':>8}{'agreement':>11}")
    for m, n in mode_all.most_common():
        agree = 100.0 * (n - mode_bad[m]) / n
        print(f"  {m:<14}{100.0 * n / total:7.1f}%{agree:10.1f}%")

    print("\nwhat they do that we do not (top disagreements)")
    print(f"  {'they played':<24}{'we played':<24}{'n':>7}")
    for (ek, ok), n in confusion.most_common(12):
        print(f"  {ek:<24}{ok:<24}{n:>7}")

    print("\nmove kinds, when we disagree")
    tot = sum(theirs_when_differ.values()) or 1
    for k in sorted(set(theirs_when_differ) | set(ours_when_differ),
                    key=lambda k: -theirs_when_differ[k]):
        print(f"  {k:<24} they {100.0 * theirs_when_differ[k] / tot:5.1f}%"
              f"   us {100.0 * ours_when_differ[k] / tot:5.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", default="/local/data/vng205/field")
    ap.add_argument("--players", default=None, help="comma-separated; default all")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    players = set(p.strip() for p in args.players.split(",")) if args.players else None
    cfg = Config.load(args.config) if args.config else Config()
    run(Path(args.src), players, args.limit, cfg)


if __name__ == "__main__":
    main()
