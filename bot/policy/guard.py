"""Hard overrides for the network, in the one place the network cannot learn.

`ClonePolicy` is argmax over masked logits and nothing else. The heuristic has a
whole tier above its move scorer — win-in-one, the deathtouch guard, the garrison
block — and the shipped net has none of it.

That gap is measured, not theoretical. `analysis/deathwatch.py` over 50 losses of
the neural bot on the ladder:

    16/50 losses: army left the general in the last 15 ticks with an enemy stack
                  within 3 steps. One trace goes 85 -> 2 with a 64-stack adjacent.
    mean garrison the tick before death: 27.7, against stacks of 30-90.

The heuristic needed three separate commits to stop doing exactly this
(`324daa0` the thrust, `fa1a5ca` the move scorer, `3b8871f` the strike stack).
The net has reinvented it, and it cannot unlearn it from self-play: at the
curriculum's short distances emptying the general is correct tempo, and in a
mirror match both sides do it, so nothing punishes it. See the symmetric-blindness
note in docs/ml-log.md.

So the fix belongs here rather than in training. Three overrides, cheapest and
safest first:

  1. WIN NOW. If a legal move captures the enemy general, take it. Rules-exact,
     no judgement, no downside.
  2. DEATHTOUCH. From turn 800 any move onto a general wins regardless of army,
     so (1) becomes far more available and must be checked before anything else.
     Same code path; the mask already encodes the rule.
  3. GARRISON VETO. Refuse a move that leaves the general when a visible enemy
     stack within `general_block_radius` out-sizes what would remain. This is the
     NARROW version — v16's `general_block_radius=3`, `general_block_ratio=0.5`.
     The wide version (an unconditional lock at radius 6) was measured at -265
     Elo on the ladder and must not be reintroduced.

The veto picks the net's next-best legal move rather than a hand-written
alternative: the network still chooses, it is only forbidden one option. That
keeps the policy's judgement everywhere the guard has no opinion.

The debug dict reports which override fired under the key `mode`, which is what
`arena/runner.py` already tallies -- a guard whose firing rate nobody can see is
indistinguishable from a guard that never fires, and the two demand opposite
fixes.
"""

from __future__ import annotations

import numpy as np

from bot import features, rules
from bot.obs import Obs

DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _enemy_general(obs: Obs):
    """Their general, if we can see it right now. No memory, no inference."""
    hit = np.argwhere((obs.type_grid == rules.T_GENERAL)
                      & (obs.owner_grid == rules.OWNER_OPP))
    return (int(hit[0][0]), int(hit[0][1])) if len(hit) else None


def _my_general(obs: Obs):
    hit = np.argwhere((obs.type_grid == rules.T_GENERAL)
                      & (obs.owner_grid == rules.OWNER_ME))
    return (int(hit[0][0]), int(hit[0][1])) if len(hit) else None


def winning_move(obs: Obs, mask: np.ndarray):
    """A legal move that captures their general this turn, or None.

    From `rules.DEATHTOUCH_TURN` any move onto a general wins whatever the
    armies are, so the army comparison is skipped rather than merely relaxed.
    """
    tgt = _enemy_general(obs)
    if tgt is None:
        return None
    tr, tc = tgt
    deathtouch = obs.turn >= rules.DEATHTOUCH_TURN
    need = int(obs.army_grid[tr, tc])
    for d, (dr, dc) in enumerate(DIRECTIONS):
        r, c = tr - dr, tc - dc                      # a tile that moves INTO the general
        if not (0 <= r < obs.H and 0 <= c < obs.W):
            continue
        if obs.owner_grid[r, c] != rules.OWNER_ME:
            continue
        src = int(obs.army_grid[r, c])
        if not deathtouch and src - 1 <= need:
            continue
        idx = features.action_to_index((rules.MOVE, r, c, d, 0))
        if idx is not None and mask[idx]:
            return idx
    return None


def drains_general(obs: Obs, idx: int, gen, radius: int, ratio: float) -> bool:
    """Would this move empty the general while something comparable is close?

    `ratio` is measured against the garrison BEFORE the move, matching v16's
    `general_block_ratio`. The threat scan is Chebyshev distance on visible enemy
    tiles only — no belief, no memory, because this file runs inside the
    submission and must stay numpy-only and cheap.
    """
    if gen is None or radius <= 0:
        return False
    act = features.index_to_action(idx)
    if act is None or act[0] != rules.MOVE:
        return False
    _, r, c, _, _ = act
    if (r, c) != gen:
        return False
    gr, gc = gen
    lo_r, hi_r = max(0, gr - radius), min(obs.H, gr + radius + 1)
    lo_c, hi_c = max(0, gc - radius), min(obs.W, gc + radius + 1)
    near = obs.army_grid[lo_r:hi_r, lo_c:hi_c]
    who = obs.owner_grid[lo_r:hi_r, lo_c:hi_c]
    enemy = near[who == rules.OWNER_OPP]
    if not enemy.size:
        return False
    return int(enemy.max()) >= ratio * max(int(obs.army_grid[gr, gc]), 1)


class GuardedPolicy:
    """A policy with the heuristic's hard-override tier bolted on.

    Wraps any agent exposing `act(obs, deadline)`. The wrapped policy still makes
    every decision the guard has no opinion about; the guard only forces a win it
    can prove and forbids one specific move it can prove is bad.
    """

    def __init__(self, inner, radius: int = 3, ratio: float = 0.5):
        self.inner = inner
        self.radius, self.ratio = radius, ratio
        self.fired = {"win": 0, "veto": 0, "turns": 0}
        self.last_debug: dict = {}

    def act(self, obs: Obs, deadline=None):
        self.fired["turns"] += 1
        mask = features.legal_mask(obs)
        if not mask.any():
            return rules.PASS_ACTION

        win = winning_move(obs, mask)
        if win is not None:
            self.fired["win"] += 1
            self.last_debug = {"mode": "guard-win", "turn": obs.turn}
            return features.index_to_action(win)

        # Ask the network, then veto. Scoring once and walking the order is what
        # keeps the net's judgement: the replacement is ITS second choice, not a
        # hand-written move that would quietly become a second policy.
        scored = getattr(self.inner, "net", None)
        if scored is None:
            act = self.inner.act(obs, deadline)
            self.last_debug = {"mode": "passthrough", "turn": obs.turn}
            return act

        logits = np.where(mask, scored.logits(obs), -np.inf)
        gen = _my_general(obs)
        order = np.argsort(-logits)
        for rank, idx in enumerate(order):
            if not np.isfinite(logits[idx]):
                break
            if drains_general(obs, int(idx), gen, self.radius, self.ratio):
                continue
            if rank:
                self.fired["veto"] += 1
            self.last_debug = {"mode": "guard-veto" if rank else "net",
                               "rank": rank, "turn": obs.turn}
            return features.index_to_action(int(idx))

        # Every legal move drains the general and something comparable is next to
        # it. Nothing here can save the position; take the net's first choice
        # rather than passing, which would hand over a free tempo.
        self.last_debug = {"mode": "guard-forced", "turn": obs.turn}
        return features.index_to_action(int(order[0]))


def selfcheck() -> None:
    from sim import engine, mapgen
    grid = mapgen.generate(7)
    st = engine.from_grid(grid)
    obs = engine.observe(st, 0)
    mask = features.legal_mask(obs)

    gen = _my_general(obs)
    assert gen is not None, "our general must be visible to us"
    # A move off the general with nothing nearby must NOT be vetoed.
    gr, gc = gen
    for d, (dr, dc) in enumerate(DIRECTIONS):
        nr, nc = gr + dr, gc + dc
        if not (0 <= nr < obs.H and 0 <= nc < obs.W):
            continue
        idx = features.action_to_index((rules.MOVE, gr, gc, d, 0))
        if idx is None or not mask[idx]:
            continue
        assert not drains_general(obs, idx, gen, 3, 0.5), \
            "vetoed with no enemy in sight"
        break

    # Plant a big enemy stack next to the general and it must veto.
    obs.owner_grid[gr, min(gc + 1, obs.W - 1)] = rules.OWNER_OPP
    obs.army_grid[gr, min(gc + 1, obs.W - 1)] = 999
    for d, (dr, dc) in enumerate(DIRECTIONS):
        idx = features.action_to_index((rules.MOVE, gr, gc, d, 0))
        if idx is not None:
            assert drains_general(obs, idx, gen, 3, 0.5), "should veto under threat"
            break
    # ...and a move from anywhere else must still be allowed.
    other = np.argwhere((obs.owner_grid == rules.OWNER_ME))
    other = [tuple(p) for p in other if tuple(p) != gen]
    if other:
        r, c = other[0]
        idx = features.action_to_index((rules.MOVE, int(r), int(c), 0, 0))
        assert idx is None or not drains_general(obs, idx, gen, 3, 0.5), \
            "only moves FROM the general may be vetoed"

    # Deathtouch: a 1-army tile beside their general wins from turn 800.
    st2 = engine.from_grid(mapgen.generate(11))
    o2 = engine.observe(st2, 0)
    o2.turn = rules.DEATHTOUCH_TURN
    eg = _enemy_general(o2)
    if eg is not None:
        er, ec = eg
        for d, (dr, dc) in enumerate(DIRECTIONS):
            r, c = er - dr, ec - dc
            if 0 <= r < o2.H and 0 <= c < o2.W:
                o2.owner_grid[r, c] = rules.OWNER_ME
                o2.army_grid[r, c] = 1
                o2.type_grid[r, c] = rules.T_PLAIN
                m2 = features.legal_mask(o2)
                got = winning_move(o2, m2)
                assert got is not None, "deathtouch win not taken"
                break
    print("guard selfcheck OK (win-in-one, deathtouch, narrow garrison veto)")


if __name__ == "__main__":
    selfcheck()
