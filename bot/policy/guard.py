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


def drains_general(obs: Obs, idx: int, gen, radius: int, ratio: float,
                   hidden_ratio: float = 0.0) -> bool:
    """Would this move empty the general while something dangerous is out there?

    TWO triggers, and the second one is the one that matters.

    The VISIBLE trigger — an enemy within `radius` out-sizing the garrison — is
    v16's `general_block_ratio` and it is nearly dead. Ladder forensics on the ten
    shortest losses, measured at the tick the general is EMPTIED rather than the
    tick it falls: the visible threat within 3 tiles is ~0 in 9 of 10, and this
    condition fires on 1 of 10. The killer arrives 7-36 turns LATER. Measuring at
    the death tick shows it adjacent every time, which is tautological and is how
    it looked correct for a day.

    The HIDDEN trigger looked obvious and is OFF BY DEFAULT because it measured
    catastrophic. Hidden army exceeds the garrison in 10 of 10 losses at the
    drain tick (1.2x to 7.7x; medians 191 against 38 in losses, 28 against 34 in
    wins), so it reads as a clean discriminator. It is not a usable trigger:

        hidden_ratio   veto fires    Elo vs the plain net, 300 games
        0 (visible)      0.23%      +5.8      <- default
        1.0             98.04%    -322.7
        2.0             46.63%    -332.8
        4.0             10.29%     -60.8

    At 1.0 the general can essentially never move. The condition holds at the
    drain tick in losses AND most of the time in every game including the wins --
    the forensics measured the signal at the moment of failure and never measured
    its FALSE-POSITIVE RATE across all the other states. A discriminator at one
    tick is not a trigger, and this is the fifth hand-written override in a day
    to measure neutral or negative against this policy.

    `opp_army` is the scoreboard total including their fogged tiles, so
    subtracting what we can see leaves the army we cannot. Both halves arrive on
    the wire every turn; this stays numpy-only and allocates nothing per call
    beyond the sum.
    """
    if gen is None:
        return False
    act = features.index_to_action(idx)
    if act is None or act[0] != rules.MOVE:
        return False
    _, r, c, _, _ = act
    if (r, c) != gen:
        return False
    gr, gc = gen

    garrison = max(int(obs.army_grid[gr, gc]), 1)
    if hidden_ratio > 0:
        visible_opp = int(obs.army_grid[obs.owner_grid == rules.OWNER_OPP].sum())
        hidden = max(int(obs.opp_army) - visible_opp, 0)
        if hidden >= hidden_ratio * garrison:
            return True
    if radius <= 0:
        return False
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

    def __init__(self, inner, radius: int = 3, ratio: float = 0.5,
                 hidden_ratio: float = 0.0):
        self.inner = inner
        self.radius, self.ratio = radius, ratio
        # Hidden army >= this multiple of the garrison vetoes leaving it. 0
        # disables, restoring the visible-only trigger that fires on 1 of 10
        # real death states.
        self.hidden_ratio = hidden_ratio
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

        # tta must come from the wrapped policy. This method re-scores instead of
        # calling `inner.act`, so a flag set on the inner policy reaches nothing
        # unless it is forwarded here -- and the SUBMISSION always wraps the net
        # in this class. Measured +32.8 Elo on a raw ClonePolicy and shipped as
        # exactly zero until this line passed it through.
        # BOTH flags, forwarded from the wrapped policy. This method re-scores
        # rather than calling `inner.act`, so anything set on the inner policy
        # reaches nothing unless it is passed here -- and the SUBMISSION always
        # wraps the net in this class. `tta` was dropped that way once and
        # shipped a measured +32.8 as exactly zero; `full` was dropped the same
        # way immediately after, which would have made the `ship8:` A/B measure
        # the four-element group on both sides and read ~0 for a change that was
        # never actually tested. If a new option is added to `Net.logits`, it
        # must be added here too.
        logits = np.where(mask, scored.logits(obs,
                                              tta=getattr(self.inner, "tta", False),
                                              full=getattr(self.inner, "full", False)),
                          -np.inf)
        gen = _my_general(obs)
        order = np.argsort(-logits)
        for rank, idx in enumerate(order):
            if not np.isfinite(logits[idx]):
                break
            if drains_general(obs, int(idx), gen, self.radius, self.ratio,
                              self.hidden_ratio):
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
