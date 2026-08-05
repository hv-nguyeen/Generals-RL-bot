"""Agent registry.

A spec is a string so runs are reproducible from a command line:

    ours                     our bot with default weights
    ours:configs/foo.json    our bot with a saved config
    expander                 the competition's reference bot, ported verbatim
    greedy                   a competent simple bot; the realistic baseline
    hunter                   the starter kit's decapitation bot; the leaderboard grades on it
    random                   uniform over legal moves
    idle                     always passes
    clone:weights.npz        a behaviour-cloned policy (learn/train.py)
    guard:weights.npz        the same policy under bot/policy/guard.py's overrides
    snipe:weights.npz        that policy, but a big stack marches at the enemy
                             general — the decapitating opponent our lineage
                             lacks. The commit threshold is RANDOM per game so a
                             trainee cannot learn its clock, only the rule
    snipe:weights.npz@120    threshold pinned, for a reproducible measurement
    stdio:dist/x/run.sh      a packaged submission, over the real wire protocol

`guard:` against `clone:` on the same weights is the only way to measure the
guard on its own — via `stdio:` the packaging differences ride along with it.

`expander` is deliberately a faithful port, bug included: it treats
structure-in-fog (type 5) as impassable, when in fact only mountains are. That
is the baseline everyone who starts from the starter kit inherits.
"""

from __future__ import annotations

import random as _random

import numpy as np

from bot import rules
from bot.board import bfs_field
from bot.config import Config
from bot.obs import Obs
from bot.policy.controller import Controller

DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class Expander:
    """The reference agent from `competition/agents/expander_python/agent.py`."""

    def __init__(self, player_id: int, h: int, w: int):
        self.H, self.W = h, w

    def act(self, obs: Obs, deadline=None):
        best_score, best_move, first_valid = -1.0, None, None
        ty, ow, ar = obs.type_grid, obs.owner_grid, obs.army_grid
        for r in range(obs.H):
            for c in range(obs.W):
                if ow[r, c] != 1:
                    continue
                src = int(ar[r, c])
                if src <= 1:
                    continue
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < obs.H and 0 <= nc < obs.W):
                        continue
                    t = int(ty[nr, nc])
                    if t == 2 or t == 5:
                        continue
                    move = (0, r, c, d, 0)
                    if first_valid is None:
                        first_valid = move
                    dest_owner = int(ow[nr, nc])
                    dest_army = int(ar[nr, nc])
                    if src <= dest_army + 1:
                        continue
                    is_opp = dest_owner == 2
                    is_visible_neutral = dest_owner == 0 and t not in (0, 5)
                    score = float(src)
                    if is_opp or is_visible_neutral:
                        score *= 10.0
                    if is_opp:
                        score *= 2.0
                    if score > best_score:
                        best_score, best_move = score, move
        return best_move or first_valid or rules.PASS_ACTION


class Greedy:
    """A competent, obvious bot — the realistic median competitor.

    Flood-fills the distance to the nearest unowned cell and always plays the
    move that either takes ground or walks the biggest stack toward ground. No
    memory, no castles, no idea where the enemy general is. `expander` deadlocks
    on corner spawns and flatters us; this does not.
    """

    def __init__(self, player_id: int, h: int, w: int):
        self.H, self.W = h, w

    def act(self, obs: Obs, deadline=None):
        ty, ow, ar = obs.type_grid, obs.owner_grid, obs.army_grid
        passable = (ty != 2) & (ty != 5)
        mine = ow == 1
        unowned = passable & ~mine
        if not unowned.any():
            return rules.PASS_ACTION
        dist = bfs_field(passable, unowned)

        best, best_score = None, -1e18
        for r, c in np.argwhere(mine & (ar >= 2)):
            r, c = int(r), int(c)
            army = int(ar[r, c])
            for d, (dr, dc) in enumerate(DIRECTIONS):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < obs.H and 0 <= nc < obs.W) or not passable[nr, nc]:
                    continue
                dest_army = int(ar[nr, nc])
                if ow[nr, nc] != 1 and army - 1 > dest_army:
                    score = 1000.0 + army + 5.0 * dest_army
                else:
                    score = 100.0 * (int(dist[r, c]) - int(dist[nr, nc])) + army * 0.5
                if score > best_score:
                    best_score, best = score, (0, r, c, d, 0)
        return best or rules.PASS_ACTION


def _shift(arr, fill, s, ax):
    out = np.roll(arr, s, axis=ax)
    edge = 0 if s == 1 else -1
    if ax == 0:
        out[edge, :] = fill
    else:
        out[:, edge] = fill
    return out


class Hunter:
    """Port of `generals/agents/hunter_agent.py` from the starter kit.

    The leaderboard grades against this, and it is a different animal from
    `expander`: it garrisons its general, converts only the surplus into a single
    advancing stack, and walks that stack at our general to decapitate. It
    punishes exactly the shape we play — spread out, army parked in rear castles.
    """

    GARRISON = 4

    def __init__(self, player_id: int, h: int, w: int, garrison: int | None = None):
        self.H, self.W = h, w
        # Lower garrison = commits earlier and harder. `hunter:2` is a rush
        # proxy for the bots that decapitate us around turn 130.
        self.GARRISON = self.GARRISON if garrison is None else garrison

    def act(self, obs: Obs, deadline=None):
        a, ty, ow = obs.army_grid, obs.type_grid, obs.owner_grid
        h, w = obs.H, obs.W
        reach = h * w
        mine = ow == 1
        castles = ty == rules.T_CASTLE
        generals = ty == rules.T_GENERAL
        opp = ow == rules.OWNER_OPP
        passable = ~((ty == rules.T_MOUNTAIN) | (ty == rules.T_STRUCTURE_IN_FOG)
                     | (castles & ~mine))

        mine_army = np.where(mine, a, 0)
        movable = mine & (a > 1)
        gen = mine & generals
        gen_army = int(np.where(gen, a, 0).sum())
        g = int(np.argmax(gen.reshape(-1).astype(np.int32)))
        from_gen = bfs_field(passable, gen)

        # enemy general > nearest enemy land > farthest fog > farthest open ground
        egen = opp & generals
        enemy = opp & ~castles
        fog = (ty == rules.T_FOG) & passable & (from_gen < reach)
        open_ = passable & ~mine & (from_gen < reach)

        def farthest(m):
            return m & (from_gen == np.max(np.where(m, from_gen, -1)))

        if egen.any():
            goal = egen
        elif enemy.any():
            goal = enemy
        elif fog.any():
            goal = farthest(fog)
        else:
            goal = farthest(open_)
        if not goal.any():
            return rules.PASS_ACTION

        to_goal = bfs_field(passable, goal)
        big = reach + 7
        vals = np.stack([
            np.where(_shift(passable, False, s, ax), _shift(to_goal, big, s, ax), big)
            for s, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1))
        ])
        direction = np.argmin(vals, 0)
        advances = np.min(vals, 0) < to_goal
        dirn = direction.reshape(-1)

        egen_army = int(np.where(egen, a, 0).sum())
        kill = movable & (to_goal == 1) & advances & (a - 1 > egen_army)
        if not egen.any():
            kill = np.zeros_like(kill)
        fwd = movable & ~gen & advances

        do_kill = bool(kill.any())
        do_feed = (not do_kill) and gen_army >= 2 * self.GARRISON and bool(advances.reshape(-1)[g])
        do_conv = (not do_kill) and (not do_feed) and bool(fwd.any())
        if not (do_kill or do_feed or do_conv):
            return rules.PASS_ACTION

        if do_kill:
            i = int(np.argmax(np.where(kill, mine_army, -1).reshape(-1)))
        elif do_feed:
            i = g
        else:
            i = int(np.argmax(np.where(fwd, mine_army, -1).reshape(-1)))
        return (0, i // w, i % w, int(dirn[i]), 1 if do_feed else 0)


class Sniper:
    """A strong net for economy, with `hunter`'s decapitation bolted on.

    THE INSTRUMENT WE DID NOT HAVE. Ladder forensics on the ten shortest losses:
    at the tick our general is emptied, hidden enemy army exceeds our garrison in
    10 of 10 (by 1.2x to 7.7x) while the VISIBLE threat within 3 tiles is ~0 in 9
    of 10. The killer arrives 7-36 turns later. We lose to army we cannot see.

    Nothing in our lineage plays that way. Every clone descends from the same
    behaviour-cloned policy, none of them stockpile and snipe, so any change that
    fixes decapitation reads ~0.500 in the arena and ~0 in comp-eval. The fix was
    unmeasurable, not absent -- which is also why `bot/policy/guard.py` scored
    neutral over 2000 games.

    `hunter` does the right thing and is saturated at 0.979 because it is bad at
    everything else, so it discriminates nothing. This is the missing combination:
    the net plays the game, and once a stack passes `STACK_MIN` that stack marches
    at the enemy general and nothing distracts it.

    Deliberately NOT a submission and it lives here rather than in `bot/`: it is
    an opponent, and `bot/` must stay self-contained.
    """

    # Measured, 400 games per arm against the plain net on the same weights:
    #
    #   @30   +124.0                 a castle costs 35, so EVERY tile that could
    #                                build instead marched. built/game was 0.00.
    #   @60    +15.6 [-18.2, +49.8]
    #   @90     +5.2 [-28.5, +39.0]
    #   @120    -6.1 [-39.8, +27.5]
    #   @160    +5.2 [-28.4, +38.9]
    #
    # Anything from 60 up is a peer. Below the build cost it is an
    # army-donation machine. 120 is the most adversarial point estimate, though
    # picking the max of a noisy sweep is ~1 se optimistic and 90-160 are
    # indistinguishable -- the value of this opponent is its STYLE, not its Elo.
    STACK_MIN = 120

    # Sampled per game when the threshold is not pinned. A FIXED threshold is a
    # clock, and a policy trained against a clock learns the clock: "he commits
    # around turn 250, garrison then". That is the overfitting objection to any
    # scripted opponent and it is fatal for training. Randomised, the only stable
    # thing to learn is the RULE -- keep army home in proportion to enemy army
    # you cannot see -- which is what transfers to the ladder.
    STACK_RANGE = (60, 200)

    def __init__(self, player_id: int, h: int, w: int, weights: str,
                 stack_min: int | None = None, seed: int = 0):
        from bot.belief import Belief
        from bot.policy.net import ClonePolicy
        self.inner = ClonePolicy(player_id, h, w, weights)
        self.H, self.W = h, w
        if stack_min is not None:
            self.stack_min = stack_min
        else:
            lo, hi = self.STACK_RANGE
            self.stack_min = int(_random.Random(seed * 6151 + player_id).uniform(lo, hi))
        # The enemy general is fogged almost always. Marching at the farthest fog
        # is a wander; `Belief` keeps an actual prior, and no net policy uses it.
        self.belief = Belief(player_id, h, w)
        self.last_debug: dict = {}

    def act(self, obs: Obs, deadline=None):
        self.belief.update(obs)
        ty, ow, a = obs.type_grid, obs.owner_grid, obs.army_grid
        mine = ow == rules.OWNER_ME
        generals = ty == rules.T_GENERAL
        castles = ty == rules.T_CASTLE
        passable = ~((ty == rules.T_MOUNTAIN) | (ty == rules.T_STRUCTURE_IN_FOG)
                     | (castles & ~mine))

        # The strike stack is the biggest army NOT sitting on our own general:
        # emptying our own general to attack is the mistake we are trying to
        # punish, not commit.
        strike = mine & ~generals & (a >= self.stack_min)
        if strike.any():
            gen = mine & generals
            from_gen = bfs_field(passable, gen) if gen.any() else None
            egen = (ow == rules.OWNER_OPP) & generals
            guess = self.belief.general_guess
            if egen.any():
                goal = egen
            elif guess is not None and passable[guess]:
                # The prior, not the farthest fog. This is the difference between
                # a strike and a tourist.
                goal = np.zeros_like(mine)
                goal[guess] = True
            elif from_gen is not None:
                # Their general is in fog. Walk at the far side of the map: that
                # is where it is, and it keeps the stack out of our own territory
                # where it would be visible early.
                #
                # `reach` comes from the OBSERVATION, not from self.H/self.W:
                # boards are 18-21 rectangles, `bfs_field` marks unreachable as
                # obs.H * obs.W, and bounding by the padded 21*21 let unreachable
                # tiles through — the goal became a cell with no path, no
                # neighbour ever improved, and the march silently never happened.
                reach = obs.H * obs.W
                far = (ty == rules.T_FOG) & passable & (from_gen < reach)
                if not far.any():
                    far = passable & ~mine & (from_gen < reach)
                goal = far & (from_gen == np.max(np.where(far, from_gen, -1))) if far.any() else None
            else:
                goal = None

            if goal is not None and goal.any():
                to_goal = bfs_field(passable, goal)
                r, c = np.unravel_index(int(np.argmax(np.where(strike, a, -1))), a.shape)
                r, c = int(r), int(c)
                best, bestd = None, int(to_goal[r, c])
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < obs.H and 0 <= nc < obs.W) or not passable[nr, nc]:
                        continue
                    if int(to_goal[nr, nc]) < bestd:
                        bestd, best = int(to_goal[nr, nc]), (0, r, c, d, 0)
                if best is not None:
                    self.last_debug = {"mode": "snipe", "turn": obs.turn}
                    return best

        act = self.inner.act(obs, deadline)
        self.last_debug = {"mode": "net", "turn": obs.turn}
        return act


class RandomAgent:
    def __init__(self, player_id: int, h: int, w: int, seed: int = 0):
        self.rng = _random.Random(seed * 7919 + player_id)
        self.H, self.W = h, w

    def act(self, obs: Obs, deadline=None):
        moves = []
        for r, c in np.argwhere((obs.owner_grid == 1) & (obs.army_grid >= 2)):
            for d, (dr, dc) in enumerate(DIRECTIONS):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < obs.H and 0 <= nc < obs.W and obs.type_grid[nr, nc] != 2:
                    moves.append((0, int(r), int(c), d, 0))
        return self.rng.choice(moves) if moves else rules.PASS_ACTION


class Idle:
    def __init__(self, player_id: int, h: int, w: int):
        pass

    def act(self, obs: Obs, deadline=None):
        return rules.PASS_ACTION


def make(spec: str, player_id: int, h: int, w: int, seed: int = 0):
    name, _, arg = spec.partition(":")
    if name == "ours":
        cfg = Config.load(arg) if arg else Config()
        return Controller(player_id, h, w, cfg)
    if name == "expander":
        return Expander(player_id, h, w)
    if name == "greedy":
        return Greedy(player_id, h, w)
    if name == "hunter":
        return Hunter(player_id, h, w, int(arg) if arg else None)
    if name == "random":
        return RandomAgent(player_id, h, w, seed)
    if name == "idle":
        return Idle(player_id, h, w)
    if name == "clone":
        from bot.policy.net import ClonePolicy
        return ClonePolicy(player_id, h, w, arg)
    if name == "snipe":
        # `snipe:weights.npz` randomises the commit threshold per game; append
        # `@120` to pin it, which is what a reproducible measurement wants.
        path, _, k = arg.partition("@")
        return Sniper(player_id, h, w, path, int(k) if k else None, seed)
    if name == "guard":
        from bot.policy.guard import GuardedPolicy
        from bot.policy.net import ClonePolicy
        cfg = Config()          # radius/ratio track the tuned defaults, not literals
        return GuardedPolicy(ClonePolicy(player_id, h, w, arg),
                             cfg.general_block_radius, cfg.general_block_ratio)
    if name == "stdio":
        from arena.stdio_agent import StdioAgent
        return StdioAgent(arg, player_id, h, w)
    raise ValueError(f"unknown agent spec: {spec!r}")
