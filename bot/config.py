"""Every tunable number the bot has, in one place.

Two rules keep this useful:
  * the policy reads knobs from here and never hardcodes a constant;
  * the whole thing flattens to a float vector, so `tools/tune.py` can search it
    without knowing what any of the numbers mean.

Defaults below are hand-set from the rules analysis in the design doc. They are
a starting point for the tuner, not a claim of optimality.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

# Behaviour modes. The mode picks which distance field drives `progress` and
# which weight block scores moves.
EXPAND = "expand"
GATHER = "gather"
ATTACK = "attack"
DEFEND = "defend"
DEATHTOUCH = "deathtouch"
MODES = (EXPAND, GATHER, ATTACK, DEFEND, DEATHTOUCH)


@dataclass
class MoveWeights:
    """Linear weights over the move feature vector (see policy/features.py)."""

    cap_neutral: float = 0.0      # capture a visible neutral tile
    reveal: float = 0.0           # x (fogged cells this move uncovers) / 8
    cap_enemy: float = 0.0        # capture an enemy tile
    cap_enemy_army: float = 0.0   # x (army taken off the enemy) / 10
    cap_castle: float = 0.0       # dest is an enemy castle
    to_own: float = 0.0           # reinforcing a tile we already hold
    progress: float = 0.0         # x (mode_field[src] - mode_field[dst])
    army: float = 0.0             # x log1p(army actually moved)
    from_general: float = 0.0     # src is our general
    keep_home: float = 0.0        # x (-dist_home[dst] / diag)
    failed_capture: float = 0.0   # attack that cannot take the tile
    contact: float = 0.0          # dest touches a visible enemy tile
    frontier: float = 0.0         # dest touches unowned ground
    stack_break: float = 0.0      # x (fraction of the source army left behind)
    toward_enemy: float = 0.0     # x (steps gained toward the enemy general)


def _w(**kw) -> MoveWeights:
    return MoveWeights(**kw)


@dataclass
class Config:
    # ---- move scoring, one weight block per mode ----------------------------
    expand: MoveWeights = field(default_factory=lambda: _w(
        cap_neutral=10.0, reveal=3.0, cap_enemy=14.0, cap_enemy_army=2.0,
        cap_castle=40.0, to_own=-0.5, progress=6.0, army=1.5,
        from_general=-2.0, keep_home=0.3, failed_capture=-50.0,
        contact=1.0, frontier=2.0, stack_break=-4.0, toward_enemy=1.5,
    ))
    gather: MoveWeights = field(default_factory=lambda: _w(
        cap_neutral=3.0, reveal=1.0, cap_enemy=10.0, cap_enemy_army=2.0,
        cap_castle=40.0, to_own=1.0, progress=12.0, army=3.0,
        from_general=-1.0, keep_home=0.0, failed_capture=-50.0,
        contact=0.0, frontier=0.0, stack_break=-8.0, toward_enemy=0.0,
    ))
    attack: MoveWeights = field(default_factory=lambda: _w(
        cap_neutral=1.0, reveal=1.0, cap_enemy=8.0, cap_enemy_army=1.5,
        cap_castle=30.0, to_own=0.5, progress=16.0, army=4.0,
        from_general=-1.0, keep_home=0.0, failed_capture=-30.0,
        contact=2.0, frontier=0.0, stack_break=-8.0, toward_enemy=0.0,
    ))
    defend: MoveWeights = field(default_factory=lambda: _w(
        cap_neutral=0.5, reveal=0.0, cap_enemy=12.0, cap_enemy_army=3.0,
        cap_castle=5.0, to_own=1.0, progress=20.0, army=4.0,
        from_general=-8.0, keep_home=2.0, failed_capture=-60.0,
        contact=1.0, frontier=0.0, stack_break=-6.0, toward_enemy=-1.0,
    ))
    deathtouch: MoveWeights = field(default_factory=lambda: _w(
        cap_neutral=1.0, reveal=1.0, cap_enemy=6.0, cap_enemy_army=0.5,
        cap_castle=5.0, to_own=0.5, progress=25.0, army=1.0,
        from_general=-1.0, keep_home=0.0, failed_capture=-10.0,
        contact=3.0, frontier=0.0, stack_break=-2.0, toward_enemy=0.0,
    ))

    # ---- opening -----------------------------------------------------------
    # Hold the general's army until it is worth a real expansion run. A stack of
    # A army captures A-1 tiles before it runs dry, and leaving early means
    # re-walking to a frontier that has not moved.
    first_expand_turn: int = 30
    # Threat-conditional general lock. A STANDING garrison was rejected three
    # times - it pays tempo every turn for insurance that mostly is not needed.
    # This pays only while an enemy stack is actually within `lock_radius`: the
    # move scorer may not drain the general below what that stack would arrive
    # with. Without it the lethal guard judges the general safe and the scorer
    # spends the army the same turn, which is how real games were lost with 39
    # troops on the general and a 30-stack two tiles away.
    lock_enabled: bool = True
    lock_radius: int = 6
    # Narrow version of the same idea, and the one that is actually on. The old
    # lock blocked the general whenever anything was within lock_radius 6, which
    # is most of the midgame, and cost 265 elo. Real losses instead look like
    # this: general reinforced 10 -> 49, then the SCORER marches all 49 out on
    # the next turn with a 47-army stack adjacent, and we die three ticks later.
    # The thrust was stopped from doing that; the scorer was not. Block the
    # general as a move source only when something close is genuinely comparable
    # to the garrison.
    general_block_radius: int = 3
    general_block_ratio: float = 0.5

    # ---- castle economy ----------------------------------------------------
    # A castle is +0.5 army/turn forever for one turn of investment; expanding
    # 34 tiles buys +0.68 army/turn for 34 turns. Turns are the scarce resource.
    castle_enabled: bool = True
    castle_min_turn: int = 45          # cannot afford 35 army much before this
    castle_max_turn: int = 900         # past here it cannot pay itself back
    castle_max_cost: int = 35          # only build where the surcharge is zero
    max_castles: int = 8
    castle_min_land: int = 12          # do not stall the opening for a castle
    castle_safe_dist: int = 7          # min steps from the nearest seen enemy tile
    castle_keep: int = 2               # army left standing on a fresh castle
    # Only start walking army to a build site once we own this much army in
    # total. This is the single highest-leverage number in the file. At 75 the
    # condition is true almost every turn after turn 100, so GATHER took 72% of
    # the midgame and captured a tile on 6% of those turns, while EXPAND
    # captures on 80%. Raising it makes castles essentially opportunistic —
    # built when a stack already stands on a legal site — and recovered ~30 land
    # by turn 200 and +160 Elo against hunter. Do not lower it without rerunning
    # `python -m analysis.expansion ours`.
    castle_gather_min_army: int = 400

    # ---- defence -----------------------------------------------------------
    # Switch to DEFEND when a visible enemy stack out-races everything we could
    # bring home. >1 means paranoid.
    defend_margin: float = 1.15
    defend_horizon: int = 18           # only worry about stacks this close
    # Sticky: once defending, stay defending for this many turns so the bot does
    # not oscillate between running home and running away.
    defend_hold: int = 6
    # Army the general refuses to spend, as a fraction of the opponent's total
    # army (reported every turn, so this tracks force we cannot see). The general
    # regrows on its own, so this costs tempo only, not army. Routing army home
    # instead was measured and is much worse: it never releases and land
    # collapses. 0 disables.
    # Army the general refuses to spend, as a fraction of the opponent's total.
    # Real losses show the general sitting on 4 troops with 600 army on the board
    # and 28 enemy troops one tile away: the game is decided locally even when the
    # global count is winning. Coarse values (0.1+) cost too much land; this is
    # the cheap-insurance end of the range.
    # Measured harmful even at 0.05 (self-play A/B: 0.352, SPRT rejects). The
    # reasoning for it is sound and the real-game evidence is real, but the
    # tempo cost is not. Left as a knob, off.
    garrison_frac: float = 0.0
    # Any enemy stack within this many steps of our general is treated as urgent,
    # instead of waiting until it is adjacent.
    guard_radius: int = 3
    garrison_cap: int = 60             # never hoard more than this on the general
    garrison_from_turn: int = 40

    # Hold the corridor, not the doorstep. DEFEND's field is distance to our own
    # general, so it pulls army inward from every direction with no notion of
    # where the attack comes from. Measured over 60 real games, 51% of all enemy
    # presence within six steps of our general sat on just three cells - against
    # a chance baseline near 4% - so their approach genuinely funnels. Meeting
    # them at the narrow point is cheaper than meeting them on the general.
    corridor_defence: bool = True
    corridor_max_dist: int = 8      # never hold a chokepoint further out than this

    # ---- attacking ---------------------------------------------------------
    attack_margin: float = 1.25        # strike stack must beat estimated defence
    # Committing to a general we have only inferred needs a bigger cushion, and
    # is not worth doing before the board has opened up.
    attack_margin_unsure: float = 2.0
    probe_turn: int = 170
    located_candidates: int = 4        # at or below this, treat the guess as known
    # What fraction of the enemy's total army we assume can reach their general
    # in time, when we cannot see it.
    attack_defense_frac: float = 0.2
    # Believe our eyes: when their general is actually visible, price the
    # defence from what is standing on it rather than a fraction of their
    # global army. Off = fall back to the global prior.
    attack_trust_sight: bool = True
    # Subtract visible enemy army that is further from their general than our
    # fist is: it cannot get home in time, so it is not defence. Off = price
    # every enemy tile as a garrison no matter where it is standing.
    attack_discount_committed: bool = True
    gather_start_turn: int = 60        # before this, expansion beats consolidation
    gather_army_ratio: float = 1.05    # gather once our army is at least this x theirs
    # Keep expanding while there is still this much free ground within reach;
    # land is income and it is never worth hoarding army instead.
    expand_floor: int = 3
    # Score below which passing beats the best available move. Effectively off
    # by default: giving up a turn is almost always worse than a bad move.
    pass_score: float = -1e6

    # ---- thrust: depth-first penetration -----------------------------------
    # EXPAND walks toward the NEAREST free tile, which spreads us thin - breadth
    # first. Rank-1 Kubic does the opposite: 0.39 capture rate, 117 idle turns
    # and a 47-army stack, i.e. it stops nibbling and drives one fist deep. A
    # thrust commits a single stack to a single deep target and holds that
    # commitment across turns; the scored move generator cannot, because it
    # re-picks a source every turn and the big stack gets distracted by whatever
    # capture is adjacent.
    thrust_enabled: bool = True
    thrust_min_turn: int = 80
    # 20/0.10, NOT the 8/0.04 that local measurement asked for. Lowering it
    # scored +99 elo in self-play over 1200 games with SPRT, and 0.977 against
    # hunter:3 versus 0.960 — two agreeing instruments — and then lost 241 elo on
    # the ladder (1737 -> 1496, 3.6 sigma), including losses to the Hunter
    # baseline we beat 0.977 locally. A small stack driven deep is free against
    # every opponent we can build and is eaten by real ones. Do not lower this
    # on local evidence.
    thrust_min_army: int = 20          # stack must be worth committing
    thrust_army_ratio: float = 0.10    # ...and be a real share of our army
    # Before a fist exists, make one: route army to a rally tile on the front
    # instead of nibbling. This is the other half of the Kubic shape - the
    # idle turns are not wasted, they are the fist being built.
    # Measured: routing army to a rally tile via GATHER is catastrophic
    # (0.230, -210 elo). GATHER captures on 9% of its turns; thrust captures
    # on 52%. The fist has to come from expansion, not from a massing phase.
    mass_enabled: bool = False
    mass_min_army: int = 60            # total army before massing is worth it
    thrust_abort_army: int = 8         # give up when the fist is spent
    # Do not commit the fist while home is exposed, and recall it if that
    # changes. Ladder evidence: with the thrust running we lost 26 games to
    # stacks of 6-25 army while AHEAD on land, general holding ~7. Self-play
    # cannot see this - both sides thrust, so the counter-attack window is
    # symmetric and cancels.
    thrust_home_safe: int = 10         # no enemy stack within this many steps

    # ---- endgame -----------------------------------------------------------
    deathtouch_prep_turn: int = 730    # start walking a unit at the enemy general
    deathtouch_guard_turn: int = 770   # start clearing our own general's approach

    # ---- engine-side safety ------------------------------------------------
    time_budget_ms: float = 110.0      # leave headroom under the 150 ms limit

    # Play the cloned net when `bot/weights.npz` was packaged alongside this
    # config. A bool, because `flatten` skips bools and floats everything else:
    # switches are chosen, not searched. Set it false to A/B the heuristic
    # against the net without repacking.
    use_net: bool = True

    # ------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        cfg = cls()
        for f in fields(cls):
            if f.name not in d:
                continue
            cur = getattr(cfg, f.name)
            if is_dataclass(cur):
                setattr(cfg, f.name, type(cur)(**d[f.name]))
            else:
                setattr(cfg, f.name, type(cur)(d[f.name]))
        return cfg

    # ---- flat vector view, for the parameter search ------------------------
    def flatten(self) -> tuple[list[str], list[float]]:
        names, vals = [], []
        for f in fields(self):
            cur = getattr(self, f.name)
            if is_dataclass(cur):
                for sub in fields(cur):
                    names.append(f"{f.name}.{sub.name}")
                    vals.append(float(getattr(cur, sub.name)))
            elif isinstance(cur, bool):
                continue  # switches are not searched, they are chosen
            else:
                names.append(f.name)
                vals.append(float(cur))
        return names, vals

    def with_vector(self, names: list[str], vals: list[float]) -> "Config":
        cfg = Config.from_dict(asdict(self))
        for name, v in zip(names, vals):
            if "." in name:
                block, attr = name.split(".")
                setattr(getattr(cfg, block), attr, float(v))
            else:
                cur = getattr(cfg, name)
                setattr(cfg, name, type(cur)(round(v) if isinstance(cur, int) else v))
        return cfg
