"""The bot.

Two tiers:

* **Hard overrides** — a move that wins outright, the deathtouch guard, and the
  castle build. These bypass scoring because they are not trade-offs.
* **Scored move generation** — every legal move gets a feature vector, dotted
  with the weight block of the current mode. Multi-turn coherence comes from the
  distance fields: a greedy pick per turn still walks a coherent multi-turn
  march because the field does not change under it.

The strategic layer is mode selection, which chooses both the weight block and
the distance field that `progress` measures against.
"""

from __future__ import annotations

import math
import time

import numpy as np

from bot import rules
from bot.board import DIRS, bfs_field_from
from bot.belief import Belief
from bot.config import ATTACK, DEATHTOUCH, DEFEND, EXPAND, GATHER, Config
from bot.obs import Obs
from bot.policy import castle
from bot.policy.analysis import Analysis


def _clamp3(v: int) -> int:
    return 3 if v > 3 else (-3 if v < -3 else v)


class Controller:
    def __init__(self, player_id: int, h: int, w: int, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.belief = Belief(player_id, h, w)
        self.H, self.W = h, w
        self._defend_until = -1
        self.thrust: dict | None = None
        self.rally: tuple[int, int] | None = None
        self.last_debug: dict = {}

    # ------------------------------------------------------------------ act
    def act(self, obs: Obs, deadline: float | None = None) -> tuple[int, int, int, int, int]:
        self.belief.update(obs)
        an = Analysis(self.belief, obs)

        win = self._winning_move(obs, an)
        if win is not None:
            self._debug(obs, an, "win", None)
            return win

        guard = self._deathtouch_guard(obs, an)
        if guard is not None:
            self._debug(obs, an, "guard", None)
            return guard

        lethal = self._lethal_guard(obs, an)
        if lethal is not None:
            self._debug(obs, an, "lethal", None)
            return lethal

        plan = castle.plan(an, self.cfg)
        mode, field = self._select_mode(obs, an, plan)

        if mode != DEFEND:
            drive = self._thrust(obs, an)
            if drive is not None:
                self._debug(obs, an, "thrust", self.thrust and self.thrust["target"])
                return drive

        if plan.action is not None and mode not in (DEFEND, ATTACK, DEATHTOUCH):
            self._debug(obs, an, "build", plan.site)
            return plan.action

        self._debug(obs, an, mode, plan.site)
        return self._best_move(obs, an, mode, field, deadline)

    # -------------------------------------------------------------- overrides
    def _winning_move(self, obs: Obs, an: Analysis):
        """Capture the enemy general now, either by force or by deathtouch."""
        eg = self.belief.enemy_general
        if eg is None:
            return None
        er, ec = eg
        touch = obs.turn >= rules.DEATHTOUCH_TURN
        defenders = int(obs.army_grid[er, ec])
        best, best_mv = None, -1
        for d, (dr, dc) in enumerate(DIRS):
            # a move in direction d from (er-dr, ec-dc) lands on the general
            sr, sc = er - dr, ec - dc
            if not (0 <= sr < self.H and 0 <= sc < self.W):
                continue
            if obs.owner_grid[sr, sc] != rules.OWNER_ME:
                continue
            mv = rules.army_to_move(int(obs.army_grid[sr, sc]), 0)
            if mv <= 0:
                continue
            if not touch and mv <= defenders:
                continue
            if mv > best_mv:
                best, best_mv = (rules.MOVE, sr, sc, d, 0), mv
        return best

    def _deathtouch_guard(self, obs: Obs, an: Analysis):
        """Stop an enemy unit that is one step from touching our general.

        A move onto their source is a *chase*, which resolves first — capture the
        source and their touch never executes. It has to come from a third tile:
        the general attacking its own attacker is a mutual chase and loses the
        tie-break.
        """
        if obs.turn < min(self.cfg.deathtouch_guard_turn, rules.DEATHTOUCH_TURN - 1):
            return None
        gr, gc = self.belief.my_general
        best, best_mv = None, -1
        for dr, dc in DIRS:
            tr, tc = gr + dr, gc + dc
            if not (0 <= tr < self.H and 0 <= tc < self.W):
                continue
            if obs.owner_grid[tr, tc] != rules.OWNER_OPP or obs.army_grid[tr, tc] < 2:
                continue
            # find a third tile that can take (tr, tc) outright
            for d, (sr_off, sc_off) in enumerate(DIRS):
                sr, sc = tr - sr_off, tc - sc_off
                if not (0 <= sr < self.H and 0 <= sc < self.W):
                    continue
                if (sr, sc) == (gr, gc):
                    continue  # mutual chase, loses the tie-break
                if obs.owner_grid[sr, sc] != rules.OWNER_ME:
                    continue
                mv = rules.army_to_move(int(obs.army_grid[sr, sc]), 0)
                if mv > int(obs.army_grid[tr, tc]) and mv > best_mv:
                    best, best_mv = (rules.MOVE, sr, sc, d, 0), mv
        return best

    def _lethal_guard(self, obs: Obs, an: Analysis):
        """An enemy stack close enough to take the general before help arrives.

        `_deathtouch_guard` only fires once they are already adjacent and only in
        the endgame. This is the general case: real losses had the general on 4
        troops with 28 enemy troops three steps away while our global army was
        winning comfortably. Answer it by taking the stack if we can, otherwise by
        putting army onto the general.
        """
        gr, gc = self.belief.my_general
        h, w = self.H, self.W
        worst = None
        for r, c in np.argwhere(an.opp_mask & (obs.army_grid >= 2)):
            r, c = int(r), int(c)
            eta = int(an.dist_home[r, c])
            if eta > self.cfg.guard_radius:
                continue
            incoming = int(obs.army_grid[r, c]) - 1
            if incoming < an.defense_within(eta):
                continue
            if worst is None or incoming > worst[0]:
                worst = (incoming, r, c)
        if worst is None:
            return None
        _, tr, tc = worst

        # Best answer: take the stack outright, from a tile that is not the
        # general (the general trading with its own attacker loses the tie-break).
        best, best_mv = None, -1
        for d, (dr, dc) in enumerate(DIRS):
            sr, sc = tr - dr, tc - dc
            if not (0 <= sr < h and 0 <= sc < w):
                continue
            if obs.owner_grid[sr, sc] != rules.OWNER_ME or (sr, sc) == (gr, gc):
                continue
            mv = rules.army_to_move(int(obs.army_grid[sr, sc]), 0)
            if mv > int(obs.army_grid[tr, tc]) and mv > best_mv:
                best, best_mv = (rules.MOVE, sr, sc, d, 0), mv
        if best is not None:
            return best

        # Otherwise feed the general from the fattest neighbour.
        for d, (dr, dc) in enumerate(DIRS):
            sr, sc = gr - dr, gc - dc
            if not (0 <= sr < h and 0 <= sc < w):
                continue
            if obs.owner_grid[sr, sc] != rules.OWNER_ME:
                continue
            mv = rules.army_to_move(int(obs.army_grid[sr, sc]), 0)
            if mv > best_mv:
                best, best_mv = (rules.MOVE, sr, sc, d, 0), mv
        return best

    # ----------------------------------------------------------------- thrust
    def _thrust(self, obs: Obs, an: Analysis):
        """Drive one committed stack at a deep target, one step per turn.

        Depth-first, not breadth-first: the stack keeps its heading across turns
        instead of being re-chosen every turn against whatever is nearest. It
        gives up the rest of the board while it runs, which is the trade the top
        of the leaderboard makes and we never did.
        """
        cfg = self.cfg
        if not cfg.thrust_enabled or obs.turn < cfg.thrust_min_turn:
            self.thrust = None
            return None

        if self.thrust is not None:
            r, c = self.thrust["pos"]
            if (obs.owner_grid[r, c] != rules.OWNER_ME
                    or int(obs.army_grid[r, c]) < cfg.thrust_abort_army):
                self.thrust = None          # the fist died or was taken

        # Home exposed? Do not start a drive, and drop one already running.
        if an.max_threat_within(cfg.thrust_home_safe) > an.general_army:
            self.thrust = None
            return None

        if self.thrust is None:
            if an.biggest_stack_pos is None or an.biggest_stack < cfg.thrust_min_army:
                return None
            if an.biggest_stack < cfg.thrust_army_ratio * max(obs.my_army, 1):
                return None
            target = self.belief.enemy_general or self.belief.general_guess
            if target is None:
                return None
            self.thrust = {"pos": an.biggest_stack_pos, "target": target}

        # Re-aim the moment we actually see their general.
        if self.belief.enemy_general is not None:
            self.thrust["target"] = self.belief.enemy_general
        r, c = self.thrust["pos"]
        target = self.thrust["target"]
        if (r, c) == target:
            self.thrust = None
            return None

        field = bfs_field_from(self.belief.passable, target)
        here = int(field[r, c])
        army = int(obs.army_grid[r, c])
        best, best_key = None, None
        for d, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.H and 0 <= nc < self.W):
                continue
            if not self.belief.passable[nr, nc] or int(field[nr, nc]) >= here:
                continue
            dest_army = int(obs.army_grid[nr, nc])
            mine = obs.owner_grid[nr, nc] == rules.OWNER_ME
            if not mine and army - 1 <= dest_army:
                continue                     # cannot punch through this tile
            # closer first, then prefer taking the most army off them
            key = (-int(field[nr, nc]), dest_army if not mine else -1)
            if best_key is None or key > best_key:
                best_key, best = key, (rules.MOVE, r, c, d, 0, nr, nc)
        if best is None:
            self.thrust = None               # blocked; fall back to scoring
            return None
        self.thrust["pos"] = (best[5], best[6])
        return best[:5]

    # ------------------------------------------------------------------ modes
    def _select_mode(self, obs: Obs, an: Analysis, plan) -> tuple[str, np.ndarray]:
        cfg = self.cfg
        passable = self.belief.passable

        if an.threat_pos is not None and an.threat_dist <= cfg.defend_horizon:
            if an.threat_army > an.defense_within(an.threat_dist) * cfg.defend_margin:
                self._defend_until = obs.turn + cfg.defend_hold

        # Deathtouch only pays if we know where to touch. Marching at a *guess*
        # burns hundreds of turns and stalls the moment the guess is wrong —
        # that was the single biggest source of draws. Unlocated, the normal
        # modes below still push at the enemy and take ground on the way.
        if self._located() and obs.turn >= min(cfg.deathtouch_prep_turn,
                                               rules.DEATHTOUCH_TURN):
            return DEATHTOUCH, an.dist_enemy_gen

        if obs.turn < self._defend_until:
            return DEFEND, an.dist_home

        if self._can_attack(obs, an):
            return ATTACK, an.dist_enemy_gen

        # Massing: no fist yet, but we have the army to build one. Route it to a
        # rally tile on the front rather than spending the turns on more nibbling.
        if (cfg.thrust_enabled and cfg.mass_enabled and self.thrust is None
                and obs.turn >= cfg.thrust_min_turn
                and obs.my_army >= cfg.mass_min_army
                and an.biggest_stack < cfg.thrust_min_army
                and self.belief.general_guess is not None):
            rally = self._rally_point(obs, an)
            if rally is not None:
                return GATHER, bfs_field_from(passable, rally)

        # Saving up for a castle: walk army onto the chosen site.
        if plan.site is not None and plan.action is None:
            return GATHER, bfs_field_from(passable, plan.site)

        # Land is income, so expansion outranks consolidation for as long as
        # there is free ground within reach. Only once the board is carved up
        # does hoarding army for a strike become the better use of a turn.
        boxed_in = an.unowned_near < cfg.expand_floor
        ready_to_stage = (obs.turn >= cfg.gather_start_turn
                          and obs.my_army >= cfg.gather_army_ratio * max(obs.opp_army, 1)
                          and self._located())
        if (boxed_in or ready_to_stage) and self.belief.general_guess is not None:
            staging = self._staging(an)
            if staging is not None:
                return GATHER, bfs_field_from(passable, staging)

        return EXPAND, an.dist_unowned

    def _located(self) -> bool:
        """Do we know where the enemy general is, well enough to commit?"""
        return (self.belief.enemy_general is not None
                or int(self.belief.candidates.sum()) <= self.cfg.located_candidates)

    def _can_attack(self, obs: Obs, an: Analysis) -> bool:
        b = self.belief
        target = b.enemy_general or b.general_guess
        if target is None or an.biggest_stack_pos is None:
            return False
        located = self._located()
        if not located and obs.turn < self.cfg.probe_turn:
            return False
        d = int(an.dist_enemy_gen[an.biggest_stack_pos])
        if d >= self.H * self.W:
            return False

        # If their general is visible RIGHT NOW, believe our eyes: count what is
        # standing on it plus the enemy army we can see near it. Pricing a visible
        # 2-army general at 20% of their global total refused kills we had already
        # won - that cost real games.
        if (self.cfg.attack_trust_sight
                and obs.type_grid[target] == rules.T_GENERAL
                and obs.owner_grid[target] == rules.OWNER_OPP):
            near = an.opp_mask & (an.dist_enemy_gen <= max(1, d // 2))
            estimate = int(obs.army_grid[target]) + int(obs.army_grid[near].sum()) + d // 2
            return (an.biggest_stack - 1) > estimate * self.cfg.attack_margin

        seen = int(b.mem_turn[target]) >= 0
        seen_army = int(b.mem_army[target]) if seen else 0
        seen_age = obs.turn - int(b.mem_turn[target]) if seen else 0
        # An unseen general used to be priced at rules.general_army_at(turn) —
        # what a general that never moved would hold. Real opponents spend it
        # (hunter garrisons 4), so that term priced every kill attempt out of
        # reach and ATTACK never fired once in 160 games. Their general cannot
        # hold more than their total army, so bound it by that and let
        # attack_defense_frac carry the estimate.
        estimate = max(
            seen_army + seen_age // 2,
            int(obs.opp_army * self.cfg.attack_defense_frac),
        )
        margin = self.cfg.attack_margin if located else self.cfg.attack_margin_unsure
        arriving = an.biggest_stack - 1
        return arriving > estimate * margin

    def _rally_point(self, obs: Obs, an: Analysis):
        """Where the fist forms. Held across turns - a rally point that moves
        every turn is one the army never actually reaches."""
        if self.rally is not None:
            r, c = self.rally
            if obs.owner_grid[r, c] == rules.OWNER_ME:
                return self.rally
        self.rally = self._staging(an)
        return self.rally

    def _staging(self, an: Analysis):
        """Our own tile closest to the enemy general — the natural front."""
        mine = an.my_mask
        if not mine.any():
            return None
        d = np.where(mine, an.dist_enemy_gen, self.H * self.W)
        idx = int(np.argmin(d))
        if d.flat[idx] >= self.H * self.W:
            return None
        return (idx // self.W, idx % self.W)

    # ------------------------------------------------------------- move search
    def _best_move(self, obs: Obs, an: Analysis, mode: str, field: np.ndarray, deadline):
        cfg = self.cfg
        w = getattr(cfg, mode)
        h, width = self.H, self.W
        gr, gc = self.belief.my_general

        ty = obs.type_grid.tolist()
        ow = obs.owner_grid.tolist()
        ar = obs.army_grid.tolist()
        fld = field.tolist()
        deg = an.dist_enemy_gen.tolist()
        dh = an.dist_home.tolist()
        passable = self.belief.passable.tolist()
        reveal = an.reveal.tolist()
        frontier = an.frontier_adj.tolist()
        contact = an.enemy_adj.tolist()
        ecastle = self.belief.enemy_castles.tolist()
        diag = an.diag

        sources = np.argwhere(an.my_mask & (obs.army_grid >= 2))
        # Opening: a stack of A army captures A-1 tiles before it runs dry, and
        # trickling two units out at a time just re-walks the same ground. Hold
        # the general's army until it is worth a full run, even if that means
        # passing for the first twenty turns.
        hold_general = mode == EXPAND and obs.turn < cfg.first_expand_turn
        # The general regrows +1 every two turns on its own, so the cheapest
        # possible defence is simply not to spend it. Their total army is
        # reported every turn, so the floor can track the force that actually
        # exists rather than the part we happen to see. Committing modes ignore
        # it — at that point the game is decided by the attack, not the base.
        floor = 0
        if mode not in (ATTACK, DEATHTOUCH):
            if cfg.garrison_frac > 0 and obs.turn >= cfg.garrison_from_turn:
                floor = min(cfg.garrison_cap, int(cfg.garrison_frac * obs.opp_army))
            if cfg.lock_enabled:
                # Only while something is actually coming: keep enough on the
                # general to survive the biggest stack in range.
                incoming = an.max_threat_within(cfg.lock_radius)
                if incoming > 0:
                    # Not min(): clamping to what is already on the general made
                    # this a refuse-to-spend rule that could never reinforce.
                    floor = max(floor, incoming + 1)

        best, best_score = None, -math.inf
        checked = 0
        for si in range(len(sources)):
            r = int(sources[si][0])
            c = int(sources[si][1])
            if hold_general and r == gr and c == gc:
                continue
            army = ar[r][c]
            f_src = fld[r][c]
            is_general = (r == gr and c == gc)

            splits = (0, 1) if army >= 4 else (0,)
            for d in range(4):
                dr, dc = DIRS[d]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < width) or not passable[nr][nc]:
                    continue
                dest_owner = ow[nr][nc]
                dest_army = ar[nr][nc]
                progress = _clamp3(f_src - fld[nr][nc])
                closing = _clamp3(deg[r][c] - deg[nr][nc])
                base = (w.progress * progress
                        + w.toward_enemy * closing
                        + w.keep_home * (-dh[nr][nc] / diag)
                        + (w.from_general if is_general else 0.0)
                        + (w.contact if contact[nr][nc] else 0.0))

                for split in splits:
                    mv = army // 2 if split else army - 1
                    if mv <= 0:
                        continue
                    if is_general and army - mv < floor:
                        continue
                    s = base + w.army * math.log1p(mv) + w.stack_break * ((army - mv) / army)
                    if dest_owner == rules.OWNER_ME:
                        s += w.to_own
                    elif mv > dest_army:
                        if dest_owner == rules.OWNER_OPP:
                            s += w.cap_enemy + w.cap_enemy_army * (dest_army / 10.0)
                            if ecastle[nr][nc]:
                                s += w.cap_castle
                        else:
                            s += w.cap_neutral
                        s += w.reveal * (reveal[nr][nc] / 8.0)
                        s += w.frontier * (frontier[nr][nc] / 4.0)
                    else:
                        s += w.failed_capture

                    if s > best_score:
                        best_score, best = s, (rules.MOVE, r, c, d, split)

            checked += 1
            if deadline is not None and (checked & 31) == 0 and time.perf_counter() > deadline:
                break

        if best is None or best_score < cfg.pass_score:
            return rules.PASS_ACTION
        return best

    # ------------------------------------------------------------------ debug
    def _debug(self, obs: Obs, an: Analysis, mode: str, site) -> None:
        self.last_debug = {
            "turn": obs.turn,
            "mode": mode,
            "guess": self.belief.general_guess,
            "known_general": self.belief.enemy_general is not None,
            "candidates": int(self.belief.candidates.sum()),
            "threat": (an.threat_army, an.threat_dist),
            "castles": int(self.belief.my_castles.sum()),
            "enemy_castles": int(self.belief.enemy_castles.sum()),
            "site": site,
            "hidden_army": self.belief.hidden_enemy_army,
        }
