"""Opt-in value reranking for a small set of policy candidates.

This is deliberately not enabled by the shipped controller. It refuses an
uncalibrated critic, evaluates only the top few legal policy moves, and keeps
the policy score as the dominant term. Promotion still requires runtime and
multi-opponent arena gates.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from bot import features, rules
from bot.obs import Obs
from bot.policy.net import ClonePolicy, ValueNet

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _validated_critic(path: str) -> dict:
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.is_file():
        raise ValueError(f"{path}: missing value-model validation sidecar {sidecar}")
    meta = json.loads(sidecar.read_text())
    m, c = meta.get("metrics", {}), meta.get("control", {})
    gain = float(m.get("evar", -1e9)) - float(c.get("evar", 1e9))
    if (meta.get("gate_passed") is not True or gain < 0.02
            or float(m.get("ece", 1e9)) > 0.15
            or float(m.get("target_var", 0.0)) <= 0.05):
        raise ValueError(f"{path}: critic gate failed (board evar gain {gain:+.3f}, "
                         f"ece {m.get('ece')})")
    return meta


def _after_own_action(obs: Obs, action) -> Obs:
    """Visible, opponent-free post-action approximation for value reranking."""
    ty, ow, army = (obs.type_grid.copy(), obs.owner_grid.copy(), obs.army_grid.copy())
    kind, r, c, d, split = map(int, action)
    my_land, my_army, opp_land = obs.my_land, obs.my_army, obs.opp_land
    if kind == rules.BUILD:
        structures = ((ow == rules.OWNER_ME)
                      & ((ty == rules.T_GENERAL) | (ty == rules.T_CASTLE)))
        cost = int(rules.build_cost_grid(structures)[r, c])
        army[r, c] -= cost
        my_army -= cost
        ty[r, c] = rules.T_CASTLE
    elif kind == rules.MOVE:
        dr, dc = DIRS[d]
        nr, nc = r + dr, c + dc
        moved = rules.army_to_move(int(army[r, c]), split)
        army[r, c] -= moved
        if ow[nr, nc] == rules.OWNER_ME:
            army[nr, nc] += moved
        else:
            previous_owner = int(ow[nr, nc])
            defended = int(army[nr, nc])
            if moved > defended:
                my_land += 1
                if previous_owner == rules.OWNER_OPP:
                    opp_land = max(0, opp_land - 1)
                ow[nr, nc] = rules.OWNER_ME
                army[nr, nc] = moved - defended
            else:
                army[nr, nc] = defended - moved
        my_army = int(army[ow == rules.OWNER_ME].sum())
    return replace(obs, my_land=my_land, my_army=my_army, opp_land=opp_land,
                   type_grid=ty, owner_grid=ow, army_grid=army)


class SearchPolicy:
    """Top-K policy reranking with a held-out, calibrated value model."""

    def __init__(self, player_id: int, h: int, w: int, policy: str, critic: str,
                 topk: int = 6, value_weight: float = 0.5,
                 tta: bool = True, full: bool = True):
        self.critic_meta = _validated_critic(critic)
        self.inner = ClonePolicy(player_id, h, w, policy, tta=tta, full=full)
        self.critic = ValueNet(critic)
        self.topk = max(1, int(topk))
        self.value_weight = float(value_weight)
        self.last_debug = {}

    def score_logits(self, obs: Obs) -> np.ndarray:
        logits = self.inner.score_logits(obs)
        mask = features.legal_mask(obs)
        legal = np.flatnonzero(mask)
        if len(legal) <= 1:
            return logits
        order = legal[np.argsort(logits[legal])[-self.topk:]]
        reranked = logits.copy()
        for idx in order:
            nxt = _after_own_action(obs, features.index_to_action(int(idx)))
            enc = features.encode(nxt, self.inner.memory)
            reranked[idx] += self.value_weight * self.critic.value_encoded(enc)
        return reranked

    def act(self, obs: Obs, deadline=None):
        scores = self.score_logits(obs)
        mask = features.legal_mask(obs)
        idx = int(np.argmax(np.where(mask, scores, -np.inf)))
        self.last_debug = {"mode": "search", "turn": obs.turn, "topk": self.topk}
        return features.index_to_action(idx)
