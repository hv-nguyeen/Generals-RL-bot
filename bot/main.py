"""Submission entry point: the stdio loop.

Two hard requirements from the sandbox, both handled here rather than in the
policy: reply inside 150 ms, and never die. An exception in the policy becomes a
pass (one fault out of an allowance of fifty) instead of a crashed process,
which is an instant forfeit.

Which policy plays is decided once, at handshake, by `make_agent`: the cloned
network if `bot/weights.npz` was packaged and `use_net` is on, the heuristic
Controller otherwise. Every outcome prints one line to stderr, including the
fallback — the ladder cannot tell you which one it played.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import protocol, rules                      # noqa: E402
from bot.config import Config                        # noqa: E402
from bot.policy.controller import Controller         # noqa: E402


def load_config() -> Config:
    path = os.environ.get("BOT_CONFIG")
    if not path:
        default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        path = default if os.path.exists(default) else None
    return Config.load(path) if path else Config()


# Not a Config field: `Config.flatten` floats every non-bool, so a path in there
# would break the tuner. The switch is the bool; the location is fixed.
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz")


def make_agent(cfg: Config, player_id: int, h: int, w: int):
    """The cloned net if one was packaged, the heuristic otherwise.

    Construction happens outside main()'s per-move try/except, so a corrupt or
    half-copied npz would otherwise kill the process before the first frame —
    an instant forfeit, in exchange for a file that is optional by design. It
    falls back instead, and says so on stderr: a submission that quietly plays
    the wrong policy is the failure that is hard to notice from the ladder.
    """
    if cfg.use_net and os.path.exists(WEIGHTS):
        try:
            from bot.policy.net import ClonePolicy
            from bot.policy.guard import GuardedPolicy
            # tta: average the logits over the board's dihedral group instead of
            # reading one orientation. Measured +32.8 Elo [+17.7, +48.0] on 2000
            # games with the SAME weights on both sides, so it is inference
            # compute and nothing else. Costs 5.3 ms mean, 14.3 ms worst, against
            # a 150 ms budget -- the axis this project had never spent.
            agent = ClonePolicy(player_id, h, w, WEIGHTS, tta=cfg.tta,
                                full=cfg.tta_full)
            # The net is argmax over masked logits and nothing else. The
            # heuristic's hard-override tier -- win-in-one, deathtouch, the
            # narrow garrison block -- has no counterpart in it, and deathwatch
            # over 50 ladder losses found the general emptied with a comparable
            # stack within 3 steps in 16 of them. Self-play cannot punish that:
            # at short curriculum distances it is correct tempo, and in a mirror
            # both sides do it.
            guarded = GuardedPolicy(agent, cfg.general_block_radius,
                                    cfg.general_block_ratio)
            print(f"policy: net {agent.net.arch} from {WEIGHTS} "
                  f"(guarded r={cfg.general_block_radius} "
                  f"ratio={cfg.general_block_ratio})", file=sys.stderr)
            return guarded
        except Exception:                             # noqa: BLE001
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(f"policy: FALLING BACK to the heuristic, {WEIGHTS} did not load",
                  file=sys.stderr)
    else:
        print(f"policy: heuristic (use_net={cfg.use_net}, "
              f"weights present={os.path.exists(WEIGHTS)})", file=sys.stderr)
    return Controller(player_id, h, w, cfg)


def main() -> None:
    cfg = load_config()
    player_id, h, w = protocol.read_handshake(sys.stdin)
    ctrl = make_agent(cfg, player_id, h, w)

    budget = cfg.time_budget_ms / 1000.0
    first = True
    while True:
        obs = protocol.read_frame(sys.stdin, h, w)
        if obs is None:
            return
        start = time.perf_counter()
        deadline = start + (rules.FIRST_MOVE_BUDGET_S * 0.8 if first else budget)
        first = False
        try:
            action = ctrl.act(obs, deadline)
        except Exception:                             # noqa: BLE001 - never crash
            import traceback
            traceback.print_exc(file=sys.stderr)
            action = rules.PASS_ACTION
        protocol.write_action(sys.stdout, action)


if __name__ == "__main__":
    main()
