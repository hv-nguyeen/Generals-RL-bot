"""Submission entry point: the stdio loop.

Two hard requirements from the sandbox, both handled here rather than in the
policy: reply inside 150 ms, and never die. An exception in the policy becomes a
pass (one fault out of an allowance of fifty) instead of a crashed process,
which is an instant forfeit.
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


def main() -> None:
    cfg = load_config()
    player_id, h, w = protocol.read_handshake(sys.stdin)
    ctrl = Controller(player_id, h, w, cfg)

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
