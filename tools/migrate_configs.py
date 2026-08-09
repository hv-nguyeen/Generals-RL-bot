"""Materialize legacy bot configs under the current strict schema.

This is intentionally explicit: loading a legacy partial config during a run
is an error because missing fields otherwise inherit mutable source defaults.

    python -m tools.migrate_configs configs/*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bot.config import Config


def migrate(path: Path, *, check: bool = False) -> bool:
    raw = json.loads(path.read_text())
    cfg = Config.migrate_legacy(raw)
    rendered = json.dumps(cfg.to_dict(), indent=2) + "\n"
    changed = rendered != path.read_text()
    if changed and not check:
        path.write_text(rendered)
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero when a file still needs migration")
    args = ap.parse_args()
    changed = []
    for path in args.paths:
        if migrate(path, check=args.check):
            changed.append(str(path))
            print(("NEEDS " if args.check else "MIGRATED ") + str(path))
        else:
            print("OK " + str(path))
    if args.check and changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
