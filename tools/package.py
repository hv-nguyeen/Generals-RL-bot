"""Build the submission zip, then play the zip.

The artifact is `dist/<name>.zip` containing a single directory with `run.sh` at
its root, which is the layout the dashboard expects. `--test` then runs that
exact directory through the real stdio protocol against the arena, so what we
upload is what we measured — handshake, frame encoding, process lifecycle and
all.

    python -m tools.package --config configs/best.json --test 4
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RUN_SH = """#!/usr/bin/env bash
# The sandbox runs this with the submission directory as cwd, but do not rely
# on that: resolve relative to the script itself.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
PY="${BOT_PYTHON:-$(command -v python3 || command -v python)}"
exec "$PY" -u -m bot.main
"""


def build(name: str, config: str | None) -> tuple[Path, Path]:
    dist = REPO / "dist"
    stage = dist / name
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "bot" / "policy").mkdir(parents=True)

    for src in sorted((REPO / "bot").glob("*.py")):
        shutil.copy2(src, stage / "bot" / src.name)
    for src in sorted((REPO / "bot" / "policy").glob("*.py")):
        shutil.copy2(src, stage / "bot" / "policy" / src.name)
    # bot/weights.npz if it exists: bot/main.py plays the net when the file is
    # there, so leaving it out of the zip silently ships the heuristic instead.
    # Measured, a 10x128 checkpoint is 5 MB zipped against a 50 MB limit.
    for src in sorted((REPO / "bot").glob("*.npz")):
        shutil.copy2(src, stage / "bot" / src.name)

    if config:
        shutil.copy2(config, stage / "bot" / "config.json")

    run_sh = stage / "run.sh"
    run_sh.write_text(RUN_SH)
    run_sh.chmod(0o755)

    archive = dist / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(str(path.relative_to(dist)))
                info.external_attr = (0o755 if path.name.endswith(".sh") else 0o644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, path.read_bytes())
    return stage, archive


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="generals-bot")
    ap.add_argument("--config", default=None, help="config json to bundle as bot/config.json")
    ap.add_argument("--test", type=int, default=0,
                    help="after packaging, play N games through the real run.sh")
    ap.add_argument("--opponent", default="greedy")
    ap.add_argument("--expect", default=None, metavar="SHA256",
                    help="fail unless the bundled bot/weights.npz has this "
                         "sha256. The one guard against shipping the previous "
                         "build's net, which survives every build and looks "
                         "entirely correct in the zip")
    args = ap.parse_args()

    stage, archive = build(args.name, args.config)
    expect = args.expect
    files = sum(1 for p in stage.rglob("*") if p.is_file())
    size_mb = archive.stat().st_size / 1e6
    unpacked_mb = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file()) / 1e6
    print(f"{archive}  {size_mb:.2f} MB zipped, {unpacked_mb:.2f} MB unpacked, {files} files")

    # WHICH net is in there. NOTHING removes bot/weights.npz -- not this script
    # and not the Makefile, whatever README once claimed -- so it survives every
    # build until someone runs `rm bot/weights.npz` by hand. The failure that
    # leaves is a later build silently shipping the PREVIOUS net, which is worse
    # than shipping the heuristic because the zip looks entirely correct.
    #
    # The size does not distinguish two checkpoints of the same architecture, so
    # print the digest. Compare it against the one you meant to ship:
    #     sha256sum runs/nn/spN.best.npz
    w = stage / "bot" / "weights.npz"
    digest = hashlib.sha256(w.read_bytes()).hexdigest() if w.exists() else None
    if digest:
        print(f"  net: bot/weights.npz  {w.stat().st_size} bytes  sha256 {digest}")
    else:
        print("  net: NONE -- this zip plays the HEURISTIC, not the policy. "
              "Copy a checkpoint to bot/weights.npz and rebuild.")
    # Printing the digest catches a stale net only if somebody reads it. --expect
    # makes the build FAIL instead, which is what you want in the one minute
    # before an upload:
    #     sha256sum runs/nn/spN.best.npz
    #     make package EXPECT=<that hash>
    if expect:
        if digest != expect:
            raise SystemExit(
                f"--expect {expect}\n     got {digest or 'NO NET AT ALL'}\n"
                f"The zip does NOT contain the checkpoint you meant to ship. "
                f"bot/weights.npz survives every build, so this is usually a "
                f"leftover from the previous one -- rm it, copy the right "
                f"checkpoint in, and rebuild.")
        print("  net matches --expect")
    for limit, actual, label in ((50, size_mb, "zip MB"), (512, unpacked_mb, "unpacked MB"),
                                 (10_000, files, "files")):
        flag = "ok" if actual <= limit else "OVER LIMIT"
        print(f"  {label:<14} {actual:>10.2f} / {limit}   {flag}")

    if args.test:
        import os
        import sys as _sys
        # Locally, `python3` on PATH may not have numpy; the sandbox's does.
        os.environ.setdefault("BOT_PYTHON", _sys.executable)
        from arena import rating
        from arena.runner import run_match, tally
        spec = f"stdio:{stage / 'run.sh'}"
        print(f"\nplaying {args.test} games as {spec} vs {args.opponent} "
              f"(real wire protocol, real 150 ms limit)")
        results = run_match(spec, args.opponent, args.test, workers=1)
        w, d, loss = tally(results)
        print(f"  {w}W {d}D {loss}L   score {rating.summary(w, d, loss)['score']:.3f}")
        slowest = max(max(r["max_ms"]) for r in results)
        faults = sum(r["faults"][r["a_seat"]] for r in results)
        print(f"  slowest move {slowest:.1f} ms, faults {faults}")
        if faults:
            print("  WARNING: the packaged bot faulted — fix before submitting")


if __name__ == "__main__":
    main()
