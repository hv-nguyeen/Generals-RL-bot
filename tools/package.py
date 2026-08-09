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
import atexit
import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RUN_SH = """#!/usr/bin/env bash
# The sandbox runs this with the submission directory as cwd, but do not rely
# on that: resolve relative to the script itself.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
# One BLAS thread. The match sandbox is single-core, so a thread pool sized to
# the host is pure overhead -- and on a many-core machine numpy spends real time
# spinning one up on its first matmul, which lands inside the first move's
# budget. A move over 150 ms is a forfeit, and a forfeit is a loss.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY="${BOT_PYTHON:-$(command -v python3 || command -v python)}"
exec "$PY" -u -m bot.main
"""


def _include_source(path: Path) -> bool:
    return (not path.name.startswith("._")
            and path.name != ".DS_Store"
            and "__pycache__" not in path.parts)


def build(name: str, config: str | None,
          include_weights: bool = True) -> tuple[Path, Path, Path]:
    dist = REPO / "dist"
    stage = dist / name
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "bot" / "policy").mkdir(parents=True)

    for src in sorted((REPO / "bot").glob("*.py")):
        if not _include_source(src):
            continue
        shutil.copy2(src, stage / "bot" / src.name)
    for src in sorted((REPO / "bot" / "policy").glob("*.py")):
        if not _include_source(src):
            continue
        shutil.copy2(src, stage / "bot" / "policy" / src.name)
    # bot/weights.npz if it exists: bot/main.py plays the net when the file is
    # there, so leaving it out of the zip silently ships the heuristic instead.
    # Measured, a 10x128 checkpoint is 5 MB zipped against a 50 MB limit.
    if include_weights:
        for src in sorted((REPO / "bot").glob("*.npz")):
            shutil.copy2(src, stage / "bot" / src.name)

    if config:
        shutil.copy2(config, stage / "bot" / "config.json")

    run_sh = stage / "run.sh"
    run_sh.write_text(RUN_SH)
    run_sh.chmod(0o755)

    archive = dist / f"{name}.zip"
    pending = dist / f".{name}.zip.tmp"
    pending.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(stage.rglob("*")):
                if path.is_file() and _include_source(path):
                    info = zipfile.ZipInfo(str(path.relative_to(dist)))
                    info.external_attr = (0o755 if path.name.endswith(".sh") else 0o644) << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    z.writestr(info, path.read_bytes())
    except Exception:
        pending.unlink(missing_ok=True)
        raise
    return stage, pending, archive


def validate_archive(stage: Path, archive: Path) -> None:
    """Reject layout drift and unexpected macOS/cache sidecars."""
    root = stage.parent
    expected = {str(p.relative_to(root)) for p in stage.rglob("*")
                if p.is_file() and _include_source(p)}
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if z.testzip() is not None:
            raise SystemExit("submission ZIP failed its CRC check")
    if len(names) != len(set(names)):
        raise SystemExit("submission ZIP has duplicate members")
    actual = set(names)
    if actual != expected:
        raise SystemExit(f"submission ZIP member mismatch: missing "
                         f"{sorted(expected - actual)}, unexpected "
                         f"{sorted(actual - expected)}")
    bad = [n for n in names if "/._" in n or n.endswith("/.DS_Store")
           or "/__pycache__/" in n or not n.startswith(stage.name + "/")]
    if bad:
        raise SystemExit(f"submission ZIP has forbidden members: {bad}")


def validate_identity(digest: str | None, expect: str | None,
                      allow_unpinned_net: bool = False,
                      allow_heuristic: bool = False) -> None:
    """Fail-closed release identity policy, kept pure for regression tests."""
    if not digest and not allow_heuristic:
        raise SystemExit("neural release requires bot/weights.npz; use "
                         "--allow-heuristic only for an intentional heuristic ZIP")
    if digest and not expect and not allow_unpinned_net:
        raise SystemExit("neural release requires --expect <checkpoint sha256>; "
                         "use --allow-unpinned-net only for a development build")
    if expect and digest != expect:
        raise SystemExit(
            f"--expect {expect}\n     got {digest or 'NO NET AT ALL'}\n"
            "The zip does NOT contain the checkpoint you meant to ship.")


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
    ap.add_argument("--allow-unpinned-net", action="store_true",
                    help="development only: permit a neural ZIP without --expect")
    ap.add_argument("--allow-heuristic", action="store_true",
                    help="explicitly build a heuristic-only release")
    args = ap.parse_args()

    # An explicit heuristic artifact excludes weights even if a stale neural
    # file exists in bot/. This makes the opt-out describe the artifact rather
    # than merely relaxing its validation.
    stage, pending, archive = build(
        args.name, args.config, include_weights=not args.allow_heuristic)
    # Any validation/test exception leaves the last known-good final archive in
    # place and removes only this unpublished candidate.
    atexit.register(lambda: pending.unlink(missing_ok=True))
    expect = args.expect
    files = sum(1 for p in stage.rglob("*") if p.is_file())
    size_mb = pending.stat().st_size / 1e6
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
        print("  net: NONE")
    try:
        validate_identity(digest, expect, args.allow_unpinned_net,
                          args.allow_heuristic)
    except SystemExit:
        pending.unlink(missing_ok=True)
        raise
    if digest and args.config and not args.allow_heuristic:
        from bot.config import Config
        if not Config.load(args.config).use_net:
            raise SystemExit(f"{args.config} has use_net=false; refusing to "
                             "publish a neural release that selects the heuristic")
    if digest:
        from bot.policy.net import Net
        Net(str(w))  # fail before publication if the packaged checkpoint cannot load
    # Printing the digest catches a stale net only if somebody reads it. --expect
    # makes the build FAIL instead, which is what you want in the one minute
    # before an upload:
    #     sha256sum runs/nn/spN.best.npz
    #     make package EXPECT=<that hash>
    if expect:
        print("  net matches --expect")
    violations = []
    for limit, actual, label in ((50, size_mb, "zip MB"), (512, unpacked_mb, "unpacked MB"),
                                 (10_000, files, "files")):
        flag = "ok" if actual <= limit else "OVER LIMIT"
        print(f"  {label:<14} {actual:>10.2f} / {limit}   {flag}")
        if actual > limit:
            violations.append(f"{label} {actual:.2f} > {limit}")
    if violations:
        pending.unlink(missing_ok=True)
        raise SystemExit("submission limit violation: " + "; ".join(violations))
    validate_archive(stage, pending)

    if args.test:
        import sys as _sys
        # Locally, `python3` on PATH may not have numpy; the sandbox's does.
        os.environ.setdefault("BOT_PYTHON", _sys.executable)
        from arena import rating
        from arena.runner import run_match, tally
        with tempfile.TemporaryDirectory(prefix="generals-package-") as td:
            with zipfile.ZipFile(pending) as z:
                z.extractall(td)
            extracted = Path(td) / stage.name
            spec = f"stdio:{extracted / 'run.sh'}"
            print(f"\nplaying {args.test} games as {spec} vs {args.opponent} "
                  f"(EXTRACTED zip, real wire protocol, real 150 ms limit)")
            results = run_match(spec, args.opponent, args.test, workers=1)
            w, d, loss = tally(results)
            print(f"  {w}W {d}D {loss}L   "
                  f"score {rating.summary(w, d, loss)['score']:.3f}")
            slowest = max(max(r["max_ms"]) for r in results)
            faults = sum(r["faults"][r["a_seat"]] for r in results)
            print(f"  slowest move {slowest:.1f} ms, faults {faults}")
            if faults:
                pending.unlink(missing_ok=True)
                raise SystemExit("packaged bot faulted; release not published")

    os.replace(pending, archive)
    zip_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(f"published {archive}  sha256 {zip_digest}")


if __name__ == "__main__":
    main()
