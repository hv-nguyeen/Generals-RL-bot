"""Reproducible run and artifact manifests.

The manifest records identity, not just filenames: source tree, git state,
command, runtime versions, fully resolved configs, and every input artifact.
It deliberately excludes environment variables so credentials cannot leak.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from bot.config import Config

MANIFEST_SCHEMA_VERSION = 1
SOURCE_SUFFIXES = {
    ".py", ".json", ".toml", ".md", ".sh", ".yaml", ".yml",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".rs", ".lock",
}
SOURCE_DIRS = (
    "bot", "sim", "arena", "analysis", "learn", "tools", "configs",
    # Training imports the vendored official engine directly; evaluation files
    # and executable scripts determine promotion/release outcomes just as surely
    # as Python source does.
    "third_party", "evaluation", "scripts",
)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(path: str | Path, root: str | Path = ".") -> dict:
    p, root = Path(path).resolve(), Path(root).resolve()
    try:
        label = str(p.relative_to(root))
    except ValueError:
        label = str(p)
    if not p.is_file():
        raise FileNotFoundError(p)
    return {"path": label, "bytes": p.stat().st_size, "sha256": sha256(p)}


def source_tree(root: str | Path = ".") -> dict:
    root = Path(root).resolve()
    files = [root / "pyproject.toml", root / "Makefile"]
    for name in SOURCE_DIRS:
        d = root / name
        if d.exists():
            files.extend(p for p in d.rglob("*")
                         if p.is_file() and p.suffix in SOURCE_SUFFIXES)
    files = sorted(set(files))
    h = hashlib.sha256()
    for p in files:
        rel = str(p.relative_to(root)).encode()
        h.update(len(rel).to_bytes(4, "big")); h.update(rel)
        digest = bytes.fromhex(sha256(p))
        h.update(digest)
    return {"sha256": h.hexdigest(), "files": len(files)}


def file_set(paths, root: str | Path = ".") -> dict:
    """Order-independent identity for a set of input files."""
    root = Path(root).resolve()
    entries = [artifact(p, root) for p in sorted(map(Path, paths))]
    h = hashlib.sha256()
    for e in entries:
        h.update(e["path"].encode()); h.update(bytes.fromhex(e["sha256"]))
    return {"sha256": h.hexdigest(), "files": len(entries),
            "bytes": sum(e["bytes"] for e in entries)}


def _git(root: Path) -> dict:
    def run(*args):
        r = subprocess.run(["git", *args], cwd=root, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return r.stdout.strip() if r.returncode == 0 else None
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(status),
            "changed_paths": status.splitlines() if status else []}


def _versions() -> dict:
    out = {}
    for name in ("numpy", "jax", "jaxlib"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def build(*, command: list[str] | None = None, artifacts: list[str | Path] = (),
          configs: list[str | Path] = (), extra: dict | None = None,
          root: str | Path = ".") -> dict:
    root = Path(root).resolve()
    resolved_configs = []
    for path in configs:
        p = Path(path)
        resolved_configs.append({**artifact(p, root), "resolved": Config.load(p).to_dict()})
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command if command is not None else sys.argv),
        "cwd": str(root),
        "git": _git(root),
        "source_tree": source_tree(root),
        "runtime": {"python": sys.version, "executable": sys.executable,
                    "platform": platform.platform(), "packages": _versions(),
                    "pid": os.getpid()},
        "artifacts": [artifact(p, root) for p in artifacts],
        "configs": resolved_configs,
        "extra": extra or {},
    }


def write(path: str | Path, **kwargs) -> dict:
    data = build(**kwargs)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return data
