#!/usr/bin/env bash
# Build a stamped release tarball. `git archive` strips .git, so write a VERSION
# file into the tree first — otherwise a pasted diagnostic says "git ?" and we
# cannot tell which build produced it.
set -euo pipefail
NAME="${1:-generals-bot-$(date +%Y%m%d-%H%M)}"
OUT="${2:-$HOME/Downloads}"
REV="$(git rev-parse --short HEAD)"
TMP="$(mktemp -d)"
git archive --format=tar --prefix=generals-bot/ HEAD | tar -x -C "$TMP"
printf '%s %s\n' "$REV" "$(date -u +%Y-%m-%dT%H:%MZ)" > "$TMP/generals-bot/VERSION"
tar -czf "$OUT/$NAME.tar.gz" -C "$TMP" generals-bot
rm -rf "$TMP"
ls -l "$OUT/$NAME.tar.gz" | awk '{print $5" bytes"}'
shasum -a 256 "$OUT/$NAME.tar.gz"
