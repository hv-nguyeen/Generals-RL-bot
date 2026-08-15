#!/usr/bin/env bash
# Turnkey: bake off ensemble candidates vs the champion, confirm the leader on a
# large paired sample, and report the shippable spec. No training. Run on the
# node (topology build + gb312 env), or via: ssh gbnode 'bash -s' < this file.
#
#   bash tools/validate_ensemble.sh            # 800-game bake-off + 2000 confirm
#
# A candidate is a WIN only if the paired lower bound beats the champion
# (elo_lo > 0) with ZERO faults inside the 150 ms move budget.
set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CH=${CH:-/local/data/vng205/champion.npz}
SP=${SP:-/local/data/vng205/champ-selfplay/selfplay.best.npz}
TN=${TN:-/local/data/vng205/topo-night-best.npz}
WORKERS=${WORKERS:-32}
cd "${REPO:-/local/data/vng205/generals-bot-topology}" || { echo "no REPO dir"; exit 1; }

for f in "$CH" "$SP" "$TN"; do
  [ -f "$f" ] || { echo "MISSING member: $f"; exit 1; }
done

SPECS=(
  "shipens:${CH}+${SP}+${TN}@logit"
  "shipens:${CH}+${SP}+${TN}@prob"
  "shipens:${CH}+${SP}@logit"
  "shipens:${CH}+${TN}@logit"
  "shipens:${SP}+${TN}@logit"
)

echo "=== BAKE-OFF (800 games each) vs ship:champion ==="
best_spec=""; best_lo="-999"
for spec in "${SPECS[@]}"; do
  out=$(python -m arena.runner --a "$spec" --b "ship:${CH}" \
        --games 800 --workers "$WORKERS" --time-limit-ms 150 --quiet 2>&1)
  line=$(echo "$out" | grep -iE "elo" | head -1)
  faults=$(echo "$out" | grep -oiE "fault[s]?[ =:]+[0-9]+" | grep -oE "[0-9]+" | paste -sd+ - | bc 2>/dev/null); faults=${faults:-0}
  lo=$(echo "$line" | grep -oE "\[[-+]?[0-9.]+" | head -1 | tr -d '[')
  echo "  ${spec##*shipens:}  ->  ${line}  faults=${faults}"
  # track the best lower bound among zero-fault candidates
  if [ "${faults:-0}" = "0" ] && [ -n "${lo:-}" ]; then
    if awk "BEGIN{exit !($lo > $best_lo)}"; then best_lo="$lo"; best_spec="$spec"; fi
  fi
done

echo
if [ -z "$best_spec" ]; then
  echo "NO zero-fault candidate; all likely tie or fault. Next: arch-diverse ensemble (champion+16x96) on the residual-tail build."
  exit 0
fi
echo "=== LEADER: $best_spec  (bake-off lo=$best_lo). Confirming at 2000 games ==="
conf=$(python -m arena.runner --a "$best_spec" --b "ship:${CH}" \
       --games 2000 --workers "$WORKERS" --time-limit-ms 150 --quiet 2>&1)
echo "$conf" | grep -iE "elo|fault"
clo=$(echo "$conf" | grep -iE "elo" | head -1 | grep -oE "\[[-+]?[0-9.]+" | head -1 | tr -d '[')
cf=$(echo "$conf"  | grep -oiE "fault[s]?[ =:]+[0-9]+" | grep -oE "[0-9]+" | paste -sd+ - | bc 2>/dev/null); cf=${cf:-0}
echo
if [ -n "${clo:-}" ] && awk "BEGIN{exit !($clo > 0)}" && [ "${cf:-0}" = "0" ]; then
  echo "RESULT: MEASURED WIN. Ship spec -> ${best_spec}"
  echo "Package with: BOT_ENSEMBLE='${best_spec#shipens:}' (guard on, tta per config)."
else
  echo "RESULT: leader did NOT clear champion at 2000 games (lo=${clo}, faults=${cf}). Champion stays; go arch-diverse."
fi
