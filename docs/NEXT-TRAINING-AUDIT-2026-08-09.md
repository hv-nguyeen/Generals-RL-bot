# Next-training audit and continuation experiments — 2026-08-09

> **Superseded operational state (2026-08-11):** the accepted local champion is
> /local/data/vng205/top3-v3/selfplay-stage4-1000-node1/selfplay.npz
> (policy SHA-256 3b3b707fc064635ae0448f021e0e0bef8136d963369b09caffe0c1aed637c028).
> It passed its final gate at 0.588 versus 0.500 initialization and beat the
> previous incumbent by +54.6 Elo over 2,000 fresh games. The active experiment
> is a function-preserving 8x64 growth arm from stage 4, 1,500 iterations,
> 256 games/iteration. Use [CURRENT-STATUS-2026-08-11.md](CURRENT-STATUS-2026-08-11.md)
> for current commands; this file remains the 2026-08-09 audit record.

This is the verification handoff for the source after checkpoint commit
`5997415` (`fix: harden neural training and promotion`). It separates reproduced
correctness defects from strategic hypotheses. None of the code changes below
is evidence that playing strength improved; only fresh terminal games and then
a sufficiently large ladder sample can establish that.

## Execution status

The r3 bundle was installed and `make verify` passed on both nodes. The finished
incumbent was used as the starting point for two divergent arms; neither earned
promotion:

| arm | method | current status |
|---|---|---|
| Node 1 `selfplay-node1` | competition-distance mirror self-play | unaccepted; Stage 5 was reached by the 200-iteration cap, not the stage gate |
| Node 2 `ceiling-r2` | neural league PPO against archive mixture | completed and rejected: trained 0.539 vs init 0.504; required +0.071 |
| Node 1 `ceiling-r1` | earlier neural league duplicate | interrupted around iteration 39; `.best/.live` only, no gate |

`--opp` in `learn.selfplay` is evaluation-only. Only `learn.league --oracle
net` trains against its archive mixture. A candidate must be tested directly
against the incumbent before any claim about a higher ceiling.

## Executive decision

The ladder-tested ZIP remains the iteration-150 artifact; it is not the current
incumbent. The safe source install, incumbent identity check, fresh snipe audit,
and first neural pilot have already been completed. The current decision loop is:

1. keep the immutable incumbent and its matched critic unchanged;
2. retain the rejected Node 2 run as negative evidence and decide whether Node 1
   is worth finishing;
3. test any viable Node 1 `.best` checkpoint directly against `ship:$P` on fresh paired
   games, then run pinned snipe and the full promotion suite;
4. add only accepted candidates as `clone:` members of a new league for
   cross-training.

No replay shard rebuild is needed for this first pilot. No garrison rule,
banking override, castle reward, territory reward, or fixed early defence has
been added.

## Artifact identity: do not mix these models

| Artifact | Identity / status |
|---|---|
| Local ladder ZIP | `generals-bot-refresh150.zip` |
| ZIP SHA-256 | `ece970c17894faa3105acd7ece5db14b656a38887b049b752f418a4e09c13a21` |
| ZIP policy SHA-256 | `53f5b2630726b94ebb5b2c64d99be9f37350e9528edf3da8c1f3680e715a4ee3` |
| Finished policy | `runs/top3-v2/incumbent-refresh-onpolicy.npz` |
| Finished policy SHA-256 | `cb8d2bfa9cff1f03f6b6f67245e322f149bf7e958a03e0c1c489e7fd4099f386` |
| Matching finished critic | `runs/top3-v2/incumbent-refresh-onpolicy.critic.npz` |
| Mutable incumbent alias | `runs/top3-v2/incumbent.npz`; verify it is byte-identical to the finished policy |
| Archived old incumbent | `runs/top3-v2/incumbent-sp16-c24-archive.npz`, SHA-256 `f7b17e600fb920f9b9b1fad2d5ec2ffb93ef21c7810bbde8f45dff533932a332` |

The iteration-150 ZIP contains 18 macOS AppleDouble `._*.py` entries. They are
not the strategic cause of its losses, but they prove the old release path did
not enforce a strict archive member set.

## Reproduced findings, ordered by severity

### P0: temporal enemy castles were missed on never-seen cells — fixed

Both `bot/memory.py` and `learn/rlenv.py` required stale opponent ownership
before classifying a new structure-in-fog as an enemy castle. A castle built
wholly in fog therefore remained invisible in temporal channel
`ENEMY_CASTLE`. NumPy/JAX parity could not catch the bug because both sides had
the same error.

Implemented:

- removed the stale-owner condition in both encoders;
- added semantic NumPy and JAX tests for never-seen cell → SIF;
- preserved first-frame provisional-mountain semantics and same-turn
  idempotence;
- made the engine verifier require activity in `ENEMY_CASTLE`.

### P0: release packaging could publish the wrong or broken bot — fixed

The previous packager only warned on a missing network, archive limits, and
runtime faults. It tested the staging directory rather than an extracted ZIP.
The stdio adapter ignored its deadline and hid fallback tracebacks behind valid
PASS actions.

Implemented:

- neural packages require a loadable `bot/weights.npz` and `--expect SHA256`;
- intentional heuristic and unpinned development builds require explicit
  opt-outs;
- `use_net=false`, digest mismatch, limits, CRC/layout drift, faults, fallback,
  tracebacks, and stdio deadlines fail closed;
- macOS sidecars/caches are excluded and forbidden;
- the canary runs `run.sh` from an extracted pending ZIP;
- the final ZIP is atomically published only after all enabled checks pass;
- `submit-test` builds once instead of rebuilding its dependency;
- the final ZIP SHA-256 is printed.

### P1: neural-oracle critic readiness was permanently latched — fixed

`learn/netoracle.py` could never refreeze after `warmed=True`. It now uses the
same five-consecutive-failure standing state machine as self-play, rejects NaN
or all-draw buffers, logs refreeze/recovery, and keeps `--warm-evar 0` as an
explicit fixed-warmup compatibility mode. Netoracle now also logs early,
midgame, and late evar plus mid/late scalar controls.

Important limitation: policy readiness still uses aggregate evar. The proposed
single-buffer rule `mid_evar <= scalar_mid => freeze` is **not implemented**.
The successful +106 Elo run itself contains individual buffers where aggregate
evar was healthy but midgame evar tied or trailed the scalar control. A strict
one-buffer gate would have rejected known-good learning. The next version needs
rolling, game-clustered or cross-fitted evidence, with the window and margin
exposed and preregistered.

### P1: actor rewind restored an incompatible critic — fixed

Self-play already stored the critic selected with the best actor, but rewound
only actor weights. Netoracle stored no selected critic at all.

Implemented:

- both trainers restore matched actor and critic snapshots;
- both actor and critic Adam states and bias-correction step counters are reset;
- critic step/readiness is reset and must recover on current games;
- netoracle saves `<oracle>.critic.npz` in the existing `phi__*` warm-start
  format;
- league rejects an “accepted” neural oracle if that critic is missing;
- neural league runs now require `--nn-init-critic`.

### P1: paired-board evaluation used per-game uncertainty — fixed

The runner correctly swaps seats on each seed, but rating and promotion treated
both games as independent. Promotion now aggregates a score in `[0,1]` for each
board pair and uses a 95% empirical-Bernstein interval over independent board
pairs. This is conservative, remains finite after a sweep, and responds to
positive/negative within-board correlation. The old per-game normal LLR is no
longer called an SPRT in promotion output; the decision is a fixed-sample
pair-aware confidence test. Raw W/D/L is still reported.

This changes historical interval values, including the printed interval around
the +106 Elo incumbent result. The point estimate is unchanged. Re-run any
artifact whose interval is promotion-critical.

### P1: reduced smoke runs could approve promotion — fixed

Any `tools.evaluate --games ...` override now writes:

- `smoke: true`;
- `approval_eligible: false`;
- `passed: false` even after a sweep;
- a separate `operational_passed` result.

Evaluation manifests now hash `summary.json` and `results.jsonl`, and summaries
record the exact bucket seed origins.

### P1: source identity omitted executable dependencies — fixed

The source-tree hash now includes relevant `third_party`, `evaluation`, and
`scripts` files and common native/shell/YAML/Rust suffixes. Environment
variables remain excluded, so credentials are not recorded.

### P1: army-3 split was incorrectly masked — fixed

Both policy masks required army 4 for a split. At army 3, full and split are
already distinct and valid (move 2 versus move 1). NumPy and JAX masks now use
army 3, with an engine-backed regression.

## Confirmed but deliberately not implemented in this tranche

These are real gaps, but they are not prerequisites for the one-oracle rush
pilot and broadening the patch would raise cluster-install risk.

1. **BC legality/objective schema.** BC shards contain only `x/y`; training and
   validation use an unmasked 3,970-way softmax; checkpoints are selected by
   top-1 rather than masked NLL. A correct schema migration must also update or
   reject `tools/distil.py`, `tools/onpolicy.py`, and `tools/mixbuilds.py`.
   Distil/onpolicy currently omit TemporalMemory as well. Do not implement this
   only in `dataset.py/train.py` and silently mix incompatible shards.
2. **Honest final critic test.** `valuetrain` uses one complete-game holdout both
   for epoch selection and the final gate. It needs source-stratified
   train/selection/test games. Its neural metrics/loss are game-balanced, but
   the scalar control is fitted by unweighted rows, so long games dominate the
   comparator.
3. **Phase-aware readiness.** Diagnostics exist; a statistically justified
   rolling control-relative gate does not.
4. **Forced stage-cap promotion.** Forced promotion remains an explicit fixed
   schedule. The implemented mitigation invalidates readiness immediately on
   every stage change, so no actor step occurs until the critic passes on the
   new distribution. Stop-versus-adaptive-mixture remains an ablation.
5. **Progressive neural roots.** League oracles still start from the fixed
   `--nn-init`. Matched oracle critics now make a future progressive option
   possible, but fixed-root versus progressive-root has not been tested.
6. **PSRO mixture deployment.** Sigma is still discarded in favor of a pure
   runoff member. Per-game policy sampling is not implemented and has no fresh
   worst-case comparison yet.
7. **Winner type.** Network/config/external winner branching is still
   ambiguous. Do not infer acceptance from `oracle-nn-0.npz` existing. Require
   `oracle-nn-0.json["accepted"] == true`, archive membership, and the accepted
   log line. Use the oracle file explicitly for this pilot, not `winner.*`.
8. **Multiple-attempt selection.** Smoke is fixed, but the promotion suite does
   not yet register repeated candidate attempts. Freeze one candidate, then use
   a new confirmation seed block before release.
9. **Evaluation-only archive members.** Current netoracle uses the same archive
   for training and raw-sigma eval/gate. Pinned snipers in `--seeds` would train
   too. Keep them external for this pilot.

## Cluster preflight: inspect the actual final pair

```bash
cd /local/data/vng205/generals-bot

export PY=/local/data/vng205/venv/bin/python
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

P=runs/top3-v2/incumbent-refresh-onpolicy.npz
V=runs/top3-v2/incumbent-refresh-onpolicy.critic.npz

test -f "$P" && test -f "$V"
cmp "$P" runs/top3-v2/incumbent.npz
sha256sum "$P" "$V" runs/top3-v2/incumbent.npz

"$PY" -c 'from bot.policy.net import Net; import sys; n=Net(sys.argv[1]); print("FINAL_LOAD_OK", n.arch)' "$P"
```

Expected policy SHA-256 is
`cb8d2bfa9cff1f03f6b6f67245e322f149bf7e958a03e0c1c489e7fd4099f386`.
Record the critic digest; do not substitute the old calibrated offline critic.

## Experiment 0: test the finished policy before changing it

These are 400 independent boards / 800 games per opponent. The updated runner
prints the pair-aware interval and fixed-sample result.

```bash
"$PY" -m arena.runner \
  --a "ship:$P" --b "snipe:$P" \
  --games 800 --workers 60 --seed0 9000000 \
  --out runs/top3-v2/final-vs-snipe-random

"$PY" -m arena.runner \
  --a "ship:$P" --b "snipe:$P@120" \
  --games 800 --workers 60 --seed0 9010000 \
  --out runs/top3-v2/final-vs-snipe-120

"$PY" -m arena.runner \
  --a "ship:$P" --b "snipe:$P@160" \
  --games 800 --workers 60 --seed0 9020000 \
  --out runs/top3-v2/final-vs-snipe-160
```

Decision rule, chosen before reading results:

- if a matchup's **upper** 95% score bound is below 0.50, the weakness is
  confirmed;
- if its interval overlaps 0.50, extend that exact matchup to at least 2,000
  games on a new seed block before choosing a training intervention;
- if its **lower** bound is at least 0.50, that sniper does not justify changing
  the final policy;
- one weak 16-game result is diagnostic evidence, not a training gate.

## Experiment 1: one fixed-root neural-oracle pilot

Run only if Experiment 0 confirms a weakness. Randomized snipe is the sole rush
training member. `@120` and `@160` remain external tests. `clone:$P` preserves an
exact economy anchor, while v16/greedy/hunter/expander are regression anchors.

The floor `0.50` means that with six archive members every minority member gets
at least `0.50 / 6 = 8.3%` of training games (about 21 of 256 per iteration).
Read both startup lines: raw sigma Neff describes eval/gate diversity; floor-
mixed Neff describes training diversity.

```bash
cd /local/data/vng205/generals-bot

RUN=runs/top3-v2/rush-pilot-r1
test ! -e "$RUN" && test ! -e "$RUN.log"

nohup "$PY" -m learn.league \
  --dir "$RUN" \
  --out "$RUN/winner.json" \
  --oracle net \
  --nn-no-fallback \
  --nn-init "$P" \
  --nn-init-critic "$V" \
  --seeds "clone:$P,snipe:$P,ours:configs/v16.json,greedy,hunter,expander" \
  --iters 1 \
  --pair-games 96 \
  --runoff-games 192 \
  --nn-iters 100 \
  --nn-games 256 \
  --nn-epochs 1 \
  --nn-minibatch 4096 \
  --nn-lr 1e-4 \
  --nn-critic-lr 1e-4 \
  --nn-warm-evar 0.10 \
  --nn-sigma-floor 0.50 \
  --workers 60 \
  --seed 9 \
  > "$RUN.log" 2>&1 &

echo $! | tee "$RUN.pid"
```

`--nn-no-fallback` is important: a rejected neural hypothesis must not spend the
iteration on config CEM and then appear to support adversarial neural training.

Monitor:

```bash
grep -E 'neural oracle:|raw sigma support|training mixture:|^iter |^eval |RE-FREEZE|COLLAPSE|^gate |accepted|REJECTED|fallback disabled' "$RUN.log" | tail -n 80
```

Mandatory interpretation:

- `neural oracle:` must show `learn.netoracle` and the exact policy/critic paths;
- raw and floor-mixed Neff must both be read; floor-mixed diversity does not
  make a raw point-mass gate diverse;
- actor updates must show finite evar and no unresolved refreeze/collapse;
- `oracle-nn-0.npz` existing is not acceptance;
- accept only if `oracle-nn-0.json` says true **and** the member appears in
  `league.json` **and** the log says `oracle-nn-0 accepted`;
- its continuation critic is `nn/oracle-nn-0.critic.npz`.

## External pilot validation

If accepted, compare the neural oracle—not `winner.*`—on fresh blocks:

```bash
C="$RUN/nn/oracle-nn-0.npz"
CV="$RUN/nn/oracle-nn-0.critic.npz"
test -f "$C" && test -f "$CV"

"$PY" -m arena.runner --a "ship:$C" --b "ship:$P" \
  --games 2000 --workers 60 --seed0 9100000 \
  --out "$RUN/oracle-vs-final"

"$PY" -m arena.runner --a "ship:$C" --b "snipe:$P@120" \
  --games 2000 --workers 60 --seed0 9110000 \
  --out "$RUN/oracle-vs-snipe-120"

"$PY" -m arena.runner --a "ship:$C" --b "snipe:$P@160" \
  --games 2000 --workers 60 --seed0 9120000 \
  --out "$RUN/oracle-vs-snipe-160"
```

The pilot succeeds only if rush survival improves and the candidate retains a
non-negative pair-aware lower bound against `ship:$P`. Then run the unmodified
full promotion suite against the final policy. Do not promote from these three
diagnostics alone.

## Research backlog with ablations and falsifiers

| Proposal | Mechanism | Cost / rollout position | Falsification criterion |
|---|---|---|---|
| Phase-aware standing gate | Rolling, game-clustered or cross-fitted midgame evar gain over an identically weighted scalar control | Low/medium; instrument first, then A/B aggregate-only vs phase-aware on identical seeds | Gate does not predict later collapse/Elo better, or rejects the known +106 run without reducing collapse |
| Masked BC schema | Store packed legal masks, assert recovered labels legal, optimize masked NLL, select by NLL, report action/phase categories | Medium; coordinated schema-v2 migration across all shard producers before the next new BC lineage | Better masked NLL fails to improve closed-loop initialization across fresh terminal games |
| Honest critic test split | Source-stratified train/selection/test complete games; identically game-weighted scalar control | Medium; before trusting another offline critic | Test-set board gain/calibration fails although selection metrics pass, or gate does not predict live-buffer readiness |
| Asymmetric actor-critic | Privileged full-state critic reduces variance while actor sees only deployable observation/memory; no critic input reaches actor inference | High; after pilot, start as critic-only A/B | Same actor seeds show no faster readiness, lower gradient variance, or terminal gain; any actor dependency on privileged state is an immediate failure |
| Competition-state restarts | Train terminal outcomes from real/self-play states near turns 100/250/650/800 with exact reconstructed memory, mixed with full games | High data/engine work; likely the strongest next curriculum after critic validation | Phase buckets improve but full-initial-distribution promotion regresses, or reconstructed continuation diverges from replay |
| Lambda 0.98–0.995 | Longer unbiased terminal credit horizon without proxy rewards | Low after a reliable critic; identical-seed sweep | Worse KL/collapse/readiness or no terminal gain versus 0.95 |
| Progressive neural roots | Start selected oracles from accepted policy+critic pairs while retaining fixed-root diversity | Medium; only after matched-critic artifacts are proven | Progressive arm collapses diversity/robust minimum or fails to compound fresh terminal gains |
| Equilibrium deployment | Sample one frozen member per game instead of averaging logits | Medium/high packaging and sample-size cost; first compare fresh mixture minimum to best pure | Mixture lower bound does not beat pure member, artifact exceeds limits, or randomization is not reproducible per game |
| Learned belief/global model | GRU/temporal transformer plus multiscale global trunk while deterministic memory remains a verified safety layer | High; after objective/data fixes, not as a width-only test | No phase-bucket or terminal gain under matched compute; inference violates budget |
| Hierarchical masked policy | Factor source then conditional direction/split/build, reducing competition among unrelated actions | High and coupled to BC schema | Worse masked NLL/action recovery or no closed-loop gain |
| Simultaneous-response search | Batched top-K own actions versus modeled top-K opponent responses, conservative scoring | Last; requires calibrated critic and opponent model | Runtime/fault gate fails or fresh promotion is no better than the base actor |

## Verification record

Before final handoff, run and record:

```bash
git diff --check
.venv/bin/python -m py_compile \
  bot/memory.py bot/features.py learn/rlenv.py learn/netoracle.py \
  learn/selfplay.py learn/league.py arena/rating.py arena/runner.py \
  arena/stdio_agent.py tools/package.py tools/evaluate.py tools/manifest.py
.venv/bin/python -m tests.test_all
make verify
```

Final local verification:

- `git diff --check`: clean;
- changed-module `py_compile`: passed;
- official-engine differential: 40/40 games identical;
- NumPy/JAX encoder maximum error: `1.2e-07`;
- temporal seam: 1,668 frames and all 16 planes active;
- build coverage: 661 legal-build frames and 192 executed builds;
- repository tests and trainer self-checks: `42/42 passed`.

The `policy: FALLING BACK to heuristic` line intentionally emitted during the
test run is the synthetic negative stdio regression; the test asserts that this
line becomes a fault instead of being hidden as a valid action.

## What this work does not claim

- It does not show that the final iteration-850 model beats the ladder-tested
  iteration-150 model on the ladder.
- It does not prove that snipe is the right exploiter; historical data says the
  random wrapper fired rarely and was mostly identical to its inner policy.
- It does not guarantee improvement, top-three placement, or even acceptance of
  the pilot.
- It does make the next result more interpretable: exact model identity,
  compatible critic continuation, standing safety gates, pair-aware inference,
  and fail-closed release behavior.
