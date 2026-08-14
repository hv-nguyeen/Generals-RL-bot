# Replay topology audit & frontier-awareness — 2026-08-14

Branch: `topology-frontier-safety-guard`. Author: automated RL session.

## 1. Motivation (the replays)

Two ladder losses, same failure:

* **229256** — reached 29 land, then stalled; a 27-army stack sat in a pocket
  walled by mountains on three sides and never backtracked.
* **229242** — reached only 5 land and held it ~450 ticks; a 49-army stack was
  stranded in a five-cell mountain pocket.

Both are **valid-action strategic failures**, not process faults. The net keeps
choosing not to walk a large stack back out to open ground, so it stalls.

## 2. Audit of the existing movement / frontier / guard / legal code

* Legality (`bot/features.legal_mask`): a move is legal off any owned tile with
  army ≥ 2 into any non-mountain neighbour. Mountains are the only terrain block.
* Deploy overrides (`bot/policy/guard.GuardedPolicy`): win-in-one, deathtouch,
  and a narrow garrison veto, applied by re-scoring the net and walking the
  argsort order. This is "the one place the net cannot learn."
* **No topology signal anywhere.** Neither the 43-channel observation nor the
  guard represents connectivity or distance-to-frontier. The net is blind to the
  invariant these replays violate — confirmed against `docs/ml-log.md` L747/L748:
  *"receptive field is the ceiling"* / *"the observation is the ceiling."* A
  local 3×3-headed conv cannot compute reachability at any width.

## 3. Design decision: teach, don't guard

A deploy-time guard was considered and **rejected** after measurement:

* A "veto entering a sealed pocket (∞ frontier distance)" rule was prototyped
  and measured on the actual replay geometries. It **never fires**: to move
  *into* a cell it must connect to your stack, so that cell shares your
  component; `dist=∞` requires your *entire* reachable region already captured,
  which the real pockets are not. Measured distances: 229256-style = 1, 229242-
  style = 5, a hand-sealed variant = 5. All finite. The ∞-veto is a no-op.
* The only deploy fix that works is a forced backtrack ("stall-break"), which is
  a hard override. Every hand-written override to date has measured neutral or
  negative (worst −322 Elo). Owner decision: **no guard — teach the net with RL
  to raise its ceiling, not a work-around.**

So the fix is a perception plane + an RL teaching signal. No deploy override was
added; `bot/policy/guard.py` is unchanged.

## 4. Implementation

One shared definition of "trapped," used by perception and by the teacher.

### 4.1 Primitive — `bot/board.frontier_field(passable, owned)`
BFS distance over passable ground to the nearest **unowned** passable cell
(enemy + neutral + fog all count as frontier). Sentinel `H*W` where no frontier
is reachable. Built on the existing `bfs_field`; `step_toward` walks a trapped
stack back out. numpy-only; `bot/` stays self-contained.

### 4.2 Perception — `FRONTIER_DIST` input plane (arch change)
* `bot/features.py`: `STRATEGIC_C` 3→4, new `FRONTIER_DIST` channel, `C` 43→44.
  Plane = `clip(frontier_field / (2*PAD), 0, 1)`; a stack deep in a captured
  dead-end reads ~1, a tile on the expansion edge reads 0. `owned` is current
  visibility (our tiles are always visible via dilate8); fog counts as frontier.
* `learn/rlenv.strategic_planes_jax`: exact JAX mirror using the existing
  `_bfs_field_jax`. Same passable set and clip as `dist_home`, so the
  clip-at-2*PAD parity argument carries over.
* Migration: `LEGACY_INPUT_CHANNELS` gained 43, so the shipped 43-channel
  champion warm-starts into the 44-channel arch by zero-padding conv0
  (`net.py:359`, `netoracle.py:1227`). The new plane starts neutral and training
  learns it. **A new net must be trained to benefit; the champion is unchanged.**

### 4.3 Teaching — topology counterfactual arm in `learn/netoracle.py`
Extends the *existing* counterexample-defense machinery (the codebase's own
chosen vehicle over reward shaping, which failed six times in `ml-log`):

* `_frontier_trappedness(obs)` — army-weighted mean normalized frontier distance
  (0 = all army on the edge, 1 = marooned).
* `_topology_risk(obs)` — a large stack (`--topology-min-army`) stranded past
  `--topology-min-trapped`. A **training gate on lost games only**, not a veto.
* `_branch_score` gains a mobility penalty `-w * trappedness`; the short-engine
  counterfactual then labels the move that walks the stack back to frontier as
  the safe action, and `defense_pairwise_loss` ranks it above the stalling move.
* New CLI arm `--topology-counterfactual` (+ `--topology-mobility-w`,
  `--topology-min-trapped`, `--topology-min-army`). **Opt-in and byte-identical
  when off**: `topology_mobility_w=0.0` guards the branch-score term, and the
  label gates/consumers require the flag.

## 5. Verification (local, this session)

* `frontier_field`: 4 unit tests (open board, sealed pocket → sentinel, dead-end
  → finite + `step_toward` points out, routes around mountains).
* Perception plane: `test_frontier_plane_encodes_normalized_distance_to_open_ground`
  asserts the plane equals an independent `frontier_field` computation, range
  [0,1]. `test_a_checkpoint_states_its_own_architecture` fixture updated (a
  width one short of C is only "stale" if not a migratable legacy width).
* Teaching helpers: `test_topology_teaching_flags_trapped_stack` — trapped stack
  reads more trapped than a mobile one; `_topology_risk` fires on the former
  only.
* **Encoder parity gate**: `verify_engine --encoders` (8 games, 9 board shapes)
  → encoder max|d| **1.2e-07**, temporal seam max|d| 1.2e-07, "8 games identical
  to the official engine." The new plane matches numpy↔JAX bit-for-bit.
* Full suite: **58/58 passed** (with jax installed). Defense arm unchanged
  (`netoracle.selfcheck` OK).

## 6. Promotion status — NOT PROMOTED

Nothing here is claimed as an improvement. Per non-negotiables, promotion
requires a fresh independent comparison + the versioned suite. Pending:

1. `make verify` full (40 games) as a final engine gate.
2. Cluster: warm-start the champion into 44-channel arch and train the topology
   arm; monitor comp-eval only.
3. Fresh paired arena: champion vs 36×64-best vs the new trained net, 2000
   games/bucket, pair-aware lower bound, zero faults, runtime margin.
4. Targeted replay-style pocket maps to confirm the stall behavior is gone.
5. No-TTA package (`tta=false`, `tta_full=false`) only if a candidate wins.

The champion `selfplay-champion-gen1` remains the accepted policy until then.

## 7. RESULT (measured on the cluster, 2026-08-14) — NOT PROMOTED

Trained on node 2 (L4). Encoder parity re-proved on the node (max|d| 1.2e-07).
Warm-started the champion (43->44 conv0 zero-pad, policy + critic) via
`learn.league --oracle net`, --nn-games 384 --workers 32
(OPENBLAS_NUM_THREADS=1 mandatory).

| run | config | outcome |
|-----|--------|---------|
| topo-r1 | topology on, mobility-w 0.2, aux 0.3 | train-wr collapsed 0.55->0.17, auto-rewound. Harmful. |
| topo-control | aux 0.0 (perception plane ALONE) | stable 0.58-0.65; gate vs champion 0.4888 vs 0.4963, +0.0707 needed -> REJECTED. **Neutral.** |
| topo-r2 | topology retuned (mobility 0.05, min-army 40, min-trapped 0.7) | labels did NOT drop (~800/iter), train-wr collapsed 0.58->0.38 again. |

**Wiring flaw found:** `_counterfactual_label` gates on `_defense_risk OR
_topology_risk`; with `--defense-aux-weight>0` (required to enable the aux loss)
the defense hidden-army risk fires ~800x/iter regardless of the topology
thresholds, so the topology arm could not be tested in isolation and the aux
loss at 0.3 is what collapses the policy (aux 0.0 control was stable).

**Verdict:** perception plane = neutral vs champion; counterfactual aux teaching
at 0.3 = collapses. Champion `selfplay-champion-gen1` remains accepted. Code
preserved on branch `topology-frontier-safety-guard` (private repo
hv-nguyeen/generals-bot). This line of work is closed as neutral/negative; see
`docs/ml-log.md` for the retry preconditions if ever revisited.
