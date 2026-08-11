# Generals.bot competition bot — design

Date: 2026-08-03
Target: top 10 on the generals.bot leaderboard (Sprint checkpoint Aug 8, Marathon after).

> **Status 2026-08-11:** this is the original rules/design specification, not the
> current training recipe. Operational instructions and verified implementation
> status live in `docs/CURRENT-STATUS-2026-08-11.md`, `README.md`,
> `docs/TOP3-V2-HANDOFF.md`, `docs/STATE.md`, and `docs/CLUSTER.md`. The
> current incumbent is a C=40/context temporal neural
> policy; the active experiments split mirror self-play from neural league PPO.
> The accepted local checkpoint and active 8x64 capacity arm are recorded in
> `docs/CURRENT-STATUS-2026-08-11.md`; this specification is not a claim that
> learned belief search or the Ataraxos method is implemented.

## 1. Ruleset (ground truth, read from the engine source, not the rules page)

Source: `strakam/generals-bots`, `GeneralsEnv(mode="competition")`.

| Rule | Value |
|---|---|
| Format | 1v1, fog of war |
| Board | rectangle, each side drawn independently in [18, 21] |
| Mountains | 24–26% of area, uniform random placement |
| Neutral castles | none — 9–11 are carved out of mountains then **stripped to plain** |
| Castle build | action `[2, r, c, 0, 0]`; cost `35 + Σ_own_structures max(0, 14 − 2·manhattan_d)` |
| Growth | generals + castles +1 on **even** ticks; every owned tile +1 when `turn % 50 == 0` |
| Vision | 3×3 (Chebyshev radius 1) around every owned tile |
| Move | `all-but-one` (split=0) or `floor(army/2)` (split=1) |
| Combat | strictly-greater attacker takes the tile, keeps the difference; ties defend |
| Move order | `chasing > reinforcing > smaller-army-first`; loser of the order resolves **second** and therefore holds a contested tile |
| Deathtouch | from turn 800 any move executing onto the enemy general wins, army irrelevant |
| Mutual capture | draw, at every turn |
| End | 1200 turns → draw |
| Sandbox | 150 ms/move (first move 10 s), 1 core, 2 GB, no network, Python 3.12.10 / numpy 2.4.6 |
| Faults | late/malformed reply → pass + 1 fault; 50 faults forfeits |

## 2. Where the edge comes from

Four facts in the engine that a ported vanilla generals.io bot will not use.

**2.1 Castle economy.** A castle is +0.5 army/turn forever for **one turn** of
investment. Expanding 34 tiles buys +0.68 army/turn but costs **34 turns**. Turns
are the scarce resource (one move per turn), so a castle is ~25× more
turn-efficient than expansion. Build cost is flat 35 when sited ≥7 manhattan from
your own structures. Nobody who ports a generals.io bot builds castles at all.

**2.2 The whole mountain map is known on turn 1.** `structures_in_fog =
invisible & (mountains | castles)`, and the competition strips every neutral
castle before the game starts. So on turn 0 **every type-5 tile is a mountain**.
Consequences:

- exact `passable` mask for the whole board from the first frame;
- a tile that later flips 0 → 5 is a **newly built enemy castle** — free intel on
  where the enemy lives, and a raid target;
- the reference `expander` bot treats type 5 as impassable. It isn't; only
  mountains are. Anyone who copied it walks around its own castles.

**2.3 Spawn placement leaks the enemy general.** The generator seats general B
uniformly among cells that are (a) BFS-reachable at ≥ 17 steps from A over
passable ground, and (b) within 5 of A's `room7` count — the number of cells
reachable within 7 steps. Since we know the mountain map exactly (2.2), we can
compute both fields on turn 1 and cut the enemy general down to ~20–30
candidates before seeing a single enemy tile.

**2.4 Deathtouch decides every stalled game.** From turn 800 any touch wins.
A bot that pre-stages a unit near the enemy general and keeps its own general's
neighbourhood clear wins every game that would otherwise draw.

Plus the unglamorous one that decides most matches: don't die to an early rush,
and never fault.

## 3. Architecture

```
bot/            the submission — self-contained, numpy only
  main.py       stdio loop, deadline guard, never faults
  protocol.py   frame parse
  belief.py     persistent memory across turns
  board.py      BFS / dilation / distance fields (numpy)
  rules.py      engine model: castle cost, growth phase, combat, move order
  config.py     every tunable weight in one dataclass
  policy/
    controller.py  mode selection + scored move generation
    modes.py       EXPAND / GATHER / ATTACK / DEFEND / DEATHTOUCH
    castle.py      build siting and timing
sim/            exact pure-numpy mirror of the competition transition
arena/          parallel headless matches, in-process agents, timing enforcement,
                Elo + pair-aware fixed-sample tests
analysis/       replay format, per-game stats, loss-cause classifier,
                self-contained HTML replay viewer and report
tools/          tune.py (CEM parameter search), verify_engine.py (differential
                test vs the official JAX engine), profile_turn.py
```

### 3.1 Decision model

Two tiers, not a full plan system.

- **Tier 1 — hard overrides.** Emergency defense (an enemy stack can reach the
  general before we can defend it), the deathtouch endgame, and a committed
  kill attempt. These bypass scoring entirely.
- **Tier 2 — scored move generation.** All legal moves are scored against a
  feature vector whose weights live in `Config`. Multi-turn coherence comes from
  precomputed distance fields (distance to nearest unowned tile, to the staging
  tile, to the enemy general estimate, to our own general), so a greedy pick per
  turn still executes a coherent multi-turn march.

The strategic layer is **mode selection**, which swaps the weight set:

```
DEFEND      an enemy stack out-races our home defense
DEATHTOUCH  turn ≥ 800 − margin
ATTACK      enemy general located and strike force > estimated defense
GATHER      army advantage, consolidating toward a staging tile
EXPAND      default
```

Castle building is evaluated separately each turn before move scoring, because it
is a different action kind.

### 3.2 Belief state

Persistent across turns: exact mountain/passable map (turn 1), fog memory
(last-seen owner/army/turn per tile), inferred enemy castles (0 → 5 transitions),
our own castle ledger, and the enemy-general candidate mask — seeded from §2.3
and narrowed by every tile we observe that is not a general and by proximity to
observed enemy territory.

## 4. Iteration loop (this is the actual deliverable)

Everything above is worthless without a measurement loop, so the tooling is
first-class:

1. `arena/runner.py` plays N games headless in parallel over the numpy mirror,
   in-process (no subprocess), enforcing the 150 ms budget and recording a
   replay per game. Replays store only the initial grid plus both action
   streams — the sim is deterministic, so a replay is a few kB.
2. `arena/rating.py` gives Elo and a statistical verdict, so "B is better than A"
   is a claim over paired fixed-sample evidence, not a vibe. Older design notes
   call this an SPRT; current output is pair-aware and fixed-sample.
3. `analysis/report.py` turns a run into an HTML report: win rate, Elo, loss
   causes, land/army curves, time-per-move histogram, castle timings.
4. `analysis/viewer.py` renders any single game to a self-contained HTML replay
   with a scrubber — works over SSH from the university box, no pygame.
5. `tools/tune.py` runs CEM over the `Config` vector against a fixed gauntlet,
   sharded by seed and resumable, for the university cluster.

Loss-driven iteration: `report.py` classifies each loss (rushed, starved,
out-macroed, timed out, drew) and links the replay, so "what went wrong" is one
click, and a fix is a weight change plus a paired fixed-sample run.

## 5. Correctness

- `tools/verify_engine.py` plays random games through the numpy mirror and the
  official JAX engine in lockstep and asserts identical state every step.
- Unit tests: castle cost surcharge, growth phase, move order, deathtouch
  resolution and its chase defense, mutual capture draw, fog visibility.
- `arena --stdio` runs the real `bot/run.sh` through the wire protocol so the
  submitted artifact is what was measured.

## 6. Original non-goals for the first implementation

The first implementation intentionally had no RL training code and no torch
dependency. That boundary has since been extended by the JAX stack under
`learn/`; the current neural training and promotion workflow is documented in
the r3 handoff, not in this original design specification. The stable bot seam
remains one interface (`Policy.act(obs) -> action`) plus a feature encoder the
analysis layer can inspect.
