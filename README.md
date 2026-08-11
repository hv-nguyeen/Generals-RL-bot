# generals.bot competition bot

A bot for the [generals.bot](https://www.generals.bot) 1v1 competition, plus the
arena and analysis tooling to make it better.

**The submission is a neural policy**, not the heuristic — the current incumbent
is an 8-layer, 32-channel C=40/context network (about 79k parameters) with
observation-only temporal memory. The heuristic remains in
`bot/policy/controller.py` as a fallback and benchmark opponent
(`ours:configs/v16.json`), but is not the submission.

## Current training status — 2026-08-11

The accepted local champion is the stage-4-to-5 self-play checkpoint
`/local/data/vng205/top3-v3/selfplay-stage4-1000-node1/selfplay.npz`
(8x32, about 80k policy parameters), with its matched critic. Its policy
SHA-256 is
`3b3b707fc064635ae0448f021e0e0bef8136d963369b09caffe0c1aed637c028`.
It passed a 1,600-game final gate at `0.588` versus `0.500` initialization
and then beat the previous incumbent in 2,000 games at `0.578`
(`+54.6 Elo`, 95% CI `[+24.6,+85.5]`). This is local evidence, not a live
ladder or top-three claim.

The active experiment is a function-preserving 8x64 growth arm, initialized
from that accepted policy/critic:

- policy: `/local/data/vng205/top3-v3/champion-8x64.npz`
- critic: `/local/data/vng205/top3-v3/champion-8x64.critic.npz`
- run: `/local/data/vng205/top3-v3/selfplay-8x64-stage4-1500-node1-g256`
- recipe: stage 4 start, 1,500 iterations, 256 games/iteration, minibatch 4096

The 8x64 arm is not promoted yet. Keep the accepted champion immutable until a
candidate passes a fresh direct incumbent test, pinned snipe checks, and the
versioned promotion suite. `comp-eval` against weak bots is a progress signal,
not a promotion decision.

The complete current state, monitoring commands, memory policy, installation
invariant, and search limitations are in
[`docs/CURRENT-STATUS-2026-08-11.md`](docs/CURRENT-STATUS-2026-08-11.md).

**Read [`docs/CURRENT-STATUS-2026-08-11.md`](docs/CURRENT-STATUS-2026-08-11.md)**
first for operations, then [`docs/TOP3-V2-HANDOFF.md`](docs/TOP3-V2-HANDOFF.md)
for implementation and gate details.
[`docs/STATE.md`](docs/STATE.md) remains the measured history. This README is the
map of the tooling. [`docs/CLUSTER.md`](docs/CLUSTER.md) is how to run anything on
the VU box. [`docs/ml-log.md`](docs/ml-log.md) is the full measured history.

Design and rules analysis: [`docs/superpowers/specs/2026-08-03-generals-bot-design.md`](docs/superpowers/specs/2026-08-03-generals-bot-design.md).

## Quick start

```bash
make setup-cpu  # venv + package + numpy + CPU JAX (uses uv)
make test       # rule and belief tests
make verify     # simulator vs the official JAX engine, step for step
make bench      # 40 games vs the greedy baseline
```

## The loop you will actually use

**One command for the whole picture** — gauntlet, loss causes, expansion
fingerprint, where the turns go, timing. Prints ~40 lines of plain text, made to
be pasted straight back into a chat:

```bash
make diag WORKERS=32                    # full, a few minutes
make diag-quick WORKERS=8               # ~1 minute
make diag CONFIG=runs/tune/best.json VS=configs/v2.json   # A/B with paired test
```

Read it in this order: any `faults` is a bug that costs real games; `land@200`
under ~105 means we are behind the field on macro whatever the win rate says;
and in `WHERE THE TURNS GO`, a mode with a big share and a low `cap/turn` is
where the Elo is hiding. That table is what found the castle-gather bug.

Individual pieces, if you want them:

```bash
# 1. sweep one knob on identical boards
python -m tools.sweep first_expand_turn 18,24,30,36 --games 120

# 2. confirm a change is real, not noise
python -m arena.runner --a ours:configs/new.json --b ours --games 400 --workers 12

# 3. run a full gauntlet with replays, then read the report
make gauntlet
open runs/<timestamp>/greedy/report.html

# 4. search many knobs at once (this is the one for the university box)
python -m tools.tune --out runs/tune1 --iters 20 --pop 12 --games 40 \
    --opponents ours,greedy --workers 60

# 5. ship the exact intended neural checkpoint
EXPECT=$(sha256sum bot/weights.npz | awk '{print $1}')
make submit-test EXPECT="$EXPECT"  # builds, extracts, and plays the ZIP over stdio

# 6. promote a neural checkpoint (all buckets + hashes + runtime/fault gates)
python -m tools.evaluate --candidate runs/new.npz --reference runs/champ.npz \
  --out runs/eval/new-vs-champ --workers 12
```

`arena/runner.py` plays in-process against a numpy mirror of the competition
engine. Every seed is played twice with the colours swapped. Results now report
a pair-aware fixed-sample test; the old `sprt` label is not used for new results.
The promotion suite uses 2000 games per bucket; use `--games` only for a smoke
test.

The report classifies every loss (`early_rush`, `out_expanded`, `out_gathered`,
`blundered`, `timeout`, `draw`), names the config knob each cause points at, and
links a scrubable HTML replay of the game. That is the intended way to turn a
lost match into a change.

## Running on the university server

**[`docs/CLUSTER.md`](docs/CLUSTER.md) is the procedure** — jump host, node
mapping, the environment exports and why each exists, the missing tools and their
replacements, and a failure-symptom table. What follows is only what the rest of
this README would otherwise imply and get wrong.

`make setup` installs the package and numpy; `make setup-cpu` adds CPU JAX for
local verification. The cluster needs its pinned CUDA JAX instead; CLUSTER.md
has that install and the node-to-node fallback for a box with no network.

Two facts that cost hours to rediscover:

* **JupyterHub is not the compute node.** Same paths, same prompt, different
  machine and a different `$HOME`, and some Hub containers expose a single CPU —
  `nproc` before choosing `--workers`.
* **`/local/data` is per-node and nothing syncs it.** `$HOME` is shared. A
  head-to-head needs both checkpoints on one node.

`WORKERS` defaults to cores−2 and every tool takes `--workers`. The arena scales
close to linearly: 2000 games in ~85 s on 32 real cores.

Reports and replays are plain HTML with everything inlined, so
`scp -r runs/<run>/ .` and open them locally — no server or X forwarding needed.

Note `make tune-big` (CEM over the heuristic's knobs) is a **measured
non-starter** — PSRO best-response to the archive scored 0.522 ± 0.013 and the
~100-knob space is exhausted. It is kept because it produced `configs/v16.json`,
which is still the strongest non-neural opponent in the gauntlet.

## What the bot knows that a ported generals.io bot does not

Read out of the engine source, not the rules page:

1. **Castles are the biggest lever.** No neutral castles exist; you build them
   for 35 army plus `max(0, 14 − 2·d)` per own structure within 6 steps. A
   castle is +0.5 army/turn forever for **one turn** of investment, where
   expanding 34 tiles buys +0.68 army/turn for **34 turns**. Measured:
   `castle_enabled` off drops the score against `greedy` from 0.93 to 0.77.
2. **The full mountain map is known on turn 1.** Fogged mountains and fogged
   castles share type code 5, but the competition strips every neutral castle
   before the game starts, so on the first frame every type-5 cell is a
   mountain. Consequences: an exact `passable` map immediately; any later 0→5
   flip is a **newly built enemy castle**, visible anywhere on the board; and
   castles are passable, which the reference `expander` gets wrong.
3. **Spawn placement leaks the enemy general.** The generator seats general B
   uniformly among cells ≥17 BFS steps from A whose `room7` count is within 5 of
   A's. Both are computable from (2), which cuts ~290 candidates to ~20 before
   first contact.
4. **Deathtouch decides stalled games.** From turn 800 any move onto the enemy
   general wins. The counter is a *chase* — taking the attacker's source cell
   resolves first and cancels the touch.

## Layout

```
bot/            the submission. numpy only, self-contained
  policy/net.py   THE BOT: local conv trunk + global/regional context
  policy/guard.py win-in-one and deathtouch overrides on top of the net
  policy/        controller (the heuristic fallback), analysis, castle siting
  features.py   board -> 40 channels; one-frame + temporal, shared with training
  memory.py     observation-only temporal state used by net inference and data
  belief.py     richer heuristic memory and enemy-general prior
  config.py     versioned, fully materialized knobs; strict on file load
sim/            exact numpy mirror of the competition transition + map generator
learn/          the training stack (jax). selfplay is the one that produced the
                shipped policy; train/ is behaviour cloning, exploit/ an exploiter
arena/          parallel headless matches, agent registry, Elo + paired tests, stdio agent
analysis/       replays, per-game stats, loss classifier, HTML viewer and report
tools/          grow, evaluate + manifests, vprobe, package, verify_engine,
                sweep, tune (CEM), profile_turn
third_party/    the official starter kit, for the differential test
```

`bot/` may not import `sim/`, `arena/` or `learn/` — it is what gets uploaded.

### How the bot decides

**The submitted bot** is `bot/policy/net.py` — argmax over masked logits from a
conv net, wrapped in `bot/policy/guard.py` for win-in-one and deathtouch. That is
the whole decision procedure. `bot/main.py` ships the net when `bot/weights.npz`
is present and falls back to the heuristic otherwise, printing to stderr.

**Nothing removes `bot/weights.npz`** — not `tools/package.py`, not the Makefile.
It survives every build, so a later `make package` silently bundles the PREVIOUS
net, which is worse than shipping the heuristic because the zip looks correct.
Delete it yourself once the artifact is safely copied out:

```bash
cp dist/generals-bot.zip ~/
rm bot/weights.npz
```

`tools/package.py` requires the intended checkpoint digest and refuses to
publish a neural ZIP without a loadable network. Check it with `sha256sum
runs/nn/spN.best.npz`, pass that value through `EXPECT`, and keep the digest with
the submission date—it is the only way to tell two uploads apart afterwards.
An intentional heuristic-only artifact requires the explicit
`ALLOW_HEURISTIC=1` opt-out.

Note the guard's third override, the garrison veto, fires on about 0.2% of turns
and measures neutral — see docs/STATE.md. Five hand-written overrides have now
been measured against this policy and none has helped.

**The heuristic below is the fallback and the benchmark opponent**, not the
submission. Two tiers. Hard overrides first: a move that wins outright, the
deathtouch guard, and the castle build. Then scored move generation: every legal
move gets a 15-feature vector dotted with the weight block of the current mode.
Multi-turn coherence comes from precomputed distance fields, so a greedy pick per
turn still executes a coherent multi-turn march.

Mode selection is the strategy layer:

| mode | when | what `progress` measures |
|---|---|---|
| `DEATHTOUCH` | general located and turn ≥ 800 (or prep) | distance to their general |
| `DEFEND` | a visible stack out-races our home defence | distance to our general |
| `ATTACK` | strike force beats estimated defence | distance to their general |
| `GATHER` | saving for a castle, or staging a strike | distance to the site/front |
| `EXPAND` | default — land is income | distance to the nearest free tile |

## Correctness

`tools/verify_engine.py` steps the numpy mirror and the official JAX engine in
lockstep on random games — including builds, splits, out-of-range coordinates and
malformed action kinds — and asserts every state field matches. 60 games, 200
turns each, identical. `tests/test_all.py` covers the derived rules the policy
reasons with (castle pricing, growth phase, move-order tie-breaks, deathtouch and
its chase defence, mutual-capture draw, fog radius) and the belief inferences.

## Submitting

```bash
EXPECT=$(sha256sum bot/weights.npz | awk '{print $1}')
make submit-test EXPECT="$EXPECT"
```

Writes `dist/generals-bot.zip` (a directory with `run.sh` at its root, which is
the layout the dashboard expects), checks it against the 50 MB / 512 MB / 10,000
file limits, then plays it through the real wire protocol as a subprocess so the
artifact that gets uploaded is the artifact that was measured. Upload it from
your dashboard.

To ship an intentional heuristic config, use `make submit-test
CONFIG=runs/tune1/best.json ALLOW_HEURISTIC=1`. Neural releases must keep the
digest-pinned command above.

## Current standing

The last recorded leaderboard snapshot is historical, not a live claim. Keep the
leaderboard JSON and exact submitted ZIP with every release; local arena Elo does
not substitute for a fresh ladder sample. The current work has one accepted local
8x32 champion and one active 8x64 capacity arm; the latter is not a promoted
candidate until it passes the direct incumbent and promotion gates.
passed the full promotion suite.

Two things worth knowing before running anything:

**Local numbers disagree with the ladder, repeatedly.** Eleven local instruments
did, always in the same direction, before the twelfth finally agreed. No single
fixed opponent measures general strength however strong it is — benchmark against
several, always include a clone of the reigning champion (a score near 0.5 is
where Elo resolution is sharpest; at 0.886 one SE is ±60 Elo), and treat anything
under 2000 games as a shortlist rather than a verdict. 2000 games costs ~85
seconds.

**The submission's weights are not in this repo.** They live on the cluster under
the run directories, and `bot/weights.npz` is deleted after packaging on purpose.
A clean checkout therefore runs the heuristic — that is the intended state, not
a bug.
