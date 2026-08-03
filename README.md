# generals.bot competition bot

A bot for the [generals.bot](https://www.generals.bot) 1v1 competition, plus the
arena and analysis tooling to make it better. Heuristic today; the policy sits
behind a single `act(obs) -> action` seam so an RL policy can drop in later.

Design and rules analysis: [`docs/superpowers/specs/2026-08-03-generals-bot-design.md`](docs/superpowers/specs/2026-08-03-generals-bot-design.md).

## Quick start

```bash
make setup      # venv + numpy (uses uv)
make test       # rule and belief tests
make verify     # simulator vs the official JAX engine, step for step
make bench      # 40 games vs the greedy baseline
```

## The loop you will actually use

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

# 5. ship
make submit-test           # builds dist/generals-bot.zip AND plays it over stdio
```

`arena/runner.py` plays in-process against a numpy mirror of the competition
engine, so 300 games take about 30 seconds on a laptop. Every seed is played
twice with the colours swapped.

The report classifies every loss (`early_rush`, `out_expanded`, `out_gathered`,
`blundered`, `timeout`, `draw`), names the config knob each cause points at, and
links a scrubable HTML replay of the game. That is the intended way to turn a
lost match into a change.

## Running on the university server

Nothing in the repo is machine-specific. `.venv/`, `runs/` and `dist/` are
gitignored, so copy the tracked tree and rebuild the venv there.

```bash
# from the laptop (~13 MB, most of it the vendored starter kit)
rsync -az --exclude .venv --exclude runs --exclude dist --exclude __pycache__ \
    ~/VU/NewGame/ user@server:~/generals-bot/

# on the server
cd ~/generals-bot
make setup          # uses uv if present, otherwise python3 -m venv
make test           # 15 tests, ~10 s — proves the rules model survived the trip
make bench          # 40 games vs greedy, ~10 s
```

`make verify` additionally needs jax (`.venv/bin/python -m pip install 'jax[cpu]'`).
It is optional: it diffs the simulator against the official engine, so run it
after touching `sim/engine.py`, not on every box.

`WORKERS` defaults to cores−2 and every tool takes `--workers`. The arena scales
close to linearly — 300 games take ~30 s on 10 cores, so a 64-core box does a
2,000-game gauntlet in about a minute.

```bash
# detached, resumable, survives a disconnect
make tune-big OUT=runs/tune-a GROUPS=opening,castle
tail -f runs/tune-a.log

# it checkpoints every iteration; after a kill just run the same command again
```

Then bring the result back and confirm it on the laptop, or confirm it there:

```bash
python -m arena.runner --a ours:runs/tune-a/best.json --b ours --games 800 --workers 60
```

Reports and replays are plain HTML with everything inlined, so
`scp -r runs/<run>/ .` and open them locally — no server or X forwarding needed.

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
  policy/       controller (modes + scored moves), analysis, castle siting
  belief.py     terrain, fog memory, enemy-castle detection, general prior
  config.py     every tunable in one dataclass; flattens to a vector for tuning
sim/            exact numpy mirror of the competition transition + map generator
arena/          parallel headless matches, agent registry, Elo + SPRT, stdio agent
analysis/       replays, per-game stats, loss classifier, HTML viewer and report
tools/          sweep, tune (CEM), verify_engine, package, profile_turn
third_party/    the official starter kit, for the differential test
```

### How the bot decides

Two tiers. **Hard overrides** first: a move that wins outright, the deathtouch
guard, and the castle build — none of these are trade-offs. Then **scored move
generation**: every legal move gets a 15-feature vector dotted with the weight
block of the current mode. Multi-turn coherence comes from precomputed distance
fields, so a greedy pick per turn still executes a coherent multi-turn march.

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
make submit-test
```

Writes `dist/generals-bot.zip` (a directory with `run.sh` at its root, which is
the layout the dashboard expects), checks it against the 50 MB / 512 MB / 10,000
file limits, then plays it through the real wire protocol as a subprocess so the
artifact that gets uploaded is the artifact that was measured. Upload it from
your dashboard.

To ship tuned weights: `make submit-test CONFIG=runs/tune1/best.json` bundles
them as `bot/config.json`, which `bot/main.py` picks up automatically.

## Current standing

Against `greedy` (a competent flood-fill expander — a fair proxy for the median
entrant, and much stronger than the starter kit's `expander`, which deadlocks on
corner spawns): **271W 17D 12L over 300 games, score 0.932**. Mean move time
1.0 ms against a 150 ms budget, zero faults.

The known weakness is the land race: at turn 200 we hold ~76 tiles to greedy's
~112 and win on castle economy instead. Closing that is the highest-value work
left, and `tools/sweep.py` over the `expand` weight block is where to start.
