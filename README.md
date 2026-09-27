# Generals RL Bot

A competition bot for [generals.bot](https://www.generals.bot), with a local simulator, self-play training code, and paired evaluation tools. The project explores how a compact neural policy can play a partially observed, turn-based strategy game under a strict move-time budget.

The submission is the self-contained, NumPy-only [`bot/`](bot/) package. When a trained checkpoint is present at `bot/weights.npz`, it uses a convolutional policy with observation-only temporal memory and a small tactical guard. A clean checkout has **no checkpoint** and runs the heuristic controller instead. Training and evaluation need additional tools and, for neural work, JAX.

## What this project demonstrates

- **Game-engine fidelity:** [`sim/engine.py`](sim/engine.py) mirrors the official engine. [`tools/verify_engine.py`](tools/verify_engine.py) compares full game states and, with `--encoders`, the training and submission observations and legal actions.
- **A deployable policy boundary:** [`bot/main.py`](bot/main.py) handles the competition's line-oriented protocol and falls back safely if an optional checkpoint cannot load. The submission code does not depend on the training stack.
- **Controlled evaluation:** [`arena/runner.py`](arena/runner.py) plays each generated board twice with seats swapped. [`arena/rating.py`](arena/rating.py) computes uncertainty over board pairs rather than treating the two games as independent.
- **Research workflow:** [`learn/`](learn/) contains self-play and training experiments; [`analysis/`](analysis/) turns match results into replays and failure reports. Historical experiments and rejected ideas are recorded in [`docs/STATE.md`](docs/STATE.md) and [`docs/ml-log.md`](docs/ml-log.md).

## Results and availability

The last documented accepted **local** neural checkpoint won 57.8% of 2,000 games against the previous incumbent on paired boards, an estimated +54.6 Elo (95% interval +24.6 to +85.5). That result is from the August 2026 experiment recorded in [`docs/CURRENT-STATUS-2026-08-11.md`](docs/CURRENT-STATUS-2026-08-11.md). It is not a current leaderboard rank or a claim that a fresh checkout reproduces the trained bot.

The checkpoint and training data were stored outside this repository. To evaluate the neural policy, supply a compatible checkpoint; without one, the commands below evaluate the heuristic fallback. See [`docs/TOP3-V2-HANDOFF.md`](docs/TOP3-V2-HANDOFF.md) for the recorded checkpoint identity and promotion criteria.

## Quick start

Requires **Python 3.12 or newer**. Run these commands from the repository root:

```bash
make setup-cpu                # create .venv; install NumPy and CPU JAX
make test                     # rule, policy, and training-seam tests
make verify                   # compare the simulator with the vendored official engine
make bench WORKERS=2          # 40-game local smoke match against greedy
```

`make verify` uses the vendored official starter kit in [`third_party/generals-bots/`](third_party/generals-bots/). The full checks take longer than the local smoke match. For a quick look at the CLI without writing match files:

```bash
.venv/bin/python -m arena.runner --a ours --b greedy --games 4 --workers 1
```

`ours` selects the neural policy only when `bot/weights.npz` exists and the configuration enables it. Otherwise it selects the heuristic controller. Match counts must be even because each seed is played from both seats. A four-game result is only a functional smoke check.

## Repository map

| Path | Purpose |
| --- | --- |
| [`bot/`](bot/) | Submission entry point, protocol, features, temporal memory, neural policy, and heuristic fallback. NumPy only. |
| [`sim/`](sim/) | Local NumPy game engine and map generation. |
| [`arena/`](arena/) | Headless matches, agent selection, and paired ratings. |
| [`learn/`](learn/) | JAX training and self-play experiments. |
| [`analysis/`](analysis/) | Replay, diagnostics, and HTML match reports. |
| [`tools/`](tools/) | Verification, evaluation, checkpoint packaging, and research utilities. |
| [`tests/`](tests/) | Rule, packaging, policy, and training-seam checks. |
| [`configs/`](configs/) | Versioned heuristic configurations. |
| [`evaluation/`](evaluation/) | Versioned promotion-suite definition. |
| [`third_party/generals-bots/`](third_party/generals-bots/) | Official starter kit, vendored for engine comparison; see its [license](third_party/generals-bots/LICENSE). |

The decision path is: protocol frame → observation and temporal features → legal-action mask → policy logits → tactical guard → protocol action. [`bot/policy/net.py`](bot/policy/net.py) implements the trained policy; [`bot/policy/controller.py`](bot/policy/controller.py) implements the fallback.

## Evaluate and package

Run a paired local comparison and write results under the ignored `runs/` directory:

```bash
.venv/bin/python -m arena.runner --a ours --b hunter --games 200 --workers 4 --out runs/hunter
.venv/bin/python -m analysis.report runs/hunter
```

For a neural candidate, [`tools/evaluate.py`](tools/evaluate.py) runs the versioned promotion suite. A small smoke match does **not** establish an improvement. The project records its acceptance criteria in [`evaluation/top3-v2.json`](evaluation/top3-v2.json) and [`CLAUDE.md`](CLAUDE.md).

To build and test a submission, place the intended checkpoint at `bot/weights.npz`, compute its SHA-256 digest, and pin the build to that digest:

```bash
EXPECT=$(.venv/bin/python -c 'import hashlib; print(hashlib.sha256(open("bot/weights.npz", "rb").read()).hexdigest())')
make submit-test EXPECT="$EXPECT"
```

This creates `dist/generals-bot.zip` and plays the packaged bot through its real stdio protocol. To build an intentional heuristic-only artifact from a clean checkout, run `make submit-test ALLOW_HEURISTIC=1`. Build output under `dist/` and experiment output under `runs/` are ignored by Git.

## Research notes

The older documents in [`docs/`](docs/) are dated experiment records, not setup instructions for a new contributor. In particular, paths under `/local/data/` refer to the original training machine. The [design and rules analysis](docs/superpowers/specs/2026-08-03-generals-bot-design.md) explains game-specific choices; the [experiment log](docs/ml-log.md) records measured failures as well as successful ideas.
