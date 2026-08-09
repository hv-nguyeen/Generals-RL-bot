PY := .venv/bin/python
WORKERS ?= $(shell (nproc 2>/dev/null || sysctl -n hw.ncpu) | awk '{print ($$1>2)?$$1-2:1}')
OUT ?= runs/tune-big
GROUPS ?= opening,castle,combat
# greedy is saturated at ~0.96; hunter is what the leaderboard grades on.
OPPONENTS ?= ours,hunter
GAMES ?= 200

.PHONY: help setup setup-cpu test verify bench gauntlet report tune tune-big diag diag-quick package submit-test profile clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

setup:  ## create the venv and install deps (uv if present, else stdlib venv)
	@if command -v uv >/dev/null 2>&1; then \
	  uv venv --python 3.12 .venv && uv pip install --python $(PY) -e .; \
	else \
	  python3 -m venv .venv && $(PY) -m pip install -q --upgrade pip && $(PY) -m pip install -q -e .; \
	fi
	@$(PY) -c "import sys,numpy;print('python',sys.version.split()[0],'numpy',numpy.__version__)"

setup-cpu: setup  ## install the CPU JAX stack for local training and verification
	@if command -v uv >/dev/null 2>&1; then \
	  uv pip install --python $(PY) -e '.[verify]'; \
	else \
	  $(PY) -m pip install -q -e '.[verify]'; \
	fi
	@$(PY) -c "import jax;print('jax',jax.__version__,jax.devices())"

test:  ## rule and belief tests
	$(PY) -m tests.test_all

verify:  ## official-engine differential + stateless/temporal encoder parity
	$(PY) -m tools.migrate_configs --check configs/*.json
	$(PY) -m tools.verify_engine --games 40 --max-turns 200 --encoders
	$(PY) -m tests.test_all

bench:  ## quick sanity match, no files written
	$(PY) -m arena.runner --a ours --b greedy --games 40 --workers $(WORKERS) --max-turns 900

gauntlet:  ## full run vs every baseline, with replays, into runs/<date>
	@set -e; d=runs/$$(date +%Y%m%d-%H%M%S); mkdir -p $$d; \
	for opp in hunter greedy expander; do \
	  echo "== vs $$opp"; \
	  $(PY) -m arena.runner --a ours --b $$opp --games $(GAMES) --workers $(WORKERS) \
	      --out $$d/$$opp --replays; \
	  $(PY) -m analysis.report $$d/$$opp >/dev/null; \
	done; \
	echo; echo "reports:"; ls $$d/*/report.html

report:  ## rebuild the HTML report for a run: make report RUN=runs/.../greedy
	$(PY) -m analysis.report $(RUN)

tune:  ## CEM parameter search (this is the one that wants the big machine)
	$(PY) -m tools.tune --out runs/tune --iters 15 --pop 12 --games 40 \
	    --opponents $(OPPONENTS) --workers $(WORKERS)

tune-big:  ## long detached tuning run; resumable, safe to disconnect
	@mkdir -p runs
	nohup $(PY) -m tools.tune --out $(OUT) --iters 40 --pop 16 --games 60 \
	    --opponents $(OPPONENTS) --workers $(WORKERS) --groups $(GROUPS) \
	    > $(OUT).log 2>&1 &
	@echo "started; tail -f $(OUT).log   (resume after a kill: same command, it checkpoints)"

package:  ## build dist/generals-bot.zip. EXPECT=<sha256> fails on the wrong net
	$(PY) -m tools.package --name generals-bot $(if $(CONFIG),--config $(CONFIG),) $(if $(EXPECT),--expect $(EXPECT),) $(if $(ALLOW_HEURISTIC),--allow-heuristic,)

submit-test:  ## build once, extract the zip, and play it through the real wire protocol
	$(PY) -m tools.package --name generals-bot $(if $(CONFIG),--config $(CONFIG),) $(if $(EXPECT),--expect $(EXPECT),) $(if $(ALLOW_HEURISTIC),--allow-heuristic,) --test 4

diag:  ## full diagnostic report, paste-able. make diag WORKERS=32 [CONFIG=x.json] [VS=y.json]
	$(PY) -m tools.diagnose --workers $(WORKERS) \
	    $(if $(CONFIG),--config $(CONFIG),) $(if $(VS),--vs $(VS),) $(DIAGARGS)

diag-quick:  ## same, ~1 minute
	$(PY) -m tools.diagnose --workers $(WORKERS) --quick \
	    $(if $(CONFIG),--config $(CONFIG),) $(if $(VS),--vs $(VS),)

profile:  ## per-move timing distribution
	$(PY) -m tools.profile_turn --games 3

clean:
	rm -rf dist runs/tune/candidates
