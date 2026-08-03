PY := .venv/bin/python
WORKERS ?= 10
GAMES ?= 200

.PHONY: help setup test verify bench gauntlet report tune package submit-test profile clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

setup:  ## create the venv and install deps (uv)
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) numpy
	@echo "optional, for tools/verify_engine.py:  uv pip install --python $(PY) 'jax[cpu]'"

test:  ## rule and belief tests
	$(PY) -m tests.test_all

verify:  ## differential test against the official JAX engine
	$(PY) -m tools.verify_engine --games 40 --max-turns 200

bench:  ## quick sanity match, no files written
	$(PY) -m arena.runner --a ours --b greedy --games 40 --workers $(WORKERS) --max-turns 900

gauntlet:  ## full run vs every baseline, with replays, into runs/<date>
	@set -e; d=runs/$$(date +%Y%m%d-%H%M%S); mkdir -p $$d; \
	for opp in greedy expander random; do \
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
	    --opponents greedy --workers $(WORKERS)

package:  ## build dist/generals-bot.zip
	$(PY) -m tools.package --name generals-bot $(if $(CONFIG),--config $(CONFIG),)

submit-test: package  ## build the zip and play it through the real wire protocol
	$(PY) -m tools.package --name generals-bot $(if $(CONFIG),--config $(CONFIG),) --test 4

profile:  ## per-move timing distribution
	$(PY) -m tools.profile_turn --games 3

clean:
	rm -rf dist runs/tune/candidates
