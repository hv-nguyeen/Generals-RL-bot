# Final design review: counterfactual build supervision mixed into PPO

> **Status 2026-08-09:** deferred research design. The current r3 training arms
> do not use this term. The measured incumbent/critic continuation and neural
> league should be evaluated first; do not add counterfactual build shaping to a
> live run without repeating the pre-flight sign test and a matched ablation.

## 1. Verdict

**DO NOT BUILD** — not as a permanent refusal, but as a refusal to spend three weeks of cluster time before one hour of measurement.

**Single strongest reason:** the quantity the entire design rests on, `delta = z_build - z_control`, was measured twice during review and came back at or below zero — **-0.486 ± 0.163** (n=37) against a building control, **-0.123 ± 0.100** (n=57) against a never-building control, the honest sp44 analogue. `delta` is *deliberately not centred*, so that mean passes straight into the gradient: `L_pref = -mean(delta * log_pi(build))` with `E[delta] <= 0` pushes build probability **down**. The design's stated purpose is to push it up. This is not a loss of statistical power, it is a sign flip.

It is corroborated on-policy by an instrument this repo already prints every iteration: `bldA` — the mean normalised PPO advantage at real build actions, `learn/selfplay.py:1994` — reads **-1.2 on the exact BC-8x64 lineage sp44 comes from** under `tools/buildbias --bias 3.0` (`docs/STATE.md:41-44`). Two instruments, independent construction, same sign.

The revised design below is what gets built **if and only if** the one-hour pre-flight in §4 returns `E[delta | stratum A] > +0.10 z` at 3σ. Everything else in this document is conditional on that number.

---

## 2. Revised design

### 2.0 Gate

Do not write a line of the training patch until §4 clears. If `E[delta]` clears, build exactly what follows; if it does not, the design is dead and `docs/ml-log.md` gains a seventh entry.

### 2.1 Backend: `--backend cpu`, both arms, non-negotiable

The original design places fork generation "in `_rollout`, mode 0 only" (`learn/selfplay.py:597`). That branch (`:638-662`) is reached only through `play(build_jobs(...))`, which is the **else** arm of `learn/selfplay.py:1723-1724`:

```python
results = (play_train(it, stage) if vec_backend
           else play(build_jobs(it, args.games, stage, args.stage_replay), snapshot))
```

with `vec_backend = args.backend != "cpu"` at `:1573`. `docs/CLUSTER.md:310` — the copy-paste launch line — says `--backend gpu`. Launch it that way and the fork code executes zero times, the aux batch is empty, and the two paired arms are byte-identical. A node-night answering nothing.

**Decision: pin `--backend cpu` on both arms and write it into the launch line.** Three objections demanded a vecroll port instead; I am refusing it. The port is real but not free — an override-action argument threaded into the jitted `make_step` (`learn/vecroll.py:110`), a synthetic pool dict for a second `rollout()` call, and one extra XLA trace — and the blocker nobody has solved is that forks begin at a **different turn per column** while `vecroll.rollout` (`:179-200`) runs one lock-step batch to the longest game. That is a day of surgery to save 2.5 hours, on a design whose premise is unproven.

What cpu actually costs, correctly priced:

- **1.7x rollout**: 17 s/iter vs 10 s at 256 games / stage 0 (`docs/ml-log.md:711-717`). Scaling both by the stage-4 turn ratio (~420 vs ~156) gives ~46 s vs ~27 s. Over 200 aux iterations that is **~2.6 h per arm**, ~5.2 h for the pair. Against three weeks it is a rounding error.
- **Board source changes** from `tools/pools` (`SP_POOL_SEED0`, `:317`) to `build_jobs` seeds (`SP_TRAIN_SEED0`, `:313`) — disjoint blocks. Two objections called this a distribution change; it is not. `tools/pools.py:37-40` states a CPU/GPU A/B "shares the distribution, not the maps". Same mapgen, same stage bands, different sample. Since this experiment is internally paired on the same seeds, the confound cannot touch its own contrast — and comp-eval, stage-eval and the final gate always run on the CPU process pool over `SP_COMP_SEED0` (`:315`) regardless of backend, so the measuring instrument is byte-identical to sp44's. No re-baselining is required. Absolute comp-eval numbers remain comparable; only the training map sample differs.
- **`--dist-tail` is refused on cpu** (`:1319-1328`). It defaults to 0.0. Irrelevant.

Add at the top of the iteration loop, unconditionally:

```python
assert args.backend == "cpu", "cf arm requires the _rollout path"
```

### 2.2 Build-cell choice: argmax over legal build cells

Measured: `bot/features.py:216` gates the build slot on affordability alone —

```python
can = mine & ~structs & (a >= rules.build_cost_grid(structs))
```

— and over v16 self-play that set is **exactly one cell on 96.1% of build-legal turn-seats** (mean 1.00-1.01, max 2), holding the seat's largest non-general stack 95.8-99.7% of the time. The design's "explicitly unspecified" cell choice is therefore nearly moot.

**Decision: argmax of the policy's logits restricted to legal build cells.** It degenerates to "the one cell" 96% of the time, is deterministic (so the pre-flight and the run measure the same thing), and costs one `np.argmax` over a masked slice. No sampling, no uniform.

### 2.3 Fork-turn sampling: two strata, tracked separately

This is the real free parameter and the original design silently set it to the worst value. Measured over v16 self-play: **88.7 build-legal turn-seats per game at stage 4** against 1.20 builds/game — so a uniform reservoir lands on a moment a competent agent would build at **~1.4%**. Corroborated twice in repo: `tools/buildprior.py` ("19.2% of turns") and `docs/STATE.md:861` (sp8.best, `p(build|legal) = 0.0034`).

Restricting *entirely* to the champion's region is also wrong: it strips the negative examples that teach where **not** to build — the -36 Elo failure mode at ~15 castles/game (`docs/ml-log.md:46`) — and makes the estimand "imitate v16's `castle_gather_min_army`", a behaviour-cloning target smuggled into an RL run.

**Decision: stratify, two strata, one fork per game, coin-flip between strata when both are available, deltas never pooled.**

Stratum A ("could pay for itself") is the predicate `bot/config.py:116-121` already encodes by hand on the only castle-profitable agent in this repo:

```python
def _stratum_a(st, seat, r, c, turn, max_turns):
    return (turn >= 45                                       # castle_min_turn
        and max_turns - turn >= 2 * rules.BASE_COST          # payback horizon
        and int(rules.build_cost_grid(structs)[r, c]) == rules.BASE_COST  # castle_max_cost
        and _cheb_to_nearest_seen_enemy(st, seat, r, c) >= 7  # castle_safe_dist
        and own_land >= 12)                                  # castle_min_land
```

Stratum B is every other build-legal turn-seat. Reservoir-sample one candidate per stratum during the parent game, then flip a fair coin at game end to pick which one becomes the fork (falling back to whichever exists). Record `cf_stratum` on the payload. **Every delta report — pre-flight, iteration line, abort check — is per-stratum.** A pooled mean is not reported anywhere.

This resolves the objection's estimand complaint without turning the term into a BC target: A supplies the positive signal the design exists to find, B supplies the negatives that keep it from becoming "build always".

### 2.4 CRN: shared seed, both continuations re-rolled

The parent stream is `np.random.default_rng(seed)` with seat 0 drawn before seat 1 (`learn/selfplay.py:638-640`). "Seed the fork from the parent stream position" is not implementable in any meaningful sense — after the intervention the two branches occupy different states, consume draws against different-shaped distributions, and desynchronise on the first turn.

**Decision: at fork turn `t`, run BOTH continuations from `np.random.default_rng([seed, t])`.** The real episode is *not* the control any more; the control is a re-rolled continuation from the same snapshot with the same fresh stream. That gives genuine common random numbers (shared noise source, identical consumption order at t=0) and makes the pair exact.

Cost: two continuations per forked game instead of one. Measured median fork turn is 382 of a ~438-turn game, so continuations are short — under 15% overhead on the forked fraction. Both branches use `sim/engine.py:49-54` `State.copy()`, which deep-copies every array; the parent game continues on the original object untouched.

The real episode's trajectory is unaffected and enters PPO exactly as today.

### 2.5 Delta treatment

`z ∈ {-1, 0, +1}` so `delta ∈ {-2, 0, +2}`. Draws need no special handling — a draw on both branches is `delta = 0` and contributes nothing to `L_pref`, which is correct.

**Decision: divide by 2 (`delta ∈ [-1, +1]`) and do not centre.** The halving makes the stated weight 0.05 mean what it says relative to a `log_pi` term of order 1. Not centring is the original design's call and it is right — mean-centring would push half of all builds up even if builds are uniformly bad. That is exactly why the §4 pre-flight is a hard gate: with no centring, a negative mean is a negative gradient, full stop.

No clipping (the support is already bounded), no standardisation, no per-stratum re-centring.

### 2.6 Transport out of `_rollout`

`_pack` (`learn/selfplay.py:432-456`) returns **exactly two dicts or `None` for the whole game**, and every element of `results` must carry `idx`/`x`/`mask`/`logp` (`:1760-1770`). `selfcheck` asserts `len(got) == 2` and `[r["seat"] for r in got] == [0, 1]` at `:1021-1022`, so a third element fails `make verify`.

Worse, a fork returned as a full trajectory dict would pass `:1760-1770` silently, land in `episodes` (`:1770`), receive a GAE advantage at `:1798`, and be **trained on** — falsifying the design's own claim that the on-policy estimate is untouched, and breaking the zero-sum property that `_pack`'s docstring (`:443-446`) and the advantage-scaling comment (`:1804-1811`) both rest on. It would also corrupt `bld`/`bldA` (`:1975-1994`), which are computed over **all** results' `idxs` and only divided by `len(g0)` — i.e. it would inflate the exact metric the experiment is read by, regardless of what seat tag it carried.

**Decision: extra keys on `out[0]`, list length stays 2.**

```python
out[0].update({"cf_x": np.stack([x_succ_build]),   # 1 state, ~10.6 kB
               "cf_mask": np.packbits(mask_fork[None], axis=1),
               "cf_idx": np.int32(build_idx),
               "cf_logp": np.float32(lp_build_under_pi),
               "cf_delta": np.float32((z_build - z_ctrl) / 2.0),
               "cf_z": np.float32(z_build),
               "cf_stratum": np.int8(0 or 1),
               "cf_x_fork": np.stack([x_fork])})    # state AT the fork
```

Harvest in the parent with `[r for r in results if "cf_idx" in r]`. `r.pop("x")` (`:1765`) and `r.pop("mask", None)` (`:1771-1772`) touch only those keys, so `cf_*` survives to the harvest. Payload is 2 states plus scalars per forked game (~21 kB) — no preallocate-and-pop care needed. `_pack` returning `None` (`:450`) drops the fork with the game; accept it, it is under 1%.

### 2.7 Aux losses and how they are batched

Both aux terms go **inside** the existing jitted steps, not into a separate pass:

```
L_pref = -mean( cf_delta * log_pi(cf_idx | cf_x_fork) )   -> added to p_objective (:1466)
L_succ = mean( (tanh(V(cf_x)) - cf_z)^2 )                 -> added to v_step      (:1478)
```

`L_succ` matches `v_step`'s own form at `:1479-1481` exactly. **Both arms must run `--value-head scalar`** (the default, `:1171`); under `hlgauss` there is no `tanh` — `values_of_hl` (`:1528-1531`) returns a mean-of-categorical divided by `hl_kappa` — and `L_succ` would have to become the same KL-against-`hl_t[rint(ret+1)]` that `v_step_hl` uses at `:1507-1523`. Not worth the second code path on an unproven term. Assert `args.value_head == "scalar"`.

**Batching — the one place I overrule an objection outright.** One objection demanded the aux gradient be applied once per iteration or scaled by `mb/n`. Both divide the aux by ~52 against its stated 0.05 and manufacture the null result. Refused. The objection's arithmetic also inflates the harm: placing a term inside the loop at `:1914-1916` multiplies its **exposure count**, not its weight — the per-step loss ratio stays exactly 0.05, and the accumulated per-iteration ratio lands between 0.05 (coherent PPO gradients) and 0.05·√52 ≈ 0.36 (fully decorrelated), never 3.0. `clip_grads(g, 0.5)` (`learn/netoracle.py:282`) bounds every combined step and the per-minibatch `KL_STOP` break at `:1940` is the designed detector.

What is real is the memorisation risk on a small fixed row set — the `tools/mixbuilds.py` failure `tools/buildprior.py`'s docstring records (3,711 frames × 320 exposures into a 73k-param net, memorised, `p(build)` unchanged at 0.0009).

**Decision: accumulating counterfactual buffer, fixed-size minibatch resampled per gradient step.** The buffer is a ring holding the last 4096 forks. Each `p_step`/`v_step` draws a **fixed 256 rows with replacement** — fixed so XLA does not retrace on a ragged tail, the same reason `:1774-1790` pads its ingest chunks. At ~64 forks/iteration the buffer fills by iteration 64 and per-row exposure falls monotonically thereafter; by iteration 200 the arm has seen ~12,800 distinct forks, not 64.

Weights `--cf-pref-weight 0.05 --cf-value-weight 0.05`, linearly decayed to zero over `--cf-iters 200`.

Two required assertions, every iteration:

```python
assert len(cf_new) > 0, "empty fork buffer -- fork generation is not running"
```

and the realised gradient-norm ratio printed next to `gn` on the iteration line (`:2000-2011`), so the effective weight is observed rather than assumed. `mb {nb}` is already on that line, so exposure count is visible today.

### 2.8 `--stage-replay`

`build_jobs` already threads `replay` (`:460-495`) and picks a per-board stage; forks ride whatever board the job drew. Stage 3 `bldA` is -0.9 to -3.0 (`docs/STATE.md:152-156`) — a known-negative stratum — and the design itself agrees builds should be zero there.

**Decision: `--stage-replay 0.0` for both arms.** Single-stage fork distribution, no cross-stage pooling to argue about. Record `cf_stage` on the payload anyway and assert it equals 4 for every fork; if a future arm turns replay back on, the assert converts a silent contamination into a crash.

### 2.9 Resume

`save_resume` (`:666`) takes arrays and scalars. **The counterfactual buffer is not persisted.** A resumed run restarts with an empty buffer and refills it in 64 iterations; the grad-norm ratio print makes the ramp visible. Persisting a 4096×10.6 kB state buffer into an `.npz` to save one hour of refill is not worth the format change or the resume-compatibility risk.

`--resume runs/nn/sp43.resume.npz` supplies the critic, `--weights runs/nn/sp44.best.npz` the policy, unchanged from the original design.

### 2.10 Launch lines

```
# treatment
python -m learn.selfplay --backend cpu --start-stage 4 --stage-replay 0 \
  --value-head scalar --games 256 --seed 7 \
  --resume runs/nn/sp43.resume.npz --weights runs/nn/sp44.best.npz \
  --cf-frac 0.25 --cf-pref-weight 0.05 --cf-value-weight 0.05 --cf-iters 200 \
  --frozen-kill 10

# control -- identical, zero weights
... --cf-frac 0.25 --cf-pref-weight 0 --cf-value-weight 0 --cf-iters 200
```

The control still **generates and logs** forks at `--cf-frac 0.25` with zero weight. That costs ~15% wall clock and buys a second independent `E[delta]` estimate on a policy the aux term never touched — the only clean read of whether the treatment moved the causal effect or merely the rate.

---

## 3. Abort criteria

| # | When | Metric | SE | Threshold | Action |
|---|---|---|---|---|---|
| A0 | Every iteration, from iter 0 | `len(cf_new)` | — | `> 0` | Hard assert. Kills a dead-code launch in 46 seconds instead of a node-night. |
| A1 | Every iteration | `aux_gn / gn`, printed next to `gn` (`:2000-2011`) | — | `> 0.5` for 5 consecutive iterations | Abort. The exposure asymmetry has escaped the stated weight. |
| A2 | Every iteration | `do_policy` false (`[critic warmup n/10]` tag, `:1996-1998`) | — | 10 consecutive frozen iterations | `--frozen-kill 10`. This is the design's own primary hazard — five configurations have died here — and 10 is deliberately tighter than the default so the aux arm dies fast rather than freezing quietly. |
| A3 | **Iteration 20** | Cumulative mean `cf_delta`, **stratum A only**, n ≈ 640 (64 forks/iter × 20 × ~0.5 in A) | sd ≈ 0.50 in halved units → **SE ≈ 0.020** | Abort unless `mean_A > +0.05`, i.e. lower 2σ bound above 0 | The pre-flight measured this on a frozen policy; iteration 20 is the first read on the *live* one. If it has gone negative under training, the term is now actively harmful. |
| A4 | **Iteration 50** | Paired comp-eval, treatment minus control, identical `SP_COMP_SEED0` boards (`:315`) | ±0.022 at the standard comp-eval n (400 games/arm) | Abort if `treat - ctrl < -0.03` | Two arms, same boards, same seed — the pairing is the point. |
| A5 | **Iteration 100** | `bld` at stage 4, treatment vs control, mean over iterations 80-100 | ±0.04 over 20 iterations × 256 games | Abort if `bld_treat <= bld_ctrl` | The term exists to raise the build rate. 100 iterations at half the aux weight and no movement means it never will. |
| A6 | **Iteration 200** | Aux weight reaches zero by design; run continues on pure PPO. | — | — | Not an abort; the phase boundary. |
| A7 | **Final** | `python -m arena.runner --a <treat.best> --b <champion> --games 2000`, `faults == 0` | paired lower bound | **positive direct-incumbent lower bound plus the unchanged promotion suite** | Current runner is fixed-sample paired evidence; measure `.best`, never `.live`. |

A3 is the load-bearing one. It is the pre-flight re-run on live weights, and it is cheap because the numbers are already being logged.

---

## 4. Pre-flight — under one hour, kills the design before the run starts

Two measurements. Both on the VU box, both before any training patch is written.

### 4a. `tools/cfprobe.py` — measure `E[delta]` directly (~15 min)

`delta` is a pure rollout quantity: no gradients, no critic, no training loop. `_rollout` mode 0 drives the policy purely through `_act(net, engine.observe(st, s), rng)` (`:643-644`) with no belief or agent state, and `sim/engine.py:49-54` `State.copy()` deep-copies everything. Nothing resembling a forced-action counterfactual rollout exists anywhere in `tools/`, `analysis/`, or `arena/` — **this number has never been measured on this repo's own candidate.** The training loop would not produce it either; `:1970-2004` logs `bld` and `bldA` and nothing else.

```
python -m tools.cfprobe --weights runs/nn/sp44.best.npz \
  --stage 4 --forks 2000 --strata a,b --workers 32
```

Reuses `arena`'s process pool. ~50 lines: play the parent, reservoir-sample per stratum (§2.3), `State.copy()`, argmax build cell (§2.2), two CRN continuations (§2.4), report per-stratum mean/sd/SE and the diagnostic breakdown (burn fraction, captured-by-opponent rate, fork turn, break-even fraction).

Cost: an 8×64 numpy net runs ~12.7 ms per full turn, so a ~400-turn stage-4 game is ~5 core-seconds. 2000 forks × (1 parent + 2 short continuations) ≈ 2 core-hours ≈ **4 minutes on 32 cores**. Note the arena's "2000 games in 85 s" figure does not transfer — that is heuristic agents, not a stochastic net.

**Kill rule, pre-registered here:** proceed only if `E[delta | stratum A] > +0.10` (in raw z units, i.e. `> +0.05` halved) at 3σ. At sd ≈ 0.99 that needs SE < 0.033, i.e. **n_A ≥ 900** — hence 2000 forks split across two strata. If stratum A is at or below zero, or indistinguishable from it, **the design does not proceed.** Zero is as fatal as negative here, because `delta` is not centred.

Expected read, for calibration: review measured -0.123 ± 0.100 with a never-building control. That is one sigma below the gate.

Also record, because they decide whether fix (a) is even available: the fraction of stratum-A forks whose castle is **captured by the opponent before the game ends** (measured at 47-49% overall — `sim/engine.py:130-132` sets `own[p]` without clearing `st.castles`, and `_global_update` at `:215-220` then grows it for its new owner, so half of all forks hand the enemy permanent income), and the fraction of the stack burned (79.7% median). Both are correct competition rules, not repo bugs, so they are real costs of the intervention.

### 4b. `tools/vprobe` on **this** checkpoint pair (~10 min)

`tools/vprobe.py:22-24` pre-registered the decision rule for exactly this hypothesis, in advance:

> `dV(castle, free) > 0` → the critic HAS identified it and the problem is elsewhere. Do not build the [...] machinery on this diagnosis.

`docs/STATE.md:1081-1085` reports `+0.28` — but that is **sp9's** 8×32 self-play critic, trained where its own policy built 1.16-1.59 castles/game. This design uses `sp43.resume.npz`, the BC 8×64 critic warmed at stage 0 where `bld` is 0.00, and it has never been probed. The repo's own precedent (commit `547da6e`, "value24 does not transfer") is that critic readings do not carry across lineages. So the starvation premise is **unmeasured on the pair the design uses**, not refuted — the objection claiming otherwise generalised a checkpoint-specific number.

```
python -m tools.vprobe --resume runs/nn/sp43.resume.npz \
  --weights runs/nn/sp44.best.npz --games 24
```

**Kill rule:** if `castle, free > 0.005`, **drop `L_succ`** by the rule vprobe pre-registered at `:22-24`. `L_pref` survives — it runs off terminal outcomes and makes no claim about the critic. Also record the `paid` arm (`vprobe.py:121-125`, castle minus 35 army), which is the trade the fork actually makes and the one the repo projects at ~0.1 z (`docs/ml-log.md:942-945`); the `free` arm is the asset with nothing paid for it.

Total: **under 30 minutes of compute, under an hour end to end**, against a three-week run. This is CLAUDE.md's second cardinal rule applied literally.

---

## 5. Decision log

| # | Original | Revised | Why |
|---|---|---|---|
| 1 | Backend unspecified | `--backend cpu`, asserted, both arms | `_rollout` mode 0 (`:638-662`) is unreachable under `vec_backend` (`:1573`, `:1723-1724`), and `docs/CLUSTER.md:310` says `--backend gpu`. Cost of cpu is the measured 1.7x (`ml-log.md:711-717`) ≈ 2.6 h/arm. Cost of the vecroll port is a day plus the lock-step-turn problem. Rejected two objections' demand for the port: wrong trade on an unproven premise. Also rejected the claim that cpu changes the distribution — `tools/pools.py:37-40` says it changes the map *sample*, and comp-eval runs on the CPU pool over `SP_COMP_SEED0` (`:315`) regardless of backend, so no re-baselining is needed. |
| 2 | Build cell unspecified | Argmax over legal build cells | Measured: exactly one legal cell on 96.1% of build-legal turn-seats (`bot/features.py:216` gates on affordability alone). The choice is near-moot; argmax is deterministic and free. |
| 3 | Uniform reservoir over build-legal turns | Two strata (`bot/config.py:116-121` predicate vs everything else), coin-flip, deltas never pooled | Uniform lands on a payable moment ~1.4% of the time (88.7 legal turn-seats/game vs 1.20 builds at stage 4). Restricting *only* to the champion's region strips the negatives that teach where not to build (`ml-log.md:46`, -36 Elo at 15 castles/game) and makes the estimand a BC target. Split the difference; report separately. |
| 4 | "CRN: fork rng seeded from parent stream position" | Both continuations re-rolled from `default_rng([seed, t])`; the real episode is no longer the control | Stream-position CRN is not implementable — the branches diverge in state on turn one and consume draws against different distributions. Shared-seed re-roll is genuine CRN. Costs one extra short continuation per fork (median fork turn 382 of 438). |
| 5 | `delta ∈ [-2, +2]`, un-centred | `delta / 2 ∈ [-1, +1]`, still un-centred, draws untreated | Halving makes weight 0.05 mean what it says against a `log_pi` of order 1. Not centring is the original design's call and it is correct — which is precisely why §4a is a hard gate: with no centring, a non-positive mean is a negative gradient. |
| 6 | Fork transport unspecified | `cf_*` keys on `out[0]`, list length stays 2, harvest by `"cf_idx" in r` | A third list element breaks `:1760-1770`, `g0` at `:1955`, and `selfcheck`'s `len(got) == 2` (`:1021`). A fork returned as a trajectory would silently enter `episodes` (`:1770`), receive GAE at `:1798`, break the zero-sum property `_pack`'s docstring guards (`:443-446`), and inflate `bld`/`bldA` (`:1975-1994`) — the exact metric the run is read by. |
| 7 | "Added to the existing PPO / critic steps" | Accumulating 4096-fork ring buffer, fixed 256-row minibatch resampled per gradient step | Rejected the objection's fix (once per iteration, or scale by `mb/n`): both divide the aux by ~52 and manufacture a null. Also rejected its arithmetic — inside-loop placement multiplies *exposure*, not weight; the per-iteration ratio is 0.05 to ~0.36, not 3.0, and `clip_grads(g, 0.5)` (`netoracle.py:282`) plus `KL_STOP` (`:1940`) bound it. The real risk is `mixbuilds`-style memorisation of a tiny fixed row set (`tools/buildprior.py` docstring), which an accumulating buffer removes. Fixed minibatch size avoids XLA retraces. |
| 8 | `--stage-replay` interaction unspecified | `--stage-replay 0.0`, `cf_stage` recorded and asserted == 4 | Stage-3 `bldA` is -0.9 to -3.0 (`STATE.md:152-156`) — mixing in a known-negative stratum. Single stage, single question. |
| 9 | hlgauss interaction unspecified | `--value-head scalar` asserted | `L_succ`'s `tanh` form matches `v_step` (`:1479-1481`) only. Under hlgauss it must become the KL against `hl_t[rint(ret+1)]` (`:1507-1523`). Not worth a second code path on an unproven term. |
| 10 | Resume interaction unspecified | CF buffer not persisted; refills in 64 iterations, ramp visible in the grad-norm print | `save_resume` (`:666`) takes arrays and scalars. Persisting 4096 states to save one hour is not worth the format risk. |
| 11 | No pre-flight | Two probes, ~30 min compute, both with pre-registered kill rules | CLAUDE.md: "Measure a candidate before building it." `delta` has never been measured in this repo; `tools/vprobe` has never been run on the sp43/sp44 pair. |
| 12 | Motivation cites only `STATE.md` on castle rates | Must cite and address `STATE.md:1081-1085` (vprobe +0.28) and `STATE.md:41-52` (`bldA` -1.2, "castles cost that policy games") | Real review gap. Neither refutes the design outright — +0.28 is sp9's critic, and `bldA` at stage 4 reads -0.05 to +0.35, i.e. inconclusive exactly where this design runs — but a design doc that asserts "the critic cannot value castles" without engaging the measurement that says otherwise is not reviewable. |
| 13 | Control = the real episode, zero aux weight | Control also generates forks at `--cf-frac 0.25` with zero weight | ~15% wall clock for a second independent `E[delta]` on a policy the aux never touched. Without it you cannot tell whether treatment moved the causal effect or just the rate. |
| 14 | Scope of `L_succ` unstated | `L_succ` is dropped if pre-flight 4b returns `castle, free > 0.005`; `L_pref` is independent of the critic premise | `vprobe.py:22-24` pre-registered this. `L_pref` runs off terminal outcomes and survives a dead starvation hypothesis; the two halves have different premises and should have different gates. |

### On the two refuted objections

Both refutations hold. The `bldA`-makes-this-redundant objection is correctly refuted — `bldA` derives from `adv`, which derives from the critic's own V (`:1798`), so both rival explanations produce `bldA <= 0` and the instrument cannot say *why*; and its proposed one-hour grep is wrong on three counts (no `runs/nn` in this working copy, `$NF` picks up `util` appended at `:1961-1966` under `vec_backend`, and `--warm-evar` freezes the policy so a BC-lineage arm prints `bldA +0.00` for 30 iterations — `:1994` returns 0.0 when `builds == 0`, exactly what commit `cb48d50` records for sp46).

The mirror-opponent-bias objection is also correctly refuted, and its proposed fix — v16 in the forked seat — is convicted by its own citations (`ml-log.md:49-57`: v16 is the +74 Elo outlier *because* it builds ~0.9 castles a game, and "no single fixed opponent measures general strength"). Neither is reintroduced.

The residue from both is real and is now in the design: §2 cites `bldA` and vprobe as prior context (change 12), and §2.10 keeps the mirror rather than swapping in a fixed opponent.
