# Running this on the VU cluster

Everything here was learned the hard way. Follow it in order.

## Getting on

Compute nodes are not reachable directly — jump through the data host. The MOTD
banner maps the names:

```
1.compute.vu.nl = pcoms008a
2.compute.vu.nl = pcoms008b
3.compute.vu.nl = pcoms009a
4.compute.vu.nl = pcoms009b
```

```bash
ssh -J vng205@ssh.data.vu.nl vng205@1.compute.vu.nl
```

The banner also prints GPU_AVAIL per node. **Pick one with ~22G free.** Two JAX
processes on one L4 OOM at ~22 GB, and the second one dies with
`RESOURCE_EXHAUSTED`.

`/home/vng205` IS shared between compute nodes, so a tarball uploaded once is
visible from all of them. `/local/data/vng205` is NOT — each node has its own,
and the two clusters have identical paths with different contents.

## JupyterHub is not the compute node

The Hub terminal runs in a container with its OWN `/home/vng205`, and it is not
necessarily on the node your job is on. Same prompt, same paths, different
machine. Always start there with:

```bash
uname -n
```

```bash
ls /local/data/vng205/generals-bot/runs/nn/
```

If the log you are looking for is absent, you are on the wrong node — SSH in
instead. Logs and `runs/` only exist where the job was launched.

Two more consequences. A job started over SSH is not the Hub shell's child, so
`kill %1` and `jobs` cannot see it; use the `/proc` scan below. And a Hub session
that has touched JAX holds GPU memory that `nvidia-smi` reports with **"No
running processes found"**, because it cannot see across containers — so do not
train from the Hub, and stop the server when a node looks mysteriously full.

## Install a tarball

Upload to home (JupyterHub, or scp with the jump host). If you used the Hub,
`ls ~` from the SSH shell FIRST and confirm the file is actually there — the two
homes are not guaranteed to be the same one. Then:

```bash
ls -lh ~/generals-bot-YYYYMMDD-HHMM.tar.gz
```

Check the byte count against what was quoted. Home has a ~3 GB quota that has
silently truncated uploads three times.

```bash
sha256sum ~/generals-bot-YYYYMMDD-HHMM.tar.gz
```

```bash
tar -tzf ~/generals-bot-YYYYMMDD-HHMM.tar.gz >/dev/null && echo OK
```

```bash
tar -xzf ~/generals-bot-YYYYMMDD-HHMM.tar.gz -C /local/data/vng205
```

The tarball has a top-level `generals-bot/` directory, so `-C /local/data/vng205`
lands it in the right place. It carries no `runs/`, so extracting over an
existing tree keeps every checkpoint and log.

## Bringing up a NODE THAT HAS NOTHING

Only `/home/vng205` is shared. `/local/data` is per-node, so a fresh node needs
the venv, the repo, the board pools and the checkpoints. In that order.

`make setup` installs numpy ONLY — `bot/` is numpy-only by design and the
training stack's jax+CUDA is not in the Makefile. Pin the versions off a working
node first so the new one matches:

```bash
$PY -c "import jax, jaxlib, numpy; print(jax.__version__, jaxlib.__version__, numpy.__version__)"
```

```bash
mkdir -p /local/data/vng205
```

```bash
python3 -m venv /local/data/vng205/venv
```

```bash
/local/data/vng205/venv/bin/python -m pip install "jax[cuda12]==VERSION" numpy
```

```bash
/local/data/vng205/venv/bin/python -c "import jax; print(jax.devices())"
```

Wants `[CudaDevice(id=0)]`. **If pip cannot reach the network**, copy the venv
node-to-node — it CANNOT go through home, which has a ~3 GB quota against the
venv's 5.5 GB:

```bash
rsync -a --info=progress2 /local/data/vng205/venv vng205@OTHERNODE:/local/data/vng205/
```

Then the repo (tarball through home), and the pools, which regenerate locally in
a few minutes and are deterministic, so they match every other node exactly:

```bash
$PY -m tools.pools --out runs/pools --workers 60
```

Checkpoints are ~300 KB each and go through home. From a node that has them:

```bash
cp runs/nn/sp9-i600.npz runs/nn/sp9-i500.npz runs/nn/sp3-i500-0733.npz runs/nn/sp9.resume.npz ~/
```

`field/` (~86 MB of harvested replays) is only needed for behaviour cloning, not
for RL. Do not re-harvest it — the fetcher got the cluster IP 403'd once.

**After 2026-08-05 the encoder is 22 channels and every older checkpoint is 20.**
`net.py:181` refuses the mismatch on purpose. Migrate before running anything:

```bash
$PY -m tools.grow --net runs/nn/CKPT.npz --out runs/nn/CKPT-c22.npz --layers 8 --channels 32
```

```bash
$PY -m tools.grow --net runs/nn/RUN.resume.npz --out runs/nn/RUN-critic-c22.npz --prefix phi --layers 8 --channels 32
```

The second is the CRITIC, which lives inside a resume file under `phi__*` and
needs `--prefix`. Without it `--init-critic` silently has nothing to load and the
run starts blank — which is what killed three runs before the flag existed.

## Going back a version

The encoder width is the only thing that makes an old checkpoint unloadable, so
the tags are the encoder boundaries:

```bash
git tag -n1
```

```
c20-nn10       C=20. What nn10 on the ladder was built from. Every checkpoint
               from before 2026-08-05 evening loads here with no migration.
c22-hidden     C=22. GARRISON + HIDDEN_OPP (hidden army separates 6.8x).
c24-maxstack   C=24. + MAX_STACK_MINE/OPP (3.1x and 2.2x).
```

To run an old build exactly as it was, use a WORKTREE rather than checking out —
a checkout would break every job running from this tree:

```bash
git worktree add /local/data/vng205/gb-c20 c20-nn10
```

Then run from `/local/data/vng205/gb-c20`, which has its own `bot/features.py`
at the old width and needs no migrated checkpoints.

**Do not `git checkout` a tag on a node with a run in flight.** The process has
its code in memory and survives, but it writes checkpoints at its own width, and
anything that auto-resumes picks up the new checkout and dies on a shape error.

Going FORWARD needs no worktree — `tools.grow` pads any narrower stem to the
current `features.C` with zero columns, so 20 -> 24 in one hop is fine.

## Environment — every new shell

```bash
cd /local/data/vng205/generals-bot
```

```bash
export PY=/local/data/vng205/venv/bin/python
```

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

Without these OpenBLAS spawns 64 threads per worker and the node kills the
process.

They matter for the arena too, and there the failure is silent rather than
loud. `arena/runner.py` turns any move over 150 ms into a fault and a forced
pass, and forfeits the game at 50 of them. Unset, 32 workers oversubscribe the
box to a ~350 ms mean move and every game ends as a timeout race that scores
0.500 — a real result, in the sense that it is reproducible and wrong.

**Always read the `faults` line before the Elo line.** Anything but 0 means
throw the run away.

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Without this JAX grabs ~75% of the GPU regardless of what it needs, so one small
job locks out the whole card and any leftover process strands 21 GB.

```bash
unset JAX_PLATFORMS
```

**Unset for GPU work. Set to `cpu` for arena runs and analysis**, so a CPU job
cannot take memory from a training job:

```bash
export JAX_PLATFORMS=cpu
```

## Verify

```bash
make test PY=$PY
```

Wants `21/21 passed`.

```bash
$PY -m tools.verify_engine --games 40 --max-turns 200 --encoders
```

Wants `ok: 40 games identical to the official engine`. This checks the encoders
too, not just the engine.

```bash
$PY -c "import jax, jax.numpy as jnp; print(jax.devices()); print(jnp.ones((1000,1000)).sum())"
```

Wants `[CudaDevice(id=0)]` and `1000000.0` with no OOM lines.

## Things this box does not have

`ps`, `pgrep`, `pkill`, `hostname`. Use these instead:

```bash
uname -n
```

```bash
for p in /proc/[0-9]*; do tr '\0' ' ' < $p/cmdline 2>/dev/null | grep -q "learn.selfplay" && echo "${p#/proc/}  $(tr '\0' ' ' < $p/cmdline | cut -c1-90)"; done
```

```bash
for p in /proc/[0-9]*; do tr '\0' ' ' < $p/cmdline 2>/dev/null | grep -q "learn.selfplay" && kill "${p#/proc/}"; done
```

`kill %1` works for jobs of the current shell. `nvidia-smi` reports memory and
utilisation but **cannot see PIDs from other containers** — memory in use with
"No running processes found" means someone else's job, possibly your own in a
JupyterHub container.

### Killing a run leaves its 60 workers behind — ALWAYS sweep after

`kill` on the selfplay parent does **not** take the `ProcessPoolExecutor`
children with it. Each orphaned run leaves ~60 `spawn_main` processes running
for ever, and they keep competing for cores.

On 2026-08-06 six killed runs had accumulated **366 orphans** against 60 cores.
The live run's iteration time went `23.5s -> 85.0s -> 469.2s`, a 20x slowdown
with every component inflated together (`roll` 33x, `d2h` 56x). It looks exactly
like a bug in the training loop and is not one. Memory was fine, so the giveaway
is `/proc/loadavg` far above the core count.

Orphans reparent to PID 1, which is how you tell them from the live run's own
workers. Count them:

```bash
for p in /proc/[0-9]*; do tr '\0' ' ' < $p/cmdline 2>/dev/null | grep -q multiprocessing && [ "$(awk '{print $4}' $p/stat 2>/dev/null)" = "1" ] && echo x; done | wc -l
```

Kill them — safe while a run is live, since its workers have a live parent:

```bash
for p in /proc/[0-9]*; do tr '\0' ' ' < $p/cmdline 2>/dev/null | grep -q multiprocessing && [ "$(awk '{print $4}' $p/stat 2>/dev/null)" = "1" ] && kill "${p#/proc/}"; done
```

`loadavg` is a decaying average and lags minutes behind; judge the fix by the
iteration time in the log, not by `loadavg`.

### A deleted log is still readable while the writer lives

`rm runs/nn/spN.log` on a running job does not stop it writing — the file is
unlinked but the descriptor is open. Recover through `/proc`:

```bash
for p in /proc/[0-9]*; do tr '\0' ' ' < $p/cmdline 2>/dev/null | grep -q "spN" && cp /proc/${p#/proc/}/fd/1 runs/nn/spN-recovered.log; done
```

Also: `tail -3 file` fails on this shell. Use `tail -n 3`, one file per command.

## Running a training job

```bash
nohup $PY -m learn.selfplay --init CHECKPOINT.npz --out runs/nn/spN.npz --backend gpu --workers 60 --iters 1200 --warm-evar 0.05 --stage-cap 400 --games 256 > runs/nn/spN.log 2>&1 &
```

`nohup` + `&` survives the SSH session dropping. **Launch it once** — two
instances writing the same `--out` clobber each other's checkpoints and split
the box.

```bash
sleep 300; grep -E "^iter|^comp-eval" runs/nn/spN.log | tail -n 4
```

Confirm `devices: [CudaDevice(id=0)]`, the expected stage, and iterations
ticking. `--backend gpu` is the fastest measured; `scan` is slower and `cpu` is
the safe default.

Watch it with:

```bash
grep -E "^comp-eval|STAGE|KILL|COLLAPSE" runs/nn/spN.log | tail -n 20
```

`comp-eval` is the only progress number. `.best.npz` is rewritten whenever it
improves, so killing the run at any point keeps the best policy.

## Packaging a submission

```bash
cp runs/nn/CHECKPOINT.npz bot/weights.npz
```

```bash
$PY -m tools.package --name generals-bot-nnN --config configs/v13.json --test 4
```

Confirm it prints `policy: net {...}` and not the heuristic fallback, and that
the slowest move is well under 150 ms.

```bash
cp dist/generals-bot-nnN.zip ~/
```

```bash
rm bot/weights.npz
```

**That last line matters.** Leave `bot/weights.npz` in place and every future
package silently ships as a net.

Then from your Mac:

```bash
scp -J vng205@ssh.data.vu.nl vng205@1.compute.vu.nl:~/generals-bot-nnN.zip ~/Downloads/
```

## Moving data between nodes

Node-to-node ssh works but asks for a password. `/local/data` is not shared:

```bash
rsync -a --info=progress2 /local/data/vng205/field pcoms008a:/local/data/vng205/
```

`field/` (harvested replays, ~86 MB) is the only thing that genuinely must be
copied — **never re-harvest**, the fetcher got the cluster IP 403'd once.
Everything else regenerates:

```bash
$PY -m learn.dataset /local/data/vng205/field --out /local/data/vng205/bc20 --workers 60
```

```bash
$PY -m tools.pools --out runs/pools --workers 60
```

The venv is 5.5 GB — rsync it node-to-node rather than through a laptop.

## When something breaks

| symptom | cause |
|---|---|
| `RESOURCE_EXHAUSTED ... GiB` | two JAX processes on one GPU. Kill one. |
| `sh: fork: retry: Resource temporarily unavailable` | hit the process limit. 60 workers plus arena runs plus a sweep is too many; use `--workers 16` for side jobs. |
| `No module named X` | wrong tarball installed. Check `cat VERSION`. |
| `-bash: -m: command not found` | `$PY` is unset — new shell, re-export. |
| `No such file or directory: runs/...` | wrong node. `uname -n`, and logs only exist where the run started. |
| GPU busy, `nvidia-smi` shows no processes | another container, possibly your own JupyterHub session. Move nodes or stop that server. |
