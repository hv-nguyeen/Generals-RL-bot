# Top-3 v2 r3 cluster upload and first start

The source archive deliberately contains no `.git`, virtual environment,
`runs/`, weights, replay data, or old packages. Extracting it over
`/local/data/vng205/generals-bot-top3-v2-r3-nodeN` creates a node-local verified
source tree. Checkpoints and logs remain outside the archive; preserve them and
never extract a new bundle over a live trainer.

## 1. Upload and verify

Upload `generals-bot-top3-v2-r3.tar.gz` to the shared home directory. The
verified SHA-256 is
`8fea299aa929ce02110112496bcd4fe3afd19c6f0896224fe40f51c97de10dec`.

```bash
uname -n
ls -lh ~/top3-v2-shared/generals-bot-top3-v2-r3.tar.gz
sha256sum ~/top3-v2-shared/generals-bot-top3-v2-r3.tar.gz
tar -tzf ~/top3-v2-shared/generals-bot-top3-v2-r3.tar.gz >/dev/null && echo ARCHIVE_OK
```

Stop any process currently importing code from the existing tree before the
upgrade. Do not extract over a live trainer.

```bash
mkdir -p /local/data/vng205/generals-bot-top3-v2-r3-nodeN
tar --strip-components=1 \
  -xzf ~/top3-v2-shared/generals-bot-top3-v2-r3.tar.gz \
  -C /local/data/vng205/generals-bot-top3-v2-r3-nodeN
cd /local/data/vng205/generals-bot-top3-v2-r3-nodeN
```

## 2. Activate the existing environment

Do not copy the 5.5 GB working CUDA environment through home and do not delete
it. Use the node's existing CUDA JAX environment or install it into node-local
storage.

```bash
export PY=/local/data/vng205/venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset JAX_PLATFORMS
```

```bash
$PY -c "import jax, jaxlib, numpy; print(jax.__version__, jaxlib.__version__, numpy.__version__); print(jax.devices())"
```

The final line must contain a CUDA device. Then verify the extracted source:

```bash
make PY=$PY verify
```

## 3. Load-check the shared incumbent pair

```bash
export P=$HOME/top3-v2-shared/weights/incumbent-refresh-onpolicy.npz
export V=$HOME/top3-v2-shared/weights/incumbent-refresh-onpolicy.critic.npz
test -f "$P" && test -f "$V"
sha256sum "$P" "$V"
```

If `incumbent.npz` already exists, compare hashes before replacing it. Stop on a
mismatch and preserve both files under different names.

```bash
$PY -m tools.grow --show "$P"
$PY - <<'PY'
import numpy as np
from bot import features
from bot.policy.net import Net
p = __import__("os").environ["P"]
z = np.load(p)
assert z["head_w"].shape[0] == features.PER_CELL
n = Net(p)
print("INCUMBENT_LOAD_OK", n.arch, z["head_w"].shape, z["conv0_w"].shape)
PY
```

After these commands pass, use a unique node-local run directory. Keep logs and
large artifacts under `/local/data`; copy only accepted checkpoints, matched
critics, hashes, and manifests to shared home. See the current continuation
section in `docs/TOP3-V2-HANDOFF.md`.
