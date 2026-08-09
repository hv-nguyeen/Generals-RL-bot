# Top-3 v2 cluster upload and first start

The source archive deliberately contains no `.git`, virtual environment,
`runs/`, weights, replay data, or old packages. Extracting it over
`/local/data/vng205/generals-bot` updates the code while preserving the cluster's
checkpoints and logs.

## 1. Upload and verify

Upload the tarball to the compute node's visible home directory. Confirm the
expected byte count and SHA-256 from the handoff message before extracting:

```bash
uname -n
ls -lh ~/generals-bot-top3-v2-r2-20260808.tar.gz
sha256sum ~/generals-bot-top3-v2-r2-20260808.tar.gz
tar -tzf ~/generals-bot-top3-v2-r2-20260808.tar.gz >/dev/null && echo ARCHIVE_OK
```

Stop any process currently importing code from the existing tree before the
upgrade. Do not extract over a live trainer.

```bash
tar -xzf ~/generals-bot-top3-v2-r2-20260808.tar.gz -C /local/data/vng205
cd /local/data/vng205/generals-bot
```

## 2. Activate the existing environment

Do not reinstall the 5.5 GB working CUDA environment and do not delete it.

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
make test PY=$PY
$PY -m tools.verify_engine --games 40 --max-turns 200 --encoders
$PY -m tools.grow --selfcheck
$PY -m tools.valueselfplay --selfcheck
```

## 3. Preserve and load-check the incumbent

```bash
mkdir -p runs/top3-v2
if [ -e runs/top3-v2/incumbent.npz ]; then
  sha256sum runs/nn/sp16-c24.npz runs/top3-v2/incumbent.npz
else
  cp runs/nn/sp16-c24.npz runs/top3-v2/incumbent.npz
fi
sha256sum runs/top3-v2/incumbent.npz | tee runs/top3-v2/incumbent.sha256
```

If `incumbent.npz` already exists, compare hashes before replacing it. Stop on a
mismatch and preserve both files under different names.

```bash
$PY -m tools.grow --show runs/top3-v2/incumbent.npz
$PY - <<'PY'
import numpy as np
from bot import features
from bot.policy.net import Net
p = "runs/top3-v2/incumbent.npz"
z = np.load(p)
assert z["head_w"].shape[0] == features.PER_CELL
n = Net(p)
print("INCUMBENT_LOAD_OK", n.arch, z["head_w"].shape, z["conv0_w"].shape)
PY
```

After these commands pass, continue at section 3 of
`docs/TOP3-V2-HANDOFF.md`. Do not jump directly to the league: the critic gate
and full-distance transfer probe are mandatory launch conditions.
