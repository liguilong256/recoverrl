#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Create the environment with: conda env create -f environment.yml
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
  echo "Activate the project environment first: conda activate recoverrl" >&2
  exit 1
fi
python - <<'PY'
import os
from pathlib import Path
import sys

if sys.platform != "linux":
    raise SystemExit("GPU setup requires a Linux CUDA host.")
if sys.version_info < (3, 11):
    raise SystemExit("Python >= 3.11 required.")
if Path(sys.prefix).resolve() != Path(os.environ["CONDA_PREFIX"]).resolve():
    raise SystemExit("Python is outside the active Conda environment; reactivate recoverrl.")
PY
VERL_COMMIT=3e0d79ea8c38264bcaddc64de1b3465589580320
mkdir -p .deps
if [[ ! -d .deps/verl ]]; then
  git clone https://github.com/verl-project/verl.git .deps/verl
  git -C .deps/verl checkout "$VERL_COMMIT"
fi
if [[ "$(git -C .deps/verl rev-parse HEAD)" != "$VERL_COMMIT" ]]; then
  echo "Existing .deps/verl has a different revision; use a separate checkout." >&2
  exit 1
fi
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install -c configs/gpu-constraints.txt -e '.deps/verl[vllm]' -e '.[data,tokenizer]' tensorboard
python -m pip install -c configs/gpu-constraints.txt flash-attn --no-build-isolation
python -m pip check
python -m recoverrl.training.preflight
