#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export JUPYTER_CONFIG_DIR="$PROJECT_DIR/.venv/jupyter-config"
export JUPYTER_DATA_DIR="$PROJECT_DIR/.venv/jupyter-data"
export JUPYTER_RUNTIME_DIR="$PROJECT_DIR/.venv/jupyter-runtime"
export IPYTHONDIR="$PROJECT_DIR/.venv/ipython"
exec "$PROJECT_DIR/.venv/bin/python" -m jupyterlab --ip=127.0.0.1 --no-browser \
  notebooks/01_phase1_foundations.ipynb
