#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=6
export PYTHONPATH="/home/admin123/一些设置/终端/train-terminal${PYTHONPATH:+:$PYTHONPATH}"
exec systemd-run --user --scope --quiet --unit="asym-eval-${$}" \
  -p MemoryHigh="${ASYMVPR_EVAL_MEMORY_HIGH:-7G}" -p MemoryMax="${ASYMVPR_EVAL_MEMORY_MAX:-9G}" -p MemorySwapMax=0 \
  "${ASYMVPR_PYTHON:-/home/admin123/miniconda3/envs/boq/bin/python}" -u -m "${ASYMVPR_EVAL_MODULE:-reproduction.evaluate}" "$@"
