#!/usr/bin/env bash
# Run inside the Linux CPU pod after extracting the indicated QwenGPU revision.
set -euo pipefail

qwengpu_source=${1:?Pass the extracted QwenGPU 632bfee source directory}
qwenputer_source=${2:?Pass a checkout of divagr18/qwenputer}
qwengpu_source=$(cd "$qwengpu_source" && pwd)
qwenputer_source=$(cd "$qwenputer_source" && pwd)

git -C "$qwenputer_source" checkout ca13310e2f9f788d1dd3feeea8213c1b0c23913c
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
python -m pip install 'numpy==2.5.4' 'safetensors==0.8.0' 'transformers==5.23.0'

cd "$qwengpu_source"
export PYTHONPATH="$qwenputer_source/experimental/riscv"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -v
