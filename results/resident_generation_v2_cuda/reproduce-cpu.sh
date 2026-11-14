#!/usr/bin/env bash
# Run in a Linux Python 3.12 CPU pod with at least 8 GB RAM.
set -euo pipefail

source_dir=${1:?Pass the QwenGPU source directory at d615810}
compiler_dir=${2:?Pass the qwenputer checkout directory}
source_dir=$(cd "$source_dir" && pwd)
compiler_dir=$(cd "$compiler_dir" && pwd)

git -C "$compiler_dir" checkout ca13310e2f9f788d1dd3feeea8213c1b0c23913c
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
python -m pip install --no-cache-dir 'numpy==2.5.4' 'safetensors==0.8.0' 'transformers==5.23.0'

cd "$source_dir"
export PYTHONPATH="$source_dir:$compiler_dir/experimental/riscv"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uname -sm
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'
cat /sys/fs/cgroup/memory.max
python -m unittest discover -s tests -v
