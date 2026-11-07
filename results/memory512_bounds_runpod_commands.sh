#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /workspace/qwengpu
export PYTHONPATH=/workspace/qwenputer/experimental/riscv:/workspace/qwengpu
/workspace/qwenputer/experimental/riscv/.venv/bin/python -m qwengpu.run_resident eval/frozen/memory512_extent_v1/matrix_extent.json --model /workspace/Qwen3.5-9B-qwengpu-resident-memory512 --device cuda --check-reference --trace --out /workspace/reports/memory512_bounds_9b_cuda_fla.json
