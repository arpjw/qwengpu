#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /workspace/qwengpu
export PYTHONPATH=/workspace/qwenputer/experimental/riscv:/workspace/qwengpu
sha256sum qwengpu/eval_memory512_handoff.py > /workspace/reports/memory512_ram_handoff_source.txt
/workspace/qwenputer/experimental/riscv/.venv/bin/python -m qwengpu.eval_memory512_handoff /workspace/Qwen3.5-9B-qwengpu-resident-memory512 --device cuda --out /workspace/reports/memory512_ram_handoff_9b_cuda_fla.json
