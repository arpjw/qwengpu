#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p /workspace/bounds-src
tar -xzf /workspace/qwengpu-compute-bounds-src.tgz -C /workspace/bounds-src
cd /workspace/bounds-src
compiler=/workspace/qwenputer/experimental/riscv
export PYTHONPATH="$compiler:$PWD"
py="$compiler/.venv/bin/python"
model=/workspace/Qwen3.5-9B-qwengpu-resident-compute
host=/workspace/reports/compute_bounds_host_a100.txt
for name in compiled_tile8.py compiled_matrix.py; do
  cmp "/workspace/compute-src/qwengpu/$name" "qwengpu/$name"
done
{
  printf 'runtime_source_commit=%s\n' "$1"
  printf 'checkpoint_source_commit=%s\n' e6be9c58ca2af1cc57654f2aa662e0957fb71831
  printf 'constructed_weights_changed=false\nreference_only_correction=true\n'
  sha256sum qwengpu/compiled_tile8.py qwengpu/compiled_matrix.py qwengpu/eval_tile8.py eval/frozen/compute_bounds_v1/matrix_extent.json
} > "$host"
"$py" -m qwengpu.run_resident eval/frozen/compute_bounds_v1/matrix_extent.json --model "$model" --device cuda --check-reference --trace --out /workspace/reports/compute_bounds_9b_cuda_fla.json
date -u +bounds_finished_utc=%FT%TZ | tee -a "$host"
