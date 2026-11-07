#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /workspace/qwengpu
py=/workspace/qwenputer/experimental/riscv/.venv/bin/python
export PYTHONPATH=/workspace/qwenputer/experimental/riscv:/workspace/qwengpu
export QWENGPU_PYTHON="$py"
base=/workspace/Qwen3.5-9B
model=/workspace/Qwen3.5-9B-qwengpu-resident-memory512
reports=/workspace/reports
mkdir -p "$reports"
host="$reports/memory512_host_a100.txt"
{
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  printf 'source_commit=%s\n' "$1"
  printf 'compiler_commit=%s\n' "$(git -C /workspace/qwenputer rev-parse HEAD)"
  printf 'base_revision=%s\n' c202236235762e1c871ad0ccb60c8ee5ba337b9a
  printf 'PYTORCH_ALLOC_CONF=%s\n' "$PYTORCH_ALLOC_CONF"
  sha256sum qwengpu/compiled_tile8.py qwengpu/memory_geometry.py qwengpu/eval_tile8.py qwengpu/run_compute.py eval/frozen/memory512_compute_v1/cases.json
} > "$host"
mark() { printf '%s_utc=%s\n' "$1" "$(date -u +%FT%TZ)" | tee -a "$host"; }
mark merge_started
"$py" -m qwengpu.merge9b "$base" --resident-memory512 --out "$model"
mark merge_finished
"$py" -m qwengpu.eval_memory512 --model "$model" --device cuda --out "$reports/memory512_9b_cuda_fla.json"
mark memory512_finished
bash scripts/qualify_resident_add32.sh --resident-memory512 "$model" --device cuda --out "$reports/memory512_inherited_9b_cuda_fla.json"
mark inherited_finished
"$py" -m qwengpu.run_matmul16 eval/frozen/matmul16_gate_v1/dense_seeded.json --grid-2x2 --model "$model" --device cuda --out "$reports/memory512_matmul16_9b_cuda_fla.json"
mark matmul_finished
"$py" -m qwengpu.eval_globalgrid --model "$model" --device cuda --out "$reports/memory512_globalgrid_9b_cuda_fla.json"
mark globalgrid_finished
"$py" -m qwengpu.run_compute examples/memory512_vector_64.json --model "$model" --device cuda --transport fixed --trace --out "$reports/memory512_public_vector_9b_cuda_fla.json"
"$py" -m qwengpu.run_compute examples/memory512_reduce_128.json --model "$model" --device cuda --transport fixed --trace --out "$reports/memory512_public_reduce_9b_cuda_fla.json"
mark public_finished
"$py" -m qwengpu.textcheck_9b "$base" "$model" --device cuda --corpus eval/frozen/text_add32_gate_v1/prompts.json --max-new-tokens 16 --out "$reports/memory512_text_9b_cuda_fla.json"
mark text_finished
"$py" -m qwengpu.eval_text_handoff "$model" --device cuda --reset-text --out "$reports/memory512_handoff_reset_9b_cuda_fla.json"
mark handoff_reset_finished
"$py" -m qwengpu.eval_cache_relaunch "$model" --addmem32 --device cuda --out "$reports/memory512_arithmetic_relaunch_9b_cuda_fla.json"
mark arithmetic_relaunch_finished
