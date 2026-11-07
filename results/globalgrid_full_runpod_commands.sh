#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p /workspace/qwengpu /workspace/reports
tar -xzf /workspace/qwengpu-globalgrid-full-src.tgz -C /workspace/qwengpu
cd /workspace/qwengpu
command -v uv >/dev/null || python3 -m pip install uv
bash scripts/bootstrap_runpod.sh
compiler=/workspace/qwenputer/experimental/riscv
export PYTHONPATH="$compiler:$PWD"
export QWENGPU_PYTHON="$compiler/.venv/bin/python"
py="$QWENGPU_PYTHON"
base=/workspace/Qwen3.5-9B
model=/workspace/Qwen3.5-9B-qwengpu-resident-globalgrid
reports=/workspace/reports
host="$reports/globalgrid_full_host_a100.txt"
{
 nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
 printf 'source_commit=%s\n' "$1"
 printf 'compiler_commit=%s\n' "$(git -C /workspace/qwenputer rev-parse HEAD)"
 printf 'base_revision=%s\n' c202236235762e1c871ad0ccb60c8ee5ba337b9a
 printf 'OMP_NUM_THREADS=%s\nMKL_NUM_THREADS=%s\nPYTORCH_ALLOC_CONF=%s\n' "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$PYTORCH_ALLOC_CONF"
 "$py" -c 'import importlib.metadata as m; print("flash-linear-attention=" + m.version("flash-linear-attention")); print("transformers=" + m.version("transformers"))'
 sha256sum qwengpu/compiled_tile8.py qwengpu/eval_tile8.py qwengpu/eval_tilegrid.py qwengpu/run_matmul16.py
} > "$host"
mark() { printf '%s_utc=%s\n' "$1" "$(date -u +%FT%TZ)" | tee -a "$host"; }
mark merge_started
"$py" -m qwengpu.merge9b "$base" --resident-globalgrid --out "$model"
mark merge_finished
bash scripts/qualify_resident_add32.sh --resident-globalgrid "$model" --device cuda --out "$reports/globalgrid_inherited_9b_cuda_fla.json"
mark inherited_finished
"$py" -m qwengpu.run_matmul16 eval/frozen/matmul16_gate_v1/dense_seeded.json --grid-2x2 --model "$model" --device cuda --out "$reports/globalgrid_matmul16_9b_cuda_fla.json"
mark matmul_finished
"$py" -m qwengpu.textcheck_9b "$base" "$model" --device cuda --corpus eval/frozen/text_add32_gate_v1/prompts.json --max-new-tokens 16 --out "$reports/globalgrid_text_9b_cuda_fla.json"
mark text_finished
"$py" -m qwengpu.eval_text_handoff "$model" --device cuda --reset-text --out "$reports/globalgrid_handoff_reset_9b_cuda_fla.json"
mark handoff_reset_finished
"$py" -m qwengpu.eval_cache_relaunch "$model" --addmem32 --device cuda --out "$reports/globalgrid_arithmetic_relaunch_9b_cuda_fla.json"
mark arithmetic_relaunch_finished
