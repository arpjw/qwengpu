#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
bash /workspace/qwengpu-compute-extent-remote.sh b2dfad86988c44964ad952e6c4f51d8c9c8c670e
mkdir -p /workspace/resume-src
tar -xzf /workspace/qwengpu-compute-resume-src.tgz -C /workspace/resume-src
cd /workspace/resume-src
compiler=/workspace/qwenputer/experimental/riscv
export PYTHONPATH="$compiler:$PWD"
py="$compiler/.venv/bin/python"
export QWENGPU_PYTHON="$py"
base=/workspace/Qwen3.5-9B
model=/workspace/Qwen3.5-9B-qwengpu-resident-compute
reports=/workspace/reports
host="$reports/compute_resume_host_a100.txt"
for name in compiled_tile8.py compiled_matrix.py eval_tile8.py; do
  cmp "/workspace/compute-src/qwengpu/$name" "qwengpu/$name"
done
{
  printf 'runtime_source_commit=%s\n' "$1"
  printf 'checkpoint_source_commit=%s\n' e6be9c58ca2af1cc57654f2aa662e0957fb71831
  printf 'constructed_weights_changed=false\n'
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  "$py" -c 'import torch,transformers,importlib.metadata as m; print("torch="+torch.__version__); print("transformers="+transformers.__version__); print("flash-linear-attention="+m.version("flash-linear-attention"))'
  sha256sum qwengpu/compiled_tile8.py qwengpu/compiled_matrix.py qwengpu/eval_tile8.py qwengpu/run_compute.py qwengpu/bench_compute.py
} > "$host"
mark() { printf '%s_utc=%s\n' "$1" "$(date -u +%FT%TZ)" | tee -a "$host"; }
mark benchmark_started
"$py" -m qwengpu.bench_compute --model "$model" --rounds 2 --out "$reports/compute_transport_9b_cuda_fla.json"
mark benchmark_finished
bash scripts/qualify_resident_add32.sh --resident-compute "$model" --device cuda --out "$reports/compute_inherited_9b_cuda_fla.json"
mark inherited_finished
"$py" -m qwengpu.run_matmul16 eval/frozen/matmul16_gate_v1/dense_seeded.json --grid-2x2 --model "$model" --device cuda --out "$reports/compute_matmul16_9b_cuda_fla.json"
mark matmul_finished
"$py" -m qwengpu.eval_globalgrid --model "$model" --device cuda --out "$reports/compute_globalgrid_9b_cuda_fla.json"
mark globalgrid_finished
"$py" -m qwengpu.textcheck_9b "$base" "$model" --device cuda --corpus eval/frozen/text_add32_gate_v1/prompts.json --max-new-tokens 16 --out "$reports/compute_text_9b_cuda_fla.json"
mark text_finished
"$py" -m qwengpu.eval_text_handoff "$model" --device cuda --reset-text --out "$reports/compute_handoff_reset_9b_cuda_fla.json"
mark handoff_reset_finished
"$py" -m qwengpu.eval_cache_relaunch "$model" --addmem32 --device cuda --out "$reports/compute_arithmetic_relaunch_9b_cuda_fla.json"
mark arithmetic_relaunch_finished
