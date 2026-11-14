#!/usr/bin/env bash
# Run with the pinned compiler and model dependencies from bootstrap_runpod.sh.
set -euo pipefail

base="${1:?expected Qwen3.5-9B source path}"
model="${2:?expected merged checkpoint destination}"
reports="${3:?expected report directory}"
python="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
mkdir -p "$reports"

"$python" -m unittest discover -s tests -p test_control.py -v > "$reports/compiler.txt" 2>&1
"$python" -m qwengpu.eval_control --generate --device cuda --out "$reports/standalone.json"
"$python" -m qwengpu.merge9b "$base" --control-generate --out "$model"
"$python" -m qwengpu.eval_control "$model" --generate --device cuda --out "$reports/merged.json"
"$python" -m qwengpu.textcheck_9b "$base" "$model" --device cuda \
  --corpus eval/frozen/text_add32_gate_v1/prompts.json --max-new-tokens 16 \
  --out "$reports/text.json"
"$python" scripts/check_control_generation.py results/control_cuda_fla.json \
  "$reports/standalone.json" "$reports/merged.json"
