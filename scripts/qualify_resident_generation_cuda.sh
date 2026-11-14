#!/usr/bin/env bash
# Build the resident generation checkpoint and replay frozen interleaved kernels.
set -euo pipefail

base="${1:?expected Qwen3.5-9B source path}"
model="${2:?expected merged checkpoint destination}"
reports="${3:?expected report directory}"
source_commit="${4:?expected source commit}"
python="${QWENGPU_PYTHON:-python}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$reports"

fixtures=(
  eval/frozen/resident_interleaved_v1/interleaved_publish.json
  eval/frozen/resident_interleaved_v1/barrier_groups.json
  eval/frozen/resident_interleaved_v1/deadlock.json
  eval/frozen/resident_interleaved_v1/cross_group_high_limb.json
)
"$python" -m qwengpu.eval_resident_generation --out "$reports/standalone.json" "${fixtures[@]}"
"$python" -m qwengpu.merge9b "$base" --resident-generate --out "$model"
"$python" -m qwengpu.eval_resident_generation --model "$model" --out "$reports/merged.json" "${fixtures[@]}"
"$python" scripts/check_resident_generation.py results/cache_register_edge_cuda \
  "$reports/standalone.json" "$reports/merged.json"
"$python" -m qwengpu.textcheck_9b "$base" "$model" --device cuda \
  --corpus eval/frozen/text_add32_gate_v1/prompts.json --max-new-tokens 16 --out "$reports/text.json"
"$python" scripts/capture_checkpoint_manifest.py "$model" --source "$source_commit" \
  --compiler ca13310e2f9f788d1dd3feeea8213c1b0c23913c \
  --base c202236235762e1c871ad0ccb60c8ee5ba337b9a --variant resident-generate \
  --out "$reports/manifest.json"
