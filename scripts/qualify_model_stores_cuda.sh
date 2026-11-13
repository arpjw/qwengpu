#!/usr/bin/env bash
# Build and qualify the model-owned store layout from a pinned base.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
base="${1:?expected pinned Qwen3.5-9B directory}"
model="${2:?expected a new model directory}"
reports="${3:?expected report directory}"
source_commit="${4:?expected source commit}"
py="${QWENGPU_PYTHON:-python3}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$reports"
{
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  printf 'source_commit=%s\n' "$source_commit"
  "$py" -c 'import torch,transformers; print(torch.__version__, transformers.__version__)'
} > "$reports/host.txt"

"$py" -m qwengpu.preflight_memory512 --internal-stores --out "$reports/preflight.json"
"$py" -m qwengpu.merge9b "$base" --internal-stores --out "$model"
"$py" -m qwengpu.textcheck_9b "$base" "$model" --device cuda \
  --corpus eval/frozen/text_add32_gate_v1/prompts.json --max-new-tokens 16 --out "$reports/text.json"
"$py" - "$reports/text.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    r = json.load(stream)
assert r["next_top1_same"] == 24 and r["greedy_identical"] >= 21 and r["merged_chip_tokens"] == 0
PY
"$py" scripts/capture_checkpoint_manifest.py "$model" --source "$source_commit" \
  --compiler ca13310e2f9f788d1dd3feeea8213c1b0c23913c \
  --base c202236235762e1c871ad0ccb60c8ee5ba337b9a --variant internal-stores \
  --out "$reports/manifest.json"
"$py" scripts/eval_model_stores.py --model "$model" --device cuda --out "$reports/model_stores.json"
bash scripts/qualify_integer.sh cuda "$reports" "$model"
"$py" -m qwengpu.eval_memory512_handoff "$model" --device cuda --out "$reports/handoff.json"
date -u > "$reports/complete.txt"
