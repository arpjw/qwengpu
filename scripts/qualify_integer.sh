#!/usr/bin/env bash
# Verify the integrated integer layout on one model per group of checks.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
device="${1:?expected mps or cuda}"
reports="${2:?expected output directory}"
model="${3:-}"
py="${QWENGPU_PYTHON:-python3}"
mkdir -p "$reports"
raw=(--full-integer)
model_arg=()
if [[ -n "$model" ]]; then
  raw=(--model "$model")
  model_arg=(--model "$model")
fi

bash scripts/qualify_rom32.sh "$device" "$reports" "$model" --full-integer
"$py" scripts/eval_scoped_reset.py "${model_arg[@]}" --full-integer \
  --device "$device" --out "$reports/scoped_reset.json"
"$py" scripts/eval_interleaved_handoff.py "${model_arg[@]}" --full-integer \
  --device "$device" --out "$reports/interleaved_handoff.json"
for fixture in 3x5x2 9x7x5; do
  "$py" -m qwengpu.run_compute "eval/frozen/tiled_compute_v1/$fixture.json" \
    "${raw[@]}" --device "$device" --trace --out "$reports/tiled_${fixture}.json"
done
for fixture in interleaved_publish barrier_groups deadlock; do
  rc=0
  "$py" -m qwengpu.run_resident "eval/frozen/resident_interleaved_v1/$fixture.json" \
    "${raw[@]}" --device "$device" --check-reference --trace --transport fused-dispatch \
    --out "$reports/fused_${fixture}.json" || rc=$?
  if [[ "$fixture" == deadlock ]]; then [[ "$rc" == 3 ]]; else [[ "$rc" == 0 ]]; fi
done
