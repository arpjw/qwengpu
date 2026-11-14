#!/usr/bin/env bash
# Compare the optional register cache edge against the stock model transport.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
model="${1:?expected the model-owned store checkpoint}"
baseline="${2:?expected the stock model qualification reports}"
reports="${3:?expected output directory}"
py="${QWENGPU_PYTHON:-python3}"
mkdir -p "$reports"

for fixture in interleaved_publish barrier_groups deadlock; do
  rc=0
  "$py" -m qwengpu.run_resident "eval/frozen/resident_interleaved_v1/$fixture.json" \
    --model "$model" --transport fused-cache-edge --device cuda --check-reference \
    --trace --out "$reports/$fixture.json" || rc=$?
  case "$fixture" in
    interleaved_publish) stock=interleaved_publish ;;
    barrier_groups) stock=interleaved_barrier_groups ;;
    deadlock) stock=interleaved_deadlock ;;
  esac
  if [[ "$fixture" == deadlock ]]; then [[ "$rc" == 3 ]]; else [[ "$rc" == 0 ]]; fi
  "$py" scripts/check_cache_register_edge.py "$baseline/$stock.json" "$reports/$fixture.json" --require-planless
done
