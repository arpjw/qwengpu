#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

device="${1:?expected mps or cuda}"
out="${2:?expected output directory}"
model="${3:-}"
variant="${4:---resident-rom32}"
py="${QWENGPU_PYTHON:-python3}"
mkdir -p "$out"
if [[ -n "$model" ]]; then
  raw=(--model "$model")
  inherited=("$model")
else
  raw=("$variant")
  inherited=()
fi

if [[ "$variant" == "--resident-interleaved" ]]; then
  "$py" -m qwengpu.run_resident eval/frozen/resident_interleaved_v1/interleaved_publish.json     "${raw[@]}" --device "$device" --check-reference --trace --out "$out/interleaved_publish.json"
  rc=0
  "$py" -m qwengpu.run_resident eval/frozen/resident_interleaved_v1/deadlock.json     "${raw[@]}" --device "$device" --check-reference --trace --out "$out/interleaved_deadlock.json" || rc=$?
  [[ "$rc" == 3 ]]
fi
if [[ "$variant" == "--resident-scoped" || "$variant" == "--resident-interleaved" ]]; then
  "$py" -m qwengpu.run_resident eval/frozen/scoped_memory_v1/isolation.json     "${raw[@]}" --device "$device" --check-reference --trace --out "$out/isolation.json"
fi
if [[ "$variant" == "--resident-alu32" || "$variant" == "--resident-scoped" || "$variant" == "--resident-interleaved" ]]; then
  for fixture in integer_edges address_walk; do
    "$py" -m qwengpu.run_resident "eval/frozen/resident_alu32_v1/$fixture.json"       "${raw[@]}" --device "$device" --check-reference --trace --out "$out/$fixture.json"
  done
fi

for fixture in loops_and_barrier workgroup_reset; do
  "$py" -m qwengpu.run_resident "eval/frozen/resident_rom32_v1/$fixture.json" \
    "${raw[@]}" --device "$device" --check-reference --trace --out "$out/$fixture.json"
done
"$py" -m qwengpu.eval_memory512 "${raw[@]}" --device "$device" \
  --out "$out/memory512.json"
bash scripts/qualify_resident_add32.sh "$variant" "${inherited[@]}" \
  --device "$device" --out "$out/inherited.json"
"$py" -m qwengpu.run_matmul16 eval/frozen/matmul16_gate_v1/dense_seeded.json \
  "${raw[@]}" --grid-2x2 --device "$device" --out "$out/matmul16.json"
