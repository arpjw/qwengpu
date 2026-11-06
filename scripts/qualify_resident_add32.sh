#!/usr/bin/env bash
# Run inherited ADD32 fixtures, with the shared32 gate on request.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

variant=--resident-add32
gates=(tile8 resident indexed masked divergent loop grid wide resident_add32 window)
if [[ "${1:-}" == "--resident-shared32" ]]; then
  variant="$1"
  shift
  gates+=(resident_shared32)
fi
fixtures=()
for gate in "${gates[@]}"; do
  for manifest in "eval/frozen/${gate}_gate_v1/"*.json; do
    fixtures+=(--kernel "$manifest")
  done
done
exec "${QWENGPU_PYTHON:-python3}" -m qwengpu.eval_tile8 "$variant" "${fixtures[@]}" "$@"
