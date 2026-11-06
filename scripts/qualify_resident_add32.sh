#!/usr/bin/env bash
# Run every inherited resident ADD32 fixture, including row windows.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

fixtures=()
for gate in tile8 resident indexed masked divergent loop grid wide resident_add32 window; do
  for manifest in "eval/frozen/${gate}_gate_v1/"*.json; do
    fixtures+=(--kernel "$manifest")
  done
done
exec "${QWENGPU_PYTHON:-python3}" -m qwengpu.eval_tile8 --resident-add32 "${fixtures[@]}" "$@"
