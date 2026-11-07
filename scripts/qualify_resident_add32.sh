#!/usr/bin/env bash
# Run inherited ADD32 fixtures and any requested extensions.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

variant=--resident-add32
gates=(tile8 resident indexed masked divergent loop grid wide resident_add32 window)
case "${1:-}" in
  --resident-shared32|--resident-addmem32|--resident-branch32|--resident-loop32|--resident-tilebanks|--resident-tilepipe|--resident-tilegrid)
    variant="$1"
    shift
    gates+=(resident_shared32)
    if [[ "$variant" != "--resident-shared32" ]]; then
      gates+=(resident_addmem32)
    fi
    if [[ "$variant" != "--resident-shared32" && "$variant" != "--resident-addmem32" ]]; then
      gates+=(resident_branch32)
    fi
    if [[ "$variant" == "--resident-loop32" || "$variant" == "--resident-tilebanks" || "$variant" == "--resident-tilepipe" || "$variant" == "--resident-tilegrid" ]]; then
      gates+=(resident_loop32)
    fi
    ;;
esac
fixtures=()
for gate in "${gates[@]}"; do
  for manifest in "eval/frozen/${gate}_gate_v1/"*.json; do
    fixtures+=(--kernel "$manifest")
  done
done
exec "${QWENGPU_PYTHON:-python3}" -m qwengpu.eval_tile8 "$variant" "${fixtures[@]}" "$@"
