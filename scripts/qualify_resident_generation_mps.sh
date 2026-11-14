#!/usr/bin/env bash
# Replay the final circuit on Apple MPS without loading the merged weights.
set -euo pipefail

reports="${1:?expected report directory}"
cuda_reports="${2:-results/resident_generation_v3_cuda}"
python="${QWENGPU_PYTHON:-python}"
mkdir -p "$reports"
"$python" -c 'import torch; assert torch.backends.mps.is_available(), "Apple MPS is unavailable"'

inherited=()
for gate in tile8 resident indexed masked divergent loop grid wide resident_add32 window \
  resident_shared32 resident_addmem32 resident_branch32 resident_loop32; do
  inherited+=(eval/frozen/"${gate}_gate_v1/"*.json)
done
"$python" -m qwengpu.eval_resident_generation --device mps \
  --out "$reports/inherited-mps.json" "${inherited[@]}"
"$python" scripts/check_resident_generation.py results/cache_register_edge_cuda \
  "$reports/inherited-mps.json" --inherited-baseline results/model_stores_merged_cuda/inherited.json

"$python" -m qwengpu.eval_resident_generation --device mps \
  --out "$reports/interleaved-mps.json" \
  eval/frozen/resident_interleaved_v1/{interleaved_publish,barrier_groups,deadlock,cross_group_high_limb}.json
"$python" scripts/check_resident_generation.py results/cache_register_edge_cuda \
  "$reports/interleaved-mps.json"

"$python" -m qwengpu.eval_resident_generation --device mps \
  --out "$reports/sequential-mps.json" \
  eval/frozen/{resident_rom32_v1/workgroup_reset,resident_alu32_v1/address_walk,scoped_memory_v1/isolation,memory512_extent_v1/matrix_extent}.json
"$python" scripts/check_resident_generation.py results/cache_register_edge_cuda \
  "$reports/sequential-mps.json" --sequential-baseline results/model_stores_merged_cuda \
  --matrix-extent-baseline results/memory512_bounds_9b_cuda_fla.json

"$python" -m qwengpu.eval_resident_generation --device mps \
  --out "$reports/memory512-raw.json" eval/frozen/memory512_v1/*.json
"$python" -m qwengpu.eval_resident_generation --device mps \
  --out "$reports/integer-programs.json" eval/frozen/full_integer_v1/indexed_words.json \
  eval/frozen/resident_alu32_v1/integer_edges.json eval/frozen/resident_rom32_v1/loops_and_barrier.json
"$python" scripts/eval_resident_generation_workloads.py --device mps \
  --suite eval/frozen/memory512_compute_v1/cases.json --out "$reports/memory512-compute.json"
"$python" scripts/eval_resident_generation_workloads.py --device mps \
  --case 3x5x2 --case 9x7x5 --out "$reports/tiled.json" \
  --matrix-fixture eval/frozen/tiled_compute_v1/3x5x2.json \
  --matrix-fixture eval/frozen/tiled_compute_v1/9x7x5.json
"$python" scripts/eval_scoped_reset.py --device mps --resident-generate \
  --out "$reports/scoped_reset.json"
"$python" scripts/eval_interleaved_handoff.py --device mps --resident-generate \
  --out "$reports/interleaved_handoff.json"
"$python" -m qwengpu.assemble --full-integer examples/scoped_shared_u32.qga \
  --data examples/scoped_shared_u32_data.json --out "$reports/scoped-shared-u32-kernel.json"
"$python" -m qwengpu.run_resident "$reports/scoped-shared-u32-kernel.json" \
  --resident-generate --device mps --check-reference --trace --out "$reports/scoped-shared-u32.json"
"$python" -m qwengpu.run_matmul16 eval/frozen/matmul16_gate_v1/dense_seeded.json \
  --resident-generate --device mps --out "$reports/matmul16.json"
"$python" scripts/check_resident_generation_v3.py --mps "$reports" --compare-cuda "$cuda_reports"
