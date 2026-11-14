#!/usr/bin/env bash
# Build or verify the pinned checkpoint and replay frozen resident kernels.
set -euo pipefail

existing=0
variant=resident-generate
while [[ "${1:-}" == --* ]]; do
  case "$1" in
    --existing) existing=1 ;;
    --v4) variant=generated-tilegrid ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done
base="${1:?expected Qwen3.5-9B source path}"
model="${2:?expected merged checkpoint path}"
reports="${3:?expected report directory}"
source_commit="${4:-b5f4898cb2e54ac559cd0fa81d93c2901ccb9222}"
if [[ "$variant" == generated-tilegrid ]]; then
  source_commit="$(git rev-parse --verify "${4:?provide the v4 build source}^{commit}")"
fi
python="${QWENGPU_PYTHON:-python}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
case "$base" in /*) ;; *) base="$PWD/$base";; esac
case "$model" in /*) ;; *) model="$PWD/$model";; esac
# Resolve an explicitly selected interpreter before entering the source archive.
case "$python" in /*) ;; */*) python="$PWD/$python";; esac
mkdir -p "$reports"

if (( existing )) && [[ "$variant" == resident-generate ]]; then
  "$python" scripts/check_checkpoint_package.py "$model" \
    --pinned-manifest results/resident_generation_v3_cuda/manifest.json \
    --expected-checksum-digest f85fcb86f80668c57bfbc241aa8d9ca08378d50623eea405add56d499dcbd415 \
    > "$reports/checkpoint-checksum.txt"
  cp "$model/build-manifest.json" "$reports/manifest.json"
elif (( ! existing )); then
  source_commit="$(git rev-parse --verify "${source_commit}^{commit}")"
  build_source="$(mktemp -d)"
  trap 'rm -rf -- "$build_source"' EXIT
  git archive "$source_commit" qwengpu | tar -xf - -C "$build_source"
  PYTHONPATH="$build_source${PYTHONPATH:+:$PYTHONPATH}" \
    "$python" -P -m qwengpu.merge9b "$base" --"$variant" --out "$model"
fi

if [[ "$variant" == generated-tilegrid ]]; then
  "$python" - "$model/qwengpu9b.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    assert json.load(stream)["layout"] == "qwengpu-resident-generation-9b-v4"
PY
fi

fixtures=(
  eval/frozen/resident_interleaved_v1/interleaved_publish.json
  eval/frozen/resident_interleaved_v1/barrier_groups.json
  eval/frozen/resident_interleaved_v1/deadlock.json
  eval/frozen/resident_interleaved_v1/cross_group_high_limb.json
)
sequential=(
  eval/frozen/resident_rom32_v1/workgroup_reset.json
  eval/frozen/resident_alu32_v1/address_walk.json
  eval/frozen/scoped_memory_v1/isolation.json
  eval/frozen/memory512_extent_v1/matrix_extent.json
)
inherited=()
for gate in tile8 resident indexed masked divergent loop grid wide resident_add32 window \
  resident_shared32 resident_addmem32 resident_branch32 resident_loop32; do
  inherited+=(eval/frozen/"${gate}_gate_v1/"*.json)
done
"$python" -m qwengpu.eval_resident_generation --out "$reports/standalone.json" "${fixtures[@]}"
"$python" -m qwengpu.eval_resident_generation --out "$reports/sequential-standalone.json" "${sequential[@]}"
"$python" -m qwengpu.textcheck_9b "$base" "$model" --device cuda \
  --corpus eval/frozen/text_add32_gate_v1/prompts.json --max-new-tokens 16 --out "$reports/text.json"
"$python" - "$reports/text.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    report = json.load(stream)
assert report["next_top1_same"] == 24 and report["greedy_identical"] >= 21
assert report["merged_chip_tokens"] == 0
PY
"$python" -m qwengpu.eval_resident_generation --model "$model" --out "$reports/interleaved-merged.json" "${fixtures[@]}"
"$python" -m qwengpu.eval_resident_generation --model "$model" --out "$reports/sequential-merged.json" "${sequential[@]}"
"$python" -m qwengpu.eval_resident_generation --model "$model" --out "$reports/inherited-merged.json" "${inherited[@]}"
"$python" scripts/check_resident_generation.py results/cache_register_edge_cuda \
  "$reports/inherited-merged.json" --inherited-baseline results/model_stores_merged_cuda/inherited.json
"$python" scripts/check_resident_generation.py results/cache_register_edge_cuda \
  "$reports/standalone.json" "$reports/interleaved-merged.json"
"$python" scripts/check_resident_generation.py results/cache_register_edge_cuda \
  "$reports/sequential-standalone.json" "$reports/sequential-merged.json" \
  --sequential-baseline results/model_stores_merged_cuda \
  --matrix-extent-baseline results/memory512_bounds_9b_cuda_fla.json
"$python" scripts/eval_resident_generation_workloads.py "$model" \
  --suite eval/frozen/memory512_compute_v1/cases.json \
  --case vector_64 --case reduce_128 --case matmul_8x4x4 --out "$reports/bounds.json"
"$python" scripts/eval_resident_generation_handoff.py "$model" --out "$reports/handoff.json"
"$python" scripts/eval_scoped_reset.py --model "$model" --device cuda \
  --resident-generate --out "$reports/scoped_reset.json"
"$python" scripts/eval_interleaved_handoff.py --model "$model" --device cuda \
  --resident-generate --out "$reports/interleaved_handoff.json"
"$python" -m qwengpu.run_compute examples/memory512_reduce_128.json \
  --model "$model" --device cuda --out "$reports/cli-reduction.json"
status=0
"$python" -m qwengpu.run_resident eval/frozen/tile8_gate_v1/deadlock.json \
  --model "$model" --device cuda --check-reference --trace \
  --out "$reports/cli-deadlock.json" || status=$?
test "$status" -eq 3
"$python" -m qwengpu.assemble --full-integer examples/scoped_shared_u32.qga \
  --data examples/scoped_shared_u32_data.json --out "$reports/scoped-shared-u32-kernel.json"
"$python" -m qwengpu.run_resident "$reports/scoped-shared-u32-kernel.json" \
  --model "$model" --device cuda --check-reference --trace --out "$reports/scoped-shared-u32.json"
"$python" scripts/eval_resident_generation_workloads.py "$model" \
  --case 3x5x2 --case 9x7x5 --out "$reports/tiled.json" \
  eval/frozen/tiled_compute_v1/{3x5x2,9x7x5}.json
if [[ "$variant" == generated-tilegrid ]]; then
  matrix_report=matmul16-single-launch.json
else
  matrix_report=matmul16.json
fi
"$python" -m qwengpu.run_matmul16 eval/frozen/matmul16_gate_v1/dense_seeded.json \
  --model "$model" --device cuda --out "$reports/$matrix_report"
"$python" -m qwengpu.eval_resident_generation --model "$model" \
  --out "$reports/memory512-raw.json" eval/frozen/memory512_v1/*.json
"$python" -m qwengpu.eval_resident_generation --model "$model" \
  --out "$reports/integer-programs.json" eval/frozen/full_integer_v1/indexed_words.json \
  eval/frozen/resident_alu32_v1/integer_edges.json eval/frozen/resident_rom32_v1/loops_and_barrier.json
"$python" scripts/eval_resident_generation_workloads.py "$model" \
  --suite eval/frozen/memory512_compute_v1/cases.json --out "$reports/memory512-compute.json"
"$python" -m qwengpu.eval_resident_generation --device cuda \
  --out "$reports/inherited-standalone.json" "${inherited[@]}"
if (( ! existing )) || [[ "$variant" == generated-tilegrid ]]; then
  "$python" scripts/capture_checkpoint_manifest.py "$model" --source "$source_commit" \
    --compiler ca13310e2f9f788d1dd3feeea8213c1b0c23913c \
    --base c202236235762e1c871ad0ccb60c8ee5ba337b9a --variant "$variant" \
    --out "$reports/manifest.json"
fi
if [[ "$variant" == generated-tilegrid ]]; then
  "$python" scripts/check_resident_generation_v3.py --v4 "$reports"
else
  "$python" scripts/check_resident_generation_v3.py "$reports"
fi
