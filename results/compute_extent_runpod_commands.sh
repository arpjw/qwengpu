#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p /workspace/extent-src
tar -xzf /workspace/qwengpu-compute-extent-src.tgz -C /workspace/extent-src
cd /workspace/extent-src
compiler=/workspace/qwenputer/experimental/riscv
export PYTHONPATH="$compiler:$PWD"
py="$compiler/.venv/bin/python"
model=/workspace/Qwen3.5-9B-qwengpu-resident-compute
reports=/workspace/reports
host="$reports/compute_extent_host_a100.txt"
for name in compiled_tile8.py compiled_matrix.py eval_tile8.py; do
  cmp "/workspace/compute-src/qwengpu/$name" "qwengpu/$name"
done
{
  printf 'runtime_source_commit=%s\n' "$1"
  printf 'checkpoint_source_commit=%s\n' e6be9c58ca2af1cc57654f2aa662e0957fb71831
  printf 'constructed_weights_changed=false\n'
  sha256sum qwengpu/compiled_tile8.py qwengpu/compiled_matrix.py qwengpu/eval_tile8.py qwengpu/run_compute.py eval/frozen/compute_extent_v1/cases.json
} > "$host"
mark() { printf '%s_utc=%s\n' "$1" "$(date -u +%FT%TZ)" | tee -a "$host"; }
mark extent_started
"$py" -m qwengpu.eval_compute --cases eval/frozen/compute_extent_v1/cases.json --model "$model" --device cuda --out "$reports/compute_extent_9b_cuda_fla.json"
mark extent_finished
"$py" - <<'PY'
from pathlib import Path
import json
request=json.loads(Path('eval/frozen/compute_extent_v1/cases.json').read_text())['reduce_64']
Path('/workspace/reports/compute_reduce64_request.json').write_text(json.dumps(request,indent=2)+'\n')
PY
"$py" -m qwengpu.run_compute "$reports/compute_reduce64_request.json" --model "$model" --device cuda --transport fixed --trace --out "$reports/compute_public_reduce64_9b_cuda_fla.json"
mark public64_finished
cp "$model/qwengpu9b.json" "$reports/compute_checkpoint_metadata.json"
