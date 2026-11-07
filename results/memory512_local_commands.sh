#!/usr/bin/env bash
# Run from the repository with the pinned sibling qwenputer checkout.
set -euo pipefail
export PYTHONPATH="$PWD/../qwenputer/experimental/riscv:$PWD"
export QWENGPU_PYTHON="$PWD/../qwenputer/experimental/riscv/.venv/bin/python"
"$QWENGPU_PYTHON" -m qwengpu.eval_memory512 --device mps --out results/memory512_mps.json
bash scripts/qualify_resident_add32.sh --resident-memory512 --device mps --out results/memory512_inherited_mps.json
"$QWENGPU_PYTHON" -m qwengpu.run_matmul16 eval/frozen/matmul16_gate_v1/dense_seeded.json --grid-2x2 --resident-memory512 --device mps --out results/memory512_matmul16_mps.json
"$QWENGPU_PYTHON" -m qwengpu.eval_globalgrid --resident-memory512 --device mps --out results/memory512_globalgrid_mps.json
"$QWENGPU_PYTHON" -m qwengpu.run_resident eval/frozen/memory512_extent_v1/matrix_extent.json --resident-memory512 --device mps --check-reference --trace --out results/memory512_bounds_mps.json
