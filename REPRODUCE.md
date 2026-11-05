# Reproduce the current QwenGPU prototype

The reference machine requires Python 3.11 or later. The circuit and model
checks require the pinned Qwenputer compiler backend. Keep Qwen3.5 weights
outside this repository.

```sh
git clone https://github.com/divagr18/qwenputer.git
cd qwenputer
git checkout ca13310e2f9f788d1dd3feeea8213c1b0c23913c
cd experimental/riscv
uv sync --extra hf --no-dev
```

In a shell with paths set to your two checkouts:

```sh
export QWENGPU_DIR=/path/to/qwengpu
export QWENPUTER_DIR=/path/to/qwenputer/experimental/riscv
export PYTHONPATH="$QWENPUTER_DIR:$QWENGPU_DIR"
"$QWENPUTER_DIR/.venv/bin/python" -m unittest discover -s "$QWENGPU_DIR/tests" -v
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_branch \
  --device mps --out "$QWENGPU_DIR/results/branch_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_memory \
  --device mps --out "$QWENGPU_DIR/results/memory_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_alu \
  --device mps --out "$QWENGPU_DIR/results/alu_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_program \
  --device mps --out "$QWENGPU_DIR/results/program_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_rom_spike \
  --device mps --out "$QWENGPU_DIR/results/rom_spike_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_kernel \
  "$QWENGPU_DIR/examples/mixed_kernel.json" --device mps \
  --out "$QWENGPU_DIR/results/example_kernel_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_kernel \
  "$QWENGPU_DIR/examples/no_else_kernel.json" --device mps \
  --out "$QWENGPU_DIR/results/no_else_kernel_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_kernel \
  "$QWENGPU_DIR/examples/masked_overwrite_kernel.json" --device mps \
  --out "$QWENGPU_DIR/results/masked_overwrite_mps.json"
```

Use `--device cuda` on a CUDA host. For the optimized DeltaNet path, install
`flash-linear-attention==0.6.0`, `fla-core==0.6.0`, and `einops` in the same
environment before running the evaluator. Causal convolution used the
Transformers reference path in the recorded CUDA checks.

The 9B merge uses the public Qwen/Qwen3.5-9B snapshot at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The measured build used a
96 GB GPU, a 60 GB container disk, and enough host RAM for the widened
checkpoint. The source snapshot was about 19 GB and the merged checkpoint
about 25 GB; the container reached about 49 GB used. Both are temporary
local files, not repository contents.

```sh
"$QWENPUTER_DIR/.venv/bin/hf" download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir /path/to/Qwen3.5-9B
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --out /path/to/Qwen3.5-9B-qwengpu
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_branch_9b \
  /path/to/Qwen3.5-9B-qwengpu --device cuda \
  --out "$QWENGPU_DIR/results/branch_9b_cuda_fla.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_memory_9b \
  /path/to/Qwen3.5-9B-qwengpu --device cuda \
  --out "$QWENGPU_DIR/results/memory_9b_cuda_fla.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_alu \
  /path/to/Qwen3.5-9B-qwengpu --device cuda \
  --out "$QWENGPU_DIR/results/alu_9b_cuda_fla.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_program \
  /path/to/Qwen3.5-9B-qwengpu --device cuda \
  --out "$QWENGPU_DIR/results/program_9b_cuda_fla.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_program \
  /path/to/Qwen3.5-9B-qwengpu --device cuda --extra-cases 96 \
  --out "$QWENGPU_DIR/results/program_9b_stress_cuda_fla.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.textcheck_9b \
  /path/to/Qwen3.5-9B /path/to/Qwen3.5-9B-qwengpu --device cuda \
  --out "$QWENGPU_DIR/results/textcheck_9b_cuda.json"
```

The merged checkpoint accepts raw SIMT token IDs by adding the
`chip_token_offset` in its `qwengpu9b.json` metadata. The copied text
tokenizer handles ordinary text only. The evaluator provides instruction
tokens and read requests; it does not calculate the expected model output.
