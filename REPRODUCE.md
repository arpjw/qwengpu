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
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_program \
  --device mps --extra-cases 96 \
  --out "$QWENGPU_DIR/results/program_stress_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_varied \
  --device mps --out "$QWENGPU_DIR/results/varied_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_rom_spike \
  --device mps --out "$QWENGPU_DIR/results/rom_spike_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_rom_spike \
  --device cpu --out "$QWENGPU_DIR/results/rom_spike_cpu.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_rom_spike \
  --device mps --extra-programs 97 \
  --out "$QWENGPU_DIR/results/rom_spike_stress_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_kernel \
  "$QWENGPU_DIR/examples/mixed_kernel.json" --device mps \
  --out "$QWENGPU_DIR/results/example_kernel_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_kernel \
  "$QWENGPU_DIR/examples/no_else_kernel.json" --device mps \
  --out "$QWENGPU_DIR/results/no_else_kernel_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_kernel \
  "$QWENGPU_DIR/examples/masked_overwrite_kernel.json" --device mps \
  --out "$QWENGPU_DIR/results/masked_overwrite_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_kernel \
  "$QWENGPU_DIR/examples/halt_only_kernel.json" --device mps \
  --out "$QWENGPU_DIR/results/halt_only_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_rom_gate_fixture \
  --device mps --out "$QWENGPU_DIR/results/rom_gate_host_mps.json"
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

## Integrated ROM variant

The ROM variant is a separate checkpoint with a different token protocol.
Use the same pinned source model, compiler, and environment above:

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_compiled_rom \
  --device mps --out "$QWENGPU_DIR/results/compiled_rom_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_compiled_rom \
  --device cuda --out "$QWENGPU_DIR/results/compiled_rom_cuda_fla.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --rom --out /path/to/Qwen3.5-9B-qwengpu-rom
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_compiled_rom \
  /path/to/Qwen3.5-9B-qwengpu-rom --device cuda \
  --out "$QWENGPU_DIR/results/compiled_rom_9b_cuda_fla.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.textcheck_9b \
  /path/to/Qwen3.5-9B /path/to/Qwen3.5-9B-qwengpu-rom --device cuda \
  --out "$QWENGPU_DIR/results/textcheck_rom_9b_cuda.json"
```

Transfer source using Git or disable macOS archive metadata with
`COPYFILE_DISABLE=1` when creating tar archives. The fixture loader ignores
AppleDouble `._` files. Check disk space before rebuilding: the ROM vocabulary
is larger than the earlier host-dispatched variant.


## Bounded internal writeback variant

Use a separate checkpoint directory for each circuit variant. The writeback
builder unrolls 16 instruction stages (51 circuit layers; 52 in the merge),
increasing layer depth while removing
all lane READ/result-feedback tokens. See [WRITEBACK_DESIGN.md](WRITEBACK_DESIGN.md).

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_writeback \
  --device mps --out "$QWENGPU_DIR/results/writeback_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_writeback \
  --device cuda --out "$QWENGPU_DIR/results/writeback_cuda_fla.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --writeback --out /path/to/Qwen3.5-9B-qwengpu-writeback
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_writeback \
  /path/to/Qwen3.5-9B-qwengpu-writeback --device cuda \
  --out "$QWENGPU_DIR/results/writeback_9b_cuda_fla.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.textcheck_9b \
  /path/to/Qwen3.5-9B /path/to/Qwen3.5-9B-qwengpu-writeback --device cuda \
  --out "$QWENGPU_DIR/results/textcheck_writeback_9b_cuda.json"
```

Pass `--kernel /path/to/kernel.json` to `qwengpu.eval_writeback` to run one
supported kernel and compare its entire machine trace to the interpreter.
The default executes the seven frozen gate inputs using one model object.
For an ephemeral CUDA pod, `scripts/bootstrap_runpod.sh` installs the pinned
compiler/environment and downloads the pinned source model. Transfer the
project source separately. This script does not provision compute.


The measured writeback checkpoint has 52,118,362,112 bytes of weights and
hidden size 5,120. A 100 GB ephemeral container disk held both the source
snapshot and merged checkpoint. The build/evaluation used a 96 GB GPU.
Set `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8` to reproduce the recorded 9B process
thread settings. The CUDA standalone run used the environment defaults.


## Loops and divergent control

Use the same pinned compiler, Qwen snapshot, and CUDA environment as above.
The H100 runs use `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_control \
  --device mps --out "$QWENGPU_DIR/results/control_mps_full.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_control \
  --device cuda --out "$QWENGPU_DIR/results/control_cuda_fla.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --control --out /path/to/Qwen3.5-9B-qwengpu-control
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_control \
  /path/to/Qwen3.5-9B-qwengpu-control --device cuda \
  --out "$QWENGPU_DIR/results/control_9b_cuda_fla.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.textcheck_9b \
  /path/to/Qwen3.5-9B /path/to/Qwen3.5-9B-qwengpu-control --device cuda \
  --out "$QWENGPU_DIR/results/textcheck_control_9b_cuda.json"
```

Use `--kernel path/to/kernel.json` to evaluate one kernel. A user kernel that
exhausts the budget fails the expected HALT outcome unless its JSON explicitly
sets `expected_outcome` to `exhausted`. In either case, the report retains the
trace and distinguishes correct execution from successful halting. The MPS
reports recorded the first 13 cases together and the later combined-depth
fixture separately; the default command now runs all 14.


The control merge was built and checked on an H100 with 80 GB VRAM and
251 GB host RAM. A 120 GB temporary container disk held source and merged
checkpoints. The merged weights total 61,946,385,152 bytes. The quoted secure
pod rate was $3.99/hour. GPU and disk were terminated after reports were
copied; weights are rebuilt from source and are not in this repository.

## Resuming execution windows

The resume variant emits 48 position-tagged tokens carrying the 380-bit warp
snapshot, plus a token containing the next PC with the local step reset. The
driver relays those tokens unchanged into a fresh cache alongside the original
ROM and link uploads. All execution windows use one model object.

The default fixture uses two levels of branches inside a data-dependent loop
and carries scratch values across window boundaries. It reaches HALT after
1,082 instructions. Use the pinned model and compiler environment above.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_resume \
  --device mps --max-steps 49 --allow-step-limit \
  --out "$QWENGPU_DIR/results/resume_mps_49.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resume --out /path/to/Qwen3.5-9B-qwengpu-resume
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_resume \
  /path/to/Qwen3.5-9B-qwengpu-resume --device cuda \
  --out "$QWENGPU_DIR/results/resume_9b_cuda_fla.json"
```

The local smoke command stops deliberately after 49 instructions. The full
command requires HALT within its 1,200-instruction guard and compares every
control token, cached snapshot, and exported handoff with an uninterrupted
interpreter.
