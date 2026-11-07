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

## Second register, indexed gather, and byte multiply

The ISA variant adds three zero-argument instructions. Each acts on active
lanes only.

| Instruction | Effect |
| --- | --- |
| `MOV_R1` | Copy the lane's accumulator R0 into R1. |
| `GATHER_R1` | Load R0 from scratch byte `R1 & 31`. |
| `MUL_R1` | Replace R0 with `(R0 * R1) & 255`. |

R1 starts at zero. Its eight bytes are appended to the existing snapshot,
for 444 bits and 56 exported state tokens per handoff. The ROM upload
preserves its existing address layout by using three reserved payloads for
the new instructions.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_resume \
  --isa --device mps --out "$QWENGPU_DIR/results/isa_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --isa --out /path/to/Qwen3.5-9B-qwengpu-isa
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_resume \
  /path/to/Qwen3.5-9B-qwengpu-isa --device cuda \
  --out "$QWENGPU_DIR/results/isa_9b_cuda_fla.json"
```

The default ISA fixture fills the scratch banks, runs a lane-dependent
loop across a window boundary, and then gathers and multiplies under a
divergent mask. Use `--kernel` to load a different program.

## Cooperative workgroups

The workgroup launcher accepts a JSON object containing exactly two entries in
`warps`. Each entry has eight `initial` bytes and up to 16 `program`
instructions ending in HALT. An optional `shared` array supplies the initial
32 scratch bytes. The default suite includes a cooperative reduction, two
interleaved loops, and a deliberate barrier deadlock.

The fixed model executes one warp instruction per call. The host schedules
ready warps in round-robin order and copies the generated private state and
shared scratch tokens into the next call. The launcher reports final R0
values for both warps and the shared bytes alongside the complete trace.

- `SCATTER_R1` writes each active lane's R0 to shared byte `R1 & 31`.
  The model chooses the lowest active lane number when addresses collide.
- `REDUCE_SUM` sums active R0 bytes modulo 256 and broadcasts the result
  to the active lanes.
- `BARRIER` advances the warp PC and emits a waiting status. The host
  releases both warps once both have arrived. The loader requires barriers
  at the top level, after any enclosing branch or loop has closed.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_workgroup \
  --device mps --out "$QWENGPU_DIR/results/workgroup_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --workgroup --out /path/to/Qwen3.5-9B-qwengpu-workgroup
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_workgroup \
  /path/to/Qwen3.5-9B-qwengpu-workgroup --device cuda \
  --out "$QWENGPU_DIR/results/workgroup_9b_cuda_fla.json"
```

Pass `--kernel path/to/workgroup.json` for one manifest and `--max-steps`
to change the aggregate instruction guard. A manifest can declare an
`expected_outcome` of `deadlock` or `step_limit` for negative cases. The
default expected outcome is `halted`.

## Model-scheduled tile workgroups

The tile variant runs two eight-lane warps. Each lane has a 16-bit
accumulator. A scheduler call reads both warp control tokens and the
previous ticket. The model selects the next ready warp, releases a barrier
when both warps are waiting, and distinguishes HALT from deadlock. The
host routes the emitted tickets and copies the model's state packets.

The 64-byte shared layout places row-major A at bytes 0–15, row-major B at
16–31, and the 16 little-endian C words at 32–63. Each warp owns two output
rows. The default program accumulates four dot-product terms in two
K phases of two terms each, then publishes its output rows. Arithmetic
wraps modulo 65,536.

| Tile instruction | Effect |
| --- | --- |
| `MAC_K k` | Multiply the selected unsigned byte operands from A and B and add the product to the lane's 16-bit accumulator. |
| `ADD16I imm` | Add an unsigned byte immediate to each 16-bit accumulator. |
| `STORE_TILE` | Publish the warp's eight accumulators to its C rows. |
| `BARRIER` | Emit a waiting control token and advance the PC. |
| `HALT` | Emit a halted control token. |

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile \
  --device mps --out "$QWENGPU_DIR/results/tile_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --tile --out /path/to/Qwen3.5-9B-qwengpu-tile
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile \
  /path/to/Qwen3.5-9B-qwengpu-tile --device cuda \
  --out "$QWENGPU_DIR/results/tile_9b_cuda_fla.json"
```

A custom manifest passed through `--kernel` can supply `a` and `b` as
row-major arrays of 16 unsigned bytes each. It can optionally supply 16
initial accumulator words or two explicit programs with up to 16
instructions each. The report includes the resulting matrix, the model's
scheduling decisions, and complete register and memory snapshots.

## 8×8 tiles and model-owned row-pair advancement

The 8×8 variant reuses two eight-lane warps over four row pairs. The
scheduler emits a new row-pair ticket and resets both PCs after both
warps halt within a pair. Each warp reads its C row, accumulates eight
products in four K panels, and publishes the row as address-tagged writes.
The host copies those writes to their indicated slots and relays state.

Shared memory contains 64 unsigned A bytes, 64 unsigned B bytes, and
128 bytes for the 64 little-endian C words. The manifest supplies
row-major `a` and `b` arrays of 64 bytes each. Optional `c` supplies
64 initial unsigned 16-bit words. The default kernel computes
`C = (C + A @ B) mod 65536`. The model loads C before the first MAC for
each row, so it can accumulate into an existing matrix.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --device mps --out "$QWENGPU_DIR/results/tile8_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --tile8 --out /path/to/Qwen3.5-9B-qwengpu-tile8
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  /path/to/Qwen3.5-9B-qwengpu-tile8 --device cuda \
  --out "$QWENGPU_DIR/results/tile8_9b_cuda_fla.json"
```

Pass `--kernel` for a custom manifest. The evaluator checks each scheduling
decision, register snapshot, tagged write, and final matrix against the
interpreter and a direct scalar matrix calculation.

## Resident 8×8 tile memory

The resident variant uploads A, B, and initial C once. It passes one
inference cache through all scheduling and execution calls and feeds back
tagged C writes before the next scheduling call. The scheduler's emitted
ticket selects the active control row. Register packets and ROM uploads
still travel between calls.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --resident --device mps --out "$QWENGPU_DIR/results/resident_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resident --out /path/to/Qwen3.5-9B-qwengpu-resident
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  /path/to/Qwen3.5-9B-qwengpu-resident --device cuda \
  --out "$QWENGPU_DIR/results/resident_9b_cuda_fla.json"
```

The default resident suite includes the 8×8 seeded multiply, deadlock,
and a case that overwrites an accumulator after a store and then reloads
from C. The reports count input tokens and check the final A, B, and C
rows directly in the model cache.

## Indexed resident shared memory

The indexed variant extends the resident tile engine with a 7-bit index
register per lane. `SAVE_INDEX` copies the low seven bits of the 16-bit
accumulator to that register. `LOAD_INDEX` reads the selected shared word
into the accumulator. `STORE_INDEX` writes the accumulator to the selected
word. The lowest lane number wins when several lanes in one instruction
target the same word. Stores from different warp turns take effect in
scheduler order.

Word addresses 0–31 cover A and 32–63 cover B, with two consecutive bytes
per word in little-endian order. Addresses 64–127 cover C. Indexed and
tiled operations share the same contents. The model computes the
addresses and emits tagged write tokens. The host feeds those tokens
back into the same inference cache before the next scheduling call and
relays the per-warp accumulator and index packets.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --indexed --device mps --out "$QWENGPU_DIR/results/indexed_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --indexed --out /path/to/Qwen3.5-9B-qwengpu-indexed
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  /path/to/Qwen3.5-9B-qwengpu-indexed --device cuda \
  --out "$QWENGPU_DIR/results/indexed_9b_cuda_fla.json"
```

The default suite includes the resident fixtures, a cross-warp
scatter-and-gather case, and a read/write sweep of all 128 word
addresses. The sweep also verifies that writes from indexed operations
are visible to tile loads and tile stores are visible to indexed loads.

## Lane masks in the resident engine

The masked variant extends the indexed engine. Each warp starts with all
eight lanes active. `MASK_BIT k` replaces the mask with accumulator bit
`k` from each lane, and `MASK_ALL` activates every lane. Both mask
instructions, the barrier, and halt operate at warp scope. The mask
persists across scheduling turns and row pairs.

Inactive lanes retain their accumulators and index registers and emit
no memory writes. Indexed scatter awards each address to its lowest
active lane. The model emits the updated mask as two tokens, which the
host relays with the register packet.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --masked --device mps --out "$QWENGPU_DIR/results/masked_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --masked --out /path/to/Qwen3.5-9B-qwengpu-masked
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  /path/to/Qwen3.5-9B-qwengpu-masked --device cuda \
  --out "$QWENGPU_DIR/results/masked_9b_cuda_fla.json"
```

## Nested branches in the resident engine

`IF_BIT k` selects the lanes that are already active and have
accumulator bit `k` set. The model saves the parent and selected masks.
`ELSE` selects the other lanes from the saved parent. `ENDIF` restores
the parent mask. `ELSE` is optional, and each warp supports two levels
of nesting. Mask replacement, barriers, and halt require a
reconverged warp. The host relays the model's branch packet with the
other per-warp registers.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --divergent --device mps --out "$QWENGPU_DIR/results/divergent_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --divergent --out /path/to/Qwen3.5-9B-qwengpu-divergent
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  /path/to/Qwen3.5-9B-qwengpu-divergent --device cuda \
  --out "$QWENGPU_DIR/results/divergent_9b_cuda_fla.json"
```

The two branch fixtures exercise a predicate whose source changes in
the first path, reconvergence to a partial mask, and an empty nested
path that emits no writes.

## Data-dependent loops in the resident engine

`WHILE_BIT k` tests accumulator bit `k` for each active lane. If any
lane qualifies, the model saves the parent mask and enters the body
with those lanes. Otherwise it selects the instruction after the
matching `ENDWHILE`. At `ENDWHILE` it restores the parent mask and
selects the matching `WHILE_BIT` to retest the lanes. The assembler
links the two program locations when it uploads the ROM. Branch and
loop stacks each support two levels per warp. Mask replacement,
barriers, and halt require a reconverged warp.

The host relays the model's control, stack, and write packets, and
enforces the evaluator's instruction budget. The suite includes
different per-lane trip counts, nested loops with a branch, a gather
after a two-warp barrier, and an intentionally nonterminating loop that
reaches the default 256-instruction budget.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --looping --device mps --out "$QWENGPU_DIR/results/looping_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --looping --out /path/to/Qwen3.5-9B-qwengpu-looping
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  /path/to/Qwen3.5-9B-qwengpu-looping --device cuda \
  --out "$QWENGPU_DIR/results/looping_9b_cuda_fla.json"
```

## Cooperative matrix multiply and gather

The cooperative fixture runs on the looping checkpoint. Warp 0
accumulates eight byte products into each 16-bit word of an even row
and publishes the row with a tiled store. Warp 1 prepares indexed
addresses from the adjacent odd row and waits at the barrier. It
gathers the published words in a permuted column order, applies a
divergent loop, and stores its own row.

The fixture freezes the result of a separate scalar calculation.
For even row `r` and column `j`, the result is
`(initial_c[r, j] + sum(a[r, k] * b[k, j] for k in range(8))) % 65536`.
For odd row `r + 1`, the initial word selects a producer address
`64 + 8*r + j`. Starting from the published value at that address, the
consumer adds 1 to an odd value or 2 to an even value while bit 2 is set.
The evaluator checks this expected matrix in addition to the full
instruction and scheduling trace.

```sh
QWENGPU_COOPERATIVE_KERNEL="$QWENGPU_DIR/eval/frozen/loop_gate_v1/cooperative_matmul.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 --looping \
  --kernel "$QWENGPU_COOPERATIVE_KERNEL" --device mps \
  --out "$QWENGPU_DIR/results/cooperative_matmul_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  /path/to/Qwen3.5-9B-qwengpu-looping \
  --kernel "$QWENGPU_COOPERATIVE_KERNEL" --device cuda \
  --out "$QWENGPU_DIR/results/cooperative_matmul_9b_cuda_fla.json"
```

## Run a resident kernel from JSON

Launch the looping checkpoint with `qwengpu.run_resident`. Use flat
64-element arrays `a` and `b` for unsigned bytes and `c` for unsigned
16-bit words, all in row-major order. Omitted arrays start at zero.
`programs` contains two arrays of one through sixteen instruction
objects, each ending in `HALT`. An instruction has an `op` and an
optional integer `arg`. Omitting `programs` selects the dense tiled
multiply. Evaluation fixtures can include `expected_matrix` and
`expected_outcome` for use by the evaluator.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
  "$QWENGPU_DIR/eval/frozen/loop_gate_v1/cooperative_matmul.json" \
  --device mps --out "$QWENGPU_DIR/results/cooperative_launch_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
  "$QWENGPU_DIR/eval/frozen/loop_gate_v1/cooperative_matmul.json" \
  --model /path/to/Qwen3.5-9B-qwengpu-looping --device cuda --trace \
  --out "$QWENGPU_DIR/results/cooperative_launch_9b_cuda_fla.json"
```

The result identifies its format, checkpoint, and input digest. It
contains the final matrix, all 128 shared words, transport counts, and
the terminal outcome. The launcher writes JSON to stdout unless given
`--out`. Pass `--trace` to include instruction snapshots and scheduler
tickets. Set `--max-steps` to override the default launch budget of 256.
Exit codes are 0 for halt, 3 for deadlock, and 4 for budget exhaustion.
The launcher writes the result before returning any of these codes.

## Resident warp programs

The resident-ROM checkpoint uploads two padded 16-slot programs and
two bank-select tokens at launch. Both programs stay in the same
recurrent ROM table. The model selects a row from the scheduled warp
and its PC. The launcher sends no further program tokens.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --resident-rom --device mps --out "$QWENGPU_DIR/results/resident_rom_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resident-rom --out /path/to/Qwen3.5-9B-qwengpu-resident-rom
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  /path/to/Qwen3.5-9B-qwengpu-resident-rom --device cuda \
  --out "$QWENGPU_DIR/results/resident_rom_9b_cuda_fla.json"
```

The evaluator checks all 32 decoded ROM entries in the final cache.
Use `--resident-rom` with the standalone JSON launcher, or select the
merged checkpoint with `--model`.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
  "$QWENGPU_DIR/eval/frozen/loop_gate_v1/cooperative_matmul.json" \
  --resident-rom --device mps --out "$QWENGPU_DIR/results/resident_rom_launch_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
  "$QWENGPU_DIR/eval/frozen/loop_gate_v1/cooperative_matmul.json" \
  --model /path/to/Qwen3.5-9B-qwengpu-resident-rom --device cuda \
  --out "$QWENGPU_DIR/results/resident_rom_launch_9b_cuda_fla.json"
```

## Resident warp register banks

The resident-state checkpoint initializes two banks in the launch cache.
Each bank holds eight 16-bit accumulators, eight 7-bit indices, an
eight-bit lane mask, and the branch and loop stacks. The scheduler's
ticket selects the bank used by the next instruction. The model emits
state tokens tagged with that bank, and the host forwards the packet
unchanged before the next scheduling call. The evaluator checks both
banks directly and keeps a canonical trace for comparison.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --resident-state --device mps --out "$QWENGPU_DIR/results/resident_state_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resident-state --out /path/to/Qwen3.5-9B-qwengpu-resident-state
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  /path/to/Qwen3.5-9B-qwengpu-resident-state --device cuda \
  --out "$QWENGPU_DIR/results/resident_state_9b_cuda_fla.json"
```

Use `--resident-state` for the standalone JSON launcher, or pass the
merged resident-state directory with `--model`.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
  "$QWENGPU_DIR/eval/frozen/loop_gate_v1/cooperative_matmul.json" \
  --resident-state --device mps --out "$QWENGPU_DIR/results/resident_state_launch_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
  "$QWENGPU_DIR/eval/frozen/loop_gate_v1/cooperative_matmul.json" \
  --model /path/to/Qwen3.5-9B-qwengpu-resident-state --device cuda \
  --out "$QWENGPU_DIR/results/resident_state_launch_9b_cuda_fla.json"
```

## Changed-only warp state updates

The resident-delta checkpoint compares each new state nibble with the
active warp's cached value. It emits a tagged update when the value
changes and a zero sentinel otherwise. The host forwards the emitted
tags unchanged. All state read requests still run, and the verifier
reconstructs the canonical trace and checks the bank contents.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --resident-delta --device mps --out "$QWENGPU_DIR/results/resident_delta_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resident-delta --out /path/to/Qwen3.5-9B-qwengpu-resident-delta
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  /path/to/Qwen3.5-9B-qwengpu-resident-delta --device cuda \
  --out "$QWENGPU_DIR/results/resident_delta_9b_cuda_fla.json"
```

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
  "$QWENGPU_DIR/eval/frozen/loop_gate_v1/cooperative_matmul.json" \
  --resident-delta --device mps --out "$QWENGPU_DIR/results/resident_delta_launch_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
  "$QWENGPU_DIR/eval/frozen/loop_gate_v1/cooperative_matmul.json" \
  --model /path/to/Qwen3.5-9B-qwengpu-resident-delta --device cuda \
  --out "$QWENGPU_DIR/results/resident_delta_launch_9b_cuda_fla.json"
```

## Model-generated state read plans

The resident-plan checkpoint emits eight tagged bytes after each tick.
Together they form a bitmap of changed warp-state nibbles. The launcher
requests the selected positions and forwards the model's tagged values.
It still requests all 32 shared-store positions. The extra read phase
adds one forward call per instruction. The report counts plan requests,
state read requests, and total forward calls separately.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --resident-plan --device mps --out "$QWENGPU_DIR/results/resident_plan_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resident-plan --out /path/to/Qwen3.5-9B-qwengpu-resident-plan
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  /path/to/Qwen3.5-9B-qwengpu-resident-plan --device cuda \
  --out "$QWENGPU_DIR/results/resident_plan_9b_cuda_fla.json"
```

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
  "$QWENGPU_DIR/eval/frozen/loop_gate_v1/cooperative_matmul.json" \
  --resident-plan --device mps --out "$QWENGPU_DIR/results/resident_plan_launch_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
  "$QWENGPU_DIR/eval/frozen/loop_gate_v1/cooperative_matmul.json" \
  --model /path/to/Qwen3.5-9B-qwengpu-resident-plan --device cuda \
  --out "$QWENGPU_DIR/results/resident_plan_launch_9b_cuda_fla.json"
```

## Model-generated state and store read plans

The resident I/O checkpoint adds one byte to the state read plan.
Its eight bits select the lanes that issue shared-memory writes.
The launcher reads four store nibbles per selected lane and requests
the state positions selected by the original bitmap. It skips the
follow-up when both sets are empty. The verifier checks both masks
against the model snapshot and checks the full canonical trace.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --resident-io --device mps --out "$QWENGPU_DIR/results/resident_io_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resident-io --out /path/to/Qwen3.5-9B-qwengpu-resident-io
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  /path/to/Qwen3.5-9B-qwengpu-resident-io --device cuda \
  --out "$QWENGPU_DIR/results/resident_io_9b_cuda_fla.json"
```

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
  "$QWENGPU_DIR/eval/frozen/loop_gate_v1/cooperative_matmul.json" \
  --resident-io --device mps --out "$QWENGPU_DIR/results/resident_io_launch_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
  "$QWENGPU_DIR/eval/frozen/loop_gate_v1/cooperative_matmul.json" \
  --model /path/to/Qwen3.5-9B-qwengpu-resident-io --device cuda \
  --out "$QWENGPU_DIR/results/resident_io_launch_9b_cuda_fla.json"
```

## Compare resident transport modes on one GPU

Use a merged resident I/O checkpoint. The benchmark runs fixed state
and store requests, state-only plans, and state plus store plans on
the same loaded weights. Each launch starts with a fresh inference
cache. It profiles one warmup per mode, then runs six rounds with
every permutation of the three modes represented once.

```sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
uptime
nproc
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,pstate,clocks.current.sm,power.limit --format=csv
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.bench_resident_transport \
  "$QWENGPU_DIR/eval/frozen/loop_gate_v1/cooperative_matmul.json" \
  --model /path/to/Qwen3.5-9B-qwengpu-resident-io \
  --out "$QWENGPU_DIR/results/resident_transport_benchmark_a100.json"
```

The timer covers model execution, host relay, and in-launch cache
verification. CUDA synchronization bounds each sample. Checkpoint
loading, warmup profiling, and result hashing are separate. A
forward hook counts model invocations, and peak allocated bytes
include the loaded weights. The benchmark checks every canonical
trace against the interpreter and the frozen matrix. It writes each
sample as it finishes and records failures before exiting.

## Select a resident I/O transport

Use one resident I/O checkpoint to run any of the three transports.
The launcher accepts `fixed`, `state-plan`, and `state-store-plan`.
Pass `--resident-io` for the standalone checkpoint or select a merged
resident I/O directory with `--model`.

```sh
QWENGPU_RELOAD_KERNEL="$QWENGPU_DIR/eval/frozen/resident_gate_v1/store_reload.json"
for mode in fixed state-plan state-store-plan; do
  "$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
    "$QWENGPU_RELOAD_KERNEL" --model /path/to/Qwen3.5-9B-qwengpu-resident-io \
    --transport "$mode" --device cuda --trace --out "results/$mode.json"
done
```

An explicit choice appears as `transport_mode` in the result.
Omitting `--transport` uses the checkpoint's existing transport.
The timing harness accepts the same fixture to compare the three
modes with repeated warm launches.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.bench_resident_transport \
  "$QWENGPU_RELOAD_KERNEL" --model /path/to/Qwen3.5-9B-qwengpu-resident-io \
  --out "$QWENGPU_DIR/results/store_reload_transport_benchmark_a100.json"
```

## Launch a subset of the tile rows

The resident grid checkpoint stores one launch token with the index of
the last row pair. The scheduler and warp resets both read it. Omit
`row_pairs` for a full tile or set it to 1, 2, or 3 to run the first
2, 4, or 6 rows. The verifier compares the complete trace and the
default kernel's matrix with the interpreter and checks the stored extent
and shared memory directly in the model cache.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --resident-grid --device mps --out "$QWENGPU_DIR/results/resident_grid_mps.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resident-grid --out /path/to/Qwen3.5-9B-qwengpu-resident-grid
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  /path/to/Qwen3.5-9B-qwengpu-resident-grid --device cuda \
  --out "$QWENGPU_DIR/results/resident_grid_9b_cuda_fla.json"
```

```sh
QWENGPU_PARTIAL_KERNEL="$QWENGPU_DIR/eval/frozen/grid_gate_v1/partial_1_pair.json"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
  "$QWENGPU_PARTIAL_KERNEL" --resident-grid --device mps --trace \
  --out "$QWENGPU_DIR/results/resident_grid_launch_mps.json"
for mode in fixed state-plan state-store-plan; do
  "$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
    "$QWENGPU_PARTIAL_KERNEL" --model /path/to/Qwen3.5-9B-qwengpu-resident-grid \
    --device cuda --transport "$mode" --trace \
    --out "$QWENGPU_DIR/results/resident_grid_launch_${mode}_9b_cuda_fla.json"
done
```

After saving both suite reports and the three launcher reports, compare
the MPS and CUDA results with each other and the prior full-tile traces:

```sh
python3 "$QWENGPU_DIR/scripts/check_resident_grid.py"
```

The two-HALT diagnostic isolates the warp-reset dependency. With the grid
head queried only by `SCHEDULE`, both reset tokens missed the extent and
the scheduler skipped a warp turn at the first advance. The saved
`resident_grid_schedule_only_mps.json` records the failing trace. Replacing
the grid head's query selector with `schedule` reproduces that failure;
the current selector also includes both reset tokens.

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --resident-grid --device mps \
  --kernel "$QWENGPU_DIR/eval/diagnostics/resident_grid_halt.json" \
  --out "$QWENGPU_DIR/results/resident_grid_halt_mps.json"
```

## Verify row windows and their composition

Use the resident grid merge from the preceding section. The launcher puts
the first pair in the initial ticket and the last pair in the grid token.
The circuit and merge recipe are the same. The five manifests share
the same A, B, and initial C. The evaluator compares each launch with
the independent interpreter on one loaded model per device.

```sh
cd "$QWENGPU_DIR"
QWENGPU_GRID_MODEL=/path/to/Qwen3.5-9B-qwengpu-resident-grid
QWENGPU_WINDOW_FIXTURES=(
  --kernel eval/frozen/tile8_gate_v1/dense_seeded.json
  --kernel eval/frozen/window_gate_v1/front_two.json
  --kernel eval/frozen/window_gate_v1/back_two.json
  --kernel eval/frozen/window_gate_v1/middle_one.json
  --kernel eval/frozen/window_gate_v1/last_one.json
)
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --resident-grid --device mps "${QWENGPU_WINDOW_FIXTURES[@]}" \
  --out results/resident_windows_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  "$QWENGPU_GRID_MODEL" --device cuda "${QWENGPU_WINDOW_FIXTURES[@]}" \
  --out results/resident_windows_9b_cuda_fla.json
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
  eval/frozen/window_gate_v1/last_one.json --model "$QWENGPU_GRID_MODEL" \
  --device cuda --transport fixed --trace \
  --out results/resident_windows_launch_9b_cuda_fla.json
python3 scripts/check_resident_windows.py
```

The final comparison checks that rows outside each window keep the
seeded C values and that stitching the front and back outputs reproduces
the full tile. The last-pair launcher manifest omits `row_pairs` to
exercise the default extent.

## Compose a 16×16 matrix multiply on one checkpoint

Each of the eight launches uses the resident grid circuit and a fresh
cache. The runner loads one model and uses fixed read requests. The
launcher feeds each 8×8 output tile back as C for the second K slice.
The model computes the multiply-accumulates modulo 65536.

```sh
cd "$QWENGPU_DIR"
QWENGPU_MATMUL16=eval/frozen/matmul16_gate_v1/dense_seeded.json
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_matmul16 "$QWENGPU_MATMUL16" \
  --device mps --out results/resident_matmul16_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_matmul16 "$QWENGPU_MATMUL16" \
  --model /path/to/Qwen3.5-9B-qwengpu-resident-grid \
  --device cuda --out results/resident_matmul16_9b_cuda_fla.json
python3 scripts/check_matmul16.py
```

The checker computes both partial products independently from the original
inputs. It compares all 256 entries after each K slice and checks that the
MPS and CUDA reports have identical per-tile trace hashes and transport
records. Pass a single report path to check one device.

## Check masked 32-bit addition

The ADD32 prototype runs eight lanes in one model operation. Each
word is uploaded as four bytes. A single execute token computes the
sums and writes a snapshot to the inference cache. The verifier
reads all 32 result bytes in one forward and compares the emitted
tokens and cached results with modular integer addition. It also
reads the operand and mask rows directly from the cache.

```sh
cd "$QWENGPU_DIR"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_add32 \
  --device mps --out results/add32_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_add32 \
  --device cuda --out results/add32_cuda_fla.json
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --add32 --out /path/to/Qwen3.5-9B-qwengpu-add32
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_add32 \
  /path/to/Qwen3.5-9B-qwengpu-add32 --device cuda --out results/add32_9b_cuda_fla.json
python3 - <<'PY'
import json
from pathlib import Path

names = ("add32_mps", "add32_cuda_fla", "add32_9b_cuda_fla")
reports = [json.loads((Path("results") / f"{name}.json").read_text()) for name in names]
assert all(report["all_exact"] for report in reports)
assert len({report["fixture_sha256"] for report in reports}) == 1
assert reports[0]["cases"] == reports[1]["cases"] == reports[2]["cases"]
PY
```

The shared carry decoder also serves the resident grid. Repeat the
dense tile regression with:

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --resident-grid --device mps --kernel eval/frozen/tile8_gate_v1/dense_seeded.json \
  --out results/add32_resident_regression_mps.json
```

## Verify high register banks in the resident grid

The wide layout uses one cache with 96 state nibbles per warp. A
launch may seed two arrays of eight unsigned 32-bit words. The
model snapshots the high halves for the scheduled warp at every
instruction. The verifier checks them and reads both register
banks directly from the cache at the end.

```sh
cd "$QWENGPU_DIR"
QWENGPU_WIDE_MODEL=/path/to/Qwen3.5-9B-qwengpu-resident-wide
QWENGPU_WIDE_FIXTURES=(
  --kernel eval/frozen/wide_gate_v1/seeded_warps.json
  --kernel eval/frozen/wide_gate_v1/seeded_cooperative.json
  --kernel eval/frozen/tile8_gate_v1/dense_seeded.json
)
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --resident-wide --device mps "${QWENGPU_WIDE_FIXTURES[@]}" \
  --out results/resident_wide_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resident-wide --out "$QWENGPU_WIDE_MODEL"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  "$QWENGPU_WIDE_MODEL" --device cuda "${QWENGPU_WIDE_FIXTURES[@]}" \
  --out results/resident_wide_9b_cuda_fla.json
for mode in fixed state-plan; do
  "$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
    eval/frozen/wide_gate_v1/seeded_warps.json --resident-wide --device mps \
    --transport "$mode" --trace --out "results/resident_wide_launch_${mode}_mps.json"
  "$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
    eval/frozen/wide_gate_v1/seeded_warps.json --model "$QWENGPU_WIDE_MODEL" \
    --device cuda --transport "$mode" --trace \
    --out "results/resident_wide_launch_${mode}_9b_cuda_fla.json"
done
python3 scripts/check_resident_wide.py
```

The evaluator uses the state and store plan. The final checker compares
the fixed and state-only launcher runs with that result and checks
the dense and cooperative traces against the previous grid reports.

## Verify ADD32I alongside tiled and cooperative kernels

The resident ADD32 layout extends the ROM payload by one opcode bit.
It accepts an unsigned byte immediate and propagates carry from bit
15 into bit 16. The model emits changes to either register half
through the existing dirty-nibble protocol. The mixed fixture executes
ADD32I, a tiled multiply, a shared-memory gather, and a divergent loop
in one launch.

```sh
cd "$QWENGPU_DIR"
QWENGPU_ADD32_MODEL=/path/to/Qwen3.5-9B-qwengpu-resident-add32
QWENGPU_ADD32_FIXTURES=(
  --kernel eval/frozen/resident_add32_gate_v1/masked_carry.json
  --kernel eval/frozen/resident_add32_gate_v1/cooperative_add32.json
  --kernel eval/frozen/wide_gate_v1/seeded_cooperative.json
  --kernel eval/frozen/tile8_gate_v1/dense_seeded.json
)
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  --resident-add32 --device mps "${QWENGPU_ADD32_FIXTURES[@]}" \
  --out results/resident_add32_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resident-add32 --out "$QWENGPU_ADD32_MODEL"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  "$QWENGPU_ADD32_MODEL" --device cuda "${QWENGPU_ADD32_FIXTURES[@]}" \
  --out results/resident_add32_9b_cuda_fla.json
for mode in fixed state-plan; do
  "$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
    eval/frozen/resident_add32_gate_v1/masked_carry.json --resident-add32 --device mps \
    --transport "$mode" --trace --out "results/resident_add32_launch_${mode}_mps.json"
  "$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
    eval/frozen/resident_add32_gate_v1/masked_carry.json --model "$QWENGPU_ADD32_MODEL" \
    --device cuda --transport "$mode" --trace \
    --out "results/resident_add32_launch_${mode}_9b_cuda_fla.json"
done
python3 scripts/check_resident_add32.py
```

To check every immediate against the ROM codec:

```sh
"$QWENPUTER_DIR/.venv/bin/python" - <<'PY'
from qwengpu.compiled_tile8 import ROM_PAYLOAD_BITS, program_tokens, rom_payload

flags = dict(indexed=True, masked=True, divergent=True, looping=True, resident_add32=True)
tokens = [program_tokens((("ADD32I", n), ("HALT", 0)))[0] for n in range(256)]
assert len(set(tokens)) == 256
for n, token in enumerate(tokens):
    assert rom_payload(token, **flags) == (1 << ROM_PAYLOAD_BITS) + n
PY
```

## Run the 16×16 multiply on the ADD32 checkpoint

Use the resident ADD32 merge from the previous section. The same
checkpoint runs the cooperative ADD32I fixture and the composed
multiply. The launcher chooses the merged layout from its metadata.
The multiply uses the existing 16-bit modular tile instructions.

```sh
cd "$QWENGPU_DIR"
QWENGPU_MATMUL16=eval/frozen/matmul16_gate_v1/dense_seeded.json
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_matmul16 "$QWENGPU_MATMUL16" \
  --resident-add32 --device mps --out results/resident_add32_matmul16_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_matmul16 "$QWENGPU_MATMUL16" \
  --model /path/to/Qwen3.5-9B-qwengpu-resident-add32 \
  --device cuda --out results/resident_add32_matmul16_9b_cuda_fla.json
python3 scripts/check_matmul16.py \
  results/resident_add32_matmul16_mps.json \
  results/resident_add32_matmul16_9b_cuda_fla.json
```

The checker compares all 256 values after each K slice with a
scalar reference and the earlier grid run. The new reports keep
the full trace hash and a second hash over the fields shared with
the grid layout. The latter includes all scheduler decisions and
removes only the added high-half snapshot prefix.

To verify the wider arithmetic on the exact merged directory used for
the multiply, run the mixed cooperative fixture with the same model path:

```sh
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tile8 \
  /path/to/Qwen3.5-9B-qwengpu-resident-add32 --device cuda \
  --kernel eval/frozen/resident_add32_gate_v1/cooperative_add32.json \
  --out results/resident_add32_matmul16_cooperative_9b_cuda_fla.json
python3 - <<'PY'
import json
from pathlib import Path

results = Path("results")
mixed = json.loads((results / "resident_add32_matmul16_cooperative_9b_cuda_fla.json").read_text())
multiply = json.loads((results / "resident_add32_matmul16_9b_cuda_fla.json").read_text())
prior = json.loads((results / "resident_add32_9b_cuda_fla.json").read_text())
assert mixed["all_exact"] and mixed["merge"] == multiply["merge"] == prior["merge"]
assert mixed["cases"] == [case for case in prior["cases"] if case["fixture"] == "cooperative_add32.json"]
PY
```

## Qualify the unified resident ADD32 checkpoint

Run the inherited resident, indexed, masked, divergent, loop, grid,
wide, and ADD32 fixtures along with all four row windows. The shell
runner passes the same 23 manifests to one loaded model on each device.

```sh
cd "$QWENGPU_DIR"
export QWENGPU_PYTHON="$QWENPUTER_DIR/.venv/bin/python"
bash scripts/qualify_resident_add32.sh --device mps \
  --out results/resident_add32_full_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$QWENGPU_PYTHON" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resident-add32 --out /path/to/Qwen3.5-9B-qwengpu-resident-add32
bash scripts/qualify_resident_add32.sh /path/to/Qwen3.5-9B-qwengpu-resident-add32 \
  --device cuda --out results/resident_add32_full_9b_cuda_fla.json
python3 scripts/check_resident_add32_full.py
```

The evaluator compares every instruction, scheduler decision, and
cache check with the interpreter. The final checker compares the
MPS and CUDA records, then projects inherited grid snapshots onto
their original fields and checks the original terminal outcomes.

## Compare fresh-cache text on the unified merge

The frozen corpus contains the original six prompts plus 18 additional
prompts covering multilingual text, copying, arithmetic, code, structured
output, and a longer retrieval task. The evaluator first verifies the
same prompt token IDs on both tokenizers. It then uses fresh caches and
the same greedy decoding configuration, including the base model's
end-of-sequence IDs.

```sh
cd "$QWENGPU_DIR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
QWENGPU_BASE=/path/to/Qwen3.5-9B
QWENGPU_UNIFIED=/path/to/Qwen3.5-9B-qwengpu-resident-add32
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.merge9b "$QWENGPU_BASE" \
  --resident-add32 --out "$QWENGPU_UNIFIED"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.textcheck_9b \
  "$QWENGPU_BASE" "$QWENGPU_UNIFIED" --device cuda \
  --corpus eval/frozen/text_add32_gate_v1/prompts.json --max-new-tokens 16 \
  --out results/resident_add32_text_9b_cuda_fla.json
```

The report measures next-token logit differences over the original
vocabulary. It records both greedy continuations, their shared prefix
length, and any emitted chip tokens. Generation uses the full logits
of each model.

The tokenizer file checks use the same two model directories:

```sh
for file in tokenizer.json tokenizer_config.json chat_template.jinja; do
  cmp "$QWENGPU_BASE/$file" "$QWENGPU_UNIFIED/$file"
done
```

## Measure text after a GPU launch in one cache

Run the handoff comparison on the same resident ADD32 merge used for
the fresh-cache text check. The harness uses prompts 0, 1, 21, and 23
from the frozen corpus, with zero-based indices. It verifies the
cooperative ADD32I kernel against the interpreter before appending
the prompt to the kernel's final cache. It replays the kernel for
each prompt.

```sh
cd "$QWENGPU_DIR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_text_handoff \
  /path/to/Qwen3.5-9B-qwengpu-resident-add32 --device cuda \
  --out results/resident_add32_handoff_9b_cuda_fla.json
```

Each prompt is measured with a fresh cache, the GPU cache, and a fresh
cache with text positions starting at the GPU cache length. The third
condition measures the effect of shifting positions while holding the
cache history empty. Each condition uses up to 16 greedy tokens over
the full vocabulary. The report records cache lengths, prompt IDs,
kernel trace digests, native next-token logit differences, and generated
token IDs.

## Clear text history while keeping GPU state

Run the same four-prompt experiment with the reset policy. The
boundary replaces the pretrained layers' cache entries with fresh
entries from the same model configuration. It keeps the GPU layers
in the same dynamic cache and checks that text starts at position
zero. The verifier reads 128 register nibbles, 4,096 indexed memory
nibbles across the lane banks, and 512 nibbles in the tiled memory
view before and after generating text.

```sh
cd "$QWENGPU_DIR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_text_handoff \
  /path/to/Qwen3.5-9B-qwengpu-resident-add32 --device cuda --reset-text \
  --out results/resident_add32_handoff_reset_9b_cuda_fla.json
```

The report compares native next-token logits and greedy continuations
with a fresh cache. It records hashes of the GPU rows before and after
the text and whether every nibble matches exactly.

## Run a second kernel after text without uploading shared memory

The round trip runs the cooperative ADD32I producer, resets the
pretrained cache entries, and generates the frozen Moon prompt's
continuation. It checks that text matches a fresh cache and that the
GPU register and memory rows are exact. It then resets the pretrained
entries again and launches a consumer in the retained GPU cache.

```sh
cd "$QWENGPU_DIR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_cache_relaunch \
  /path/to/Qwen3.5-9B-qwengpu-resident-add32 --device cuda \
  --out results/resident_add32_cache_relaunch_9b_cuda_fla.json
```

The consumer uploads its own control, program, and register state.
It uses indexed reads to copy the first two produced rows into rows
six and seven. The host checks the result against the interpreter
and a fresh-cache consumer. A model forward hook audits cache identity
and shared-memory tokens for every consumer call, including the
tags returned for its own stores.

## Accumulate two slices into one resident output tile

Use the top-left output tile of the frozen 16×16 multiply. The first
launch uploads A, B, and initial C. The second launch refreshes A and
B in the same cache and reads the partial C from the model. A separate
fresh-cache run serves as the second-slice control.

```sh
cd "$QWENGPU_DIR"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_cached_tile \
  --device mps --out results/resident_add32_cached_tile_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_cached_tile \
  --model /path/to/Qwen3.5-9B-qwengpu-resident-add32 --device cuda \
  --out results/resident_add32_cached_tile_9b_cuda_fla.json
python3 scripts/check_cached_tile.py
```

The evaluator checks both scalar results and each instruction against
the interpreter. A forward hook verifies the cache identity and counts
the operand and C tokens actually sent to the model. The checker also
compares the trace digests with the earlier composed multiply.

## Retain partial outputs in the 16×16 launcher

Run both K slices for one output tile before moving to the next tile.
The ADD32 launcher starts four fresh caches and uses each for two
launches. Each second launch refreshes A and B and retains C.

```sh
cd "$QWENGPU_DIR"
fixture=eval/frozen/matmul16_gate_v1/dense_seeded.json
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_matmul16 "$fixture" \
  --resident-add32 --retain-c --device mps \
  --out results/resident_add32_matmul16_retained_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_matmul16 "$fixture" \
  --model /path/to/Qwen3.5-9B-qwengpu-resident-add32 --retain-c --device cuda \
  --out results/resident_add32_matmul16_retained_9b_cuda_fla.json
python3 scripts/check_matmul16.py results/resident_add32_matmul16_retained_*.json
```

The checker compares both pass matrices with the scalar reference
and all eight traces with the fixed-transport launcher. The reports
record the operand and C tokens sent to each model forward, including
the C tokens relayed for stores.

## Qualify 32-bit indexed exchange between warps

The shared32 checkpoint extends resident ADD32 with `LOAD_INDEX_HI`
and `STORE_INDEX_HI`. The instructions access the word at the
lane's saved index XOR 1. The low half uses the existing indexed
operations at the saved index. Use an even base for consecutive
low-then-high storage. Each half is a separate instruction.

```sh
cd "$QWENGPU_DIR"
export QWENGPU_PYTHON="$QWENPUTER_DIR/.venv/bin/python"
fixture=eval/frozen/resident_shared32_gate_v1/cooperative_shared32.json
"$QWENGPU_PYTHON" -m qwengpu.run_resident "$fixture" \
  --resident-shared32 --device mps --trace
bash scripts/qualify_resident_add32.sh --resident-shared32 --device mps \
  --out results/resident_shared32_full_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
model=/path/to/Qwen3.5-9B-qwengpu-resident-shared32
"$QWENGPU_PYTHON" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resident-shared32 --out "$model"
bash scripts/qualify_resident_add32.sh --resident-shared32 "$model" --device cuda \
  --out results/resident_shared32_full_9b_cuda_fla.json
python3 scripts/check_resident_shared32.py
```

The handoff fixture includes two lanes that target the same pair
with different results. The lowest active lane wins both stores.
The consumer reads after a barrier. The checker compares the final
registers, all 128 shared words, both store grants, and all 23
inherited ADD32 case records.

To replay the retained-memory launchers on the same shared32 checkpoint:

```sh
matmul=eval/frozen/matmul16_gate_v1/dense_seeded.json
"$QWENGPU_PYTHON" -m qwengpu.run_matmul16 "$matmul" \
  --resident-shared32 --retain-c --device mps \
  --out results/resident_shared32_matmul16_retained_mps.json
"$QWENGPU_PYTHON" -m qwengpu.run_matmul16 "$matmul" \
  --model "$model" --retain-c --device cuda \
  --out results/resident_shared32_matmul16_retained_9b_cuda_fla.json
"$QWENGPU_PYTHON" -m qwengpu.eval_cached_tile --model "$model" --device cuda \
  --out results/resident_shared32_cached_tile_9b_cuda_fla.json
"$QWENGPU_PYTHON" -m qwengpu.eval_cache_relaunch "$model" --device cuda \
  --out results/resident_shared32_cache_relaunch_9b_cuda_fla.json
python3 scripts/check_matmul16.py results/resident_shared32_matmul16_retained_*.json
python3 scripts/check_resident_shared32.py --cache-replays
```

## Add two full-width shared-memory operands

The addmem32 layout first loads the left word pair with the shared32
instructions. `ADD_INDEX_LO` adds the low right word and carries
into the high register. `ADD_INDEX_HI` adds the high right word.
Both use the saved index XOR the same even immediate to locate the
right pair. The high instruction toggles bit zero of that address.

```sh
cd "$QWENGPU_DIR"
export QWENGPU_PYTHON="$QWENPUTER_DIR/.venv/bin/python"
matmul=eval/frozen/matmul16_gate_v1/dense_seeded.json
bash scripts/qualify_resident_add32.sh --resident-addmem32 --device mps \
  --out results/resident_addmem32_full_mps.json
"$QWENGPU_PYTHON" -m qwengpu.run_matmul16 "$matmul" \
  --resident-addmem32 --retain-c --device mps \
  --out results/resident_addmem32_matmul16_retained_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
model=/path/to/Qwen3.5-9B-qwengpu-resident-addmem32
"$QWENGPU_PYTHON" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resident-addmem32 --out "$model"
bash scripts/qualify_resident_add32.sh --resident-addmem32 "$model" --device cuda \
  --out results/resident_addmem32_full_9b_cuda_fla.json
"$QWENGPU_PYTHON" -m qwengpu.run_matmul16 "$matmul" \
  --model "$model" --retain-c --device cuda \
  --out results/resident_addmem32_matmul16_retained_9b_cuda_fla.json
python3 scripts/check_resident_addmem32.py
python3 scripts/check_matmul16.py results/resident_addmem32_matmul16_retained_*.json
```

## Verify text and cache handoff on addmem32

Use the same addmem32 merge that passed the arithmetic suite.
The fresh-cache check compares the pinned base and merge on the
24 frozen prompts. The other checks reset the pretrained layers'
text history while retaining the GPU layers in the cache.

```sh
cd "$QWENGPU_DIR"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export QWENGPU_PYTHON="$QWENPUTER_DIR/.venv/bin/python"
base=/path/to/Qwen3.5-9B
model=/path/to/Qwen3.5-9B-qwengpu-resident-addmem32
"$QWENGPU_PYTHON" -m qwengpu.textcheck_9b "$base" "$model" --device cuda \
  --corpus eval/frozen/text_add32_gate_v1/prompts.json --max-new-tokens 16 \
  --out results/resident_addmem32_text_9b_cuda_fla.json
"$QWENGPU_PYTHON" -m qwengpu.eval_cache_relaunch "$model" --device cuda \
  --out results/resident_addmem32_cache_relaunch_9b_cuda_fla.json
"$QWENGPU_PYTHON" -m qwengpu.eval_text_handoff "$model" --device cuda \
  --reset-text --out results/resident_addmem32_handoff_reset_9b_cuda_fla.json
```

For the same-device control, build and measure the earlier ADD32
checkpoint from the same pinned base on the same CUDA host.
Reserve space for the control weights or finish copying the
addmem32 evidence before reusing its scratch storage.

```sh
control=/path/to/Qwen3.5-9B-qwengpu-resident-add32
"$QWENGPU_PYTHON" -m qwengpu.merge9b "$base" --resident-add32 --out "$control"
"$QWENGPU_PYTHON" -m qwengpu.textcheck_9b "$base" "$control" --device cuda \
  --corpus eval/frozen/text_add32_gate_v1/prompts.json --max-new-tokens 16 \
  --out results/resident_add32_text_9b_cuda_h100_control.json
python3 scripts/check_addmem32_text.py
```

## Relaunch after full-width arithmetic

The producer is the frozen addmem32 sum. The second kernel has
two warps that load zeroed destination pairs and add the retained
source pairs into them. Both halves are stored at the destination.
The checker derives the source words from scalar sums, verifies
all 128 final words, and compares with the fresh-cache control.

```sh
cd "$QWENGPU_DIR"
export QWENGPU_PYTHON="$QWENPUTER_DIR/.venv/bin/python"
"$QWENGPU_PYTHON" -m qwengpu.eval_cache_relaunch --addmem32 --device mps \
  --out results/resident_addmem32_arithmetic_relaunch_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
model=/path/to/Qwen3.5-9B-qwengpu-resident-addmem32
"$QWENGPU_PYTHON" -m qwengpu.eval_cache_relaunch "$model" --addmem32 --device cuda \
  --out results/resident_addmem32_arithmetic_relaunch_9b_cuda_fla.json
"$QWENGPU_PYTHON" -m qwengpu.eval_cache_relaunch "$model" --device cuda \
  --out results/resident_addmem32_cache_relaunch_legacy_replay_9b_cuda_fla.json
python3 scripts/check_addmem32_relaunch.py
```

The MPS run relaunches the consumer immediately. The merged run
generates text between the launches, with a fresh text context
and a retained GPU cache. It checks text against a fresh-cache
control and verifies the GPU cache rows before and after text.

## Branch on all 32 register bits

The branch32 layout uses the existing `IF_BIT` instruction and
extends its accepted argument from 0 through 15 to 0 through 31.
The fixture adds two indexed operands, branches on bit 31, then
reconverges and shares the result with a second warp. Every active
lane has the same low half at the branch.

```sh
cd "$QWENGPU_DIR"
export QWENGPU_PYTHON="$QWENPUTER_DIR/.venv/bin/python"
matmul=eval/frozen/matmul16_gate_v1/dense_seeded.json
bash scripts/qualify_resident_add32.sh --resident-branch32 --device mps \
  --out results/resident_branch32_full_mps.json
"$QWENGPU_PYTHON" -m qwengpu.run_matmul16 "$matmul" \
  --resident-branch32 --retain-c --device mps \
  --out results/resident_branch32_matmul16_retained_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
model=/path/to/Qwen3.5-9B-qwengpu-resident-branch32
"$QWENGPU_PYTHON" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resident-branch32 --out "$model"
bash scripts/qualify_resident_add32.sh --resident-branch32 "$model" --device cuda \
  --out results/resident_branch32_full_9b_cuda_fla.json
"$QWENGPU_PYTHON" -m qwengpu.run_matmul16 "$matmul" \
  --model "$model" --retain-c --device cuda \
  --out results/resident_branch32_matmul16_retained_9b_cuda_fla.json
python3 scripts/check_resident_branch32.py
python3 scripts/check_matmul16.py results/resident_branch32_matmul16_retained_*.json
```

## Loop on all 32 register bits

The loop32 layout keeps the original low-half loop encoding and
adds encodings for bits 16 through 31. The frozen kernel reads
bit 16 and increments by one until carry clears that bit.
The checker reconstructs the trip counts, masks, and final
shared words independently of the interpreter.

```sh
cd "$QWENGPU_DIR"
export QWENGPU_PYTHON="$QWENPUTER_DIR/.venv/bin/python"
matmul=eval/frozen/matmul16_gate_v1/dense_seeded.json
bash scripts/qualify_resident_add32.sh --resident-loop32 --device mps \
  --out results/resident_loop32_full_mps.json
"$QWENGPU_PYTHON" -m qwengpu.run_matmul16 "$matmul" \
  --resident-loop32 --retain-c --device mps \
  --out results/resident_loop32_matmul16_retained_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
model=/path/to/Qwen3.5-9B-qwengpu-resident-loop32
"$QWENGPU_PYTHON" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resident-loop32 --out "$model"
bash scripts/qualify_resident_add32.sh --resident-loop32 "$model" --device cuda \
  --out results/resident_loop32_full_9b_cuda_fla.json
"$QWENGPU_PYTHON" -m qwengpu.run_matmul16 "$matmul" \
  --model "$model" --retain-c --device cuda \
  --out results/resident_loop32_matmul16_retained_9b_cuda_fla.json
python3 scripts/check_resident_loop32.py
python3 scripts/check_matmul16.py results/resident_loop32_matmul16_retained_*.json
```

## Preload two K-slice operand banks

The tilebanks layout adds a read-only operand bank for tile MACs.
The evaluator uploads both A/B pairs and initial C before the
first launch. The second launch uses the same cache, selects the
second bank, and uploads zero initial memory tokens. Indexed
loads and stores continue to use the original shared memory.

```sh
cd "$QWENGPU_DIR"
export QWENGPU_PYTHON="$QWENPUTER_DIR/.venv/bin/python"
"$QWENGPU_PYTHON" -m qwengpu.eval_cached_tile --preload-operands --device mps \
  --out results/resident_tilebanks_cached_tile_mps.json
bash scripts/qualify_resident_add32.sh --resident-tilebanks --device mps \
  --out results/resident_tilebanks_full_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
model=/path/to/Qwen3.5-9B-qwengpu-resident-tilebanks
"$QWENGPU_PYTHON" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resident-tilebanks --out "$model"
"$QWENGPU_PYTHON" -m qwengpu.eval_cached_tile --model "$model" \
  --preload-operands --device cuda \
  --out results/resident_tilebanks_cached_tile_9b_cuda_fla.json
bash scripts/qualify_resident_add32.sh --resident-tilebanks "$model" --device cuda \
  --out results/resident_tilebanks_full_9b_cuda_fla.json
python3 scripts/check_tilebanks.py --full
```

## Let the model transition between preloaded K slices

The tilepipe layout starts with both operand banks and one C image.
The scheduler emits a bank-transition ticket after the final row
pair of the first slice. The runner relays that ticket unchanged.
The model selects bank one, resets both warp PCs, and restarts
at row zero with the same ROM and cache.

```sh
cd "$QWENGPU_DIR"
export QWENGPU_PYTHON="$QWENPUTER_DIR/.venv/bin/python"
"$QWENGPU_PYTHON" -m qwengpu.eval_tilepipe --device mps \
  --out results/resident_tilepipe_mps.json
bash scripts/qualify_resident_add32.sh --resident-tilepipe --device mps \
  --out results/resident_tilepipe_full_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
model=/path/to/Qwen3.5-9B-qwengpu-resident-tilepipe
"$QWENGPU_PYTHON" -m qwengpu.merge9b /path/to/Qwen3.5-9B \
  --resident-tilepipe --out "$model"
"$QWENGPU_PYTHON" -m qwengpu.eval_tilepipe --model "$model" --device cuda \
  --out results/resident_tilepipe_9b_cuda_fla.json
bash scripts/qualify_resident_add32.sh --resident-tilepipe "$model" --device cuda \
  --out results/resident_tilepipe_full_9b_cuda_fla.json
python3 scripts/check_tilepipe.py --full
```

The evaluator compares each slice to a separate execution and
a scalar matrix product. Its forward audit records the initial
memory and grid setup, the relayed transition, and the C stores.

## Run the full 16×16 multiply with scheduled K slices

The public `--schedule-k` mode starts one tilepipe launch per output
tile. Both K slices run inside that launch. It reports the two
16×16 pass matrices, the four model-generated transitions, and all
grid, operand and C tokens sent to the model.

```sh
cd "$QWENGPU_DIR"
export QWENGPU_PYTHON="$QWENPUTER_DIR/.venv/bin/python"
fixture=eval/frozen/matmul16_gate_v1/dense_seeded.json
"$QWENGPU_PYTHON" -m qwengpu.run_matmul16 "$fixture" --schedule-k --device mps \
  --out results/resident_tilepipe_matmul16_mps.json
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
base=/path/to/Qwen3.5-9B
model=/path/to/Qwen3.5-9B-qwengpu-resident-tilepipe
"$QWENGPU_PYTHON" -m qwengpu.run_matmul16 "$fixture" --schedule-k \
  --model "$model" --device cuda --out results/resident_tilepipe_matmul16_9b_cuda_fla.json
"$QWENGPU_PYTHON" -m qwengpu.textcheck_9b "$base" "$model" --device cuda \
  --corpus eval/frozen/text_add32_gate_v1/prompts.json --max-new-tokens 16 \
  --out results/resident_tilepipe_text_9b_cuda_fla.json
"$QWENGPU_PYTHON" -m qwengpu.eval_text_handoff "$model" --device cuda \
  --reset-text --out results/resident_tilepipe_handoff_reset_9b_cuda_fla.json
"$QWENGPU_PYTHON" -m qwengpu.eval_cache_relaunch "$model" --addmem32 --device cuda \
  --out results/resident_tilepipe_arithmetic_relaunch_9b_cuda_fla.json
python3 scripts/check_tilepipe_matmul16.py --text
```

## Verify model-selected workgroups in one cache

Run the standalone grid check with the pinned compiler environment above.

```sh
cd "$QWENGPU_DIR"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_vector_grid --device mps \
  --out /tmp/qwengpu-vector-grid-mps.json
```

The report compares output tokens and physical recurrent memory against
scalar results for six lengths. The 13-word case should emit both group
tickets and retain all three trailing guard words. It receives one
upload, then relayed tickets, fixed steps, and read requests.

## Assemble and check a resident program

The tilepipe v1 assembler accepts two sections named `.warp 0` and
`.warp 1`. Each line contains an opcode and an optional integer
operand. Use `#` for comments. Structured `IF_BIT` and `WHILE_BIT`
blocks use the same validation as JSON kernels.

```sh
cd "$QWENGPU_DIR"
export QWENGPU_PYTHON="$QWENPUTER_DIR/.venv/bin/python"
"$QWENGPU_PYTHON" -m qwengpu.assemble examples/cooperative_pair.qga \
  --data examples/cooperative_pair_data.json --out /tmp/qwengpu-cooperative-pair.json
"$QWENGPU_PYTHON" -m qwengpu.run_resident /tmp/qwengpu-cooperative-pair.json \
  --resident-tilepipe --device mps --check-reference --trace \
  --out /tmp/qwengpu-cooperative-pair-mps.json
```

The second warp should finish with
`[4, 65541, 8, 65545, 12, 65549, 16, 65553]`. The reference check
compares every instruction, scheduling decision, and final state field.

## Verify the tiled workgroup transition

```sh
cd "$QWENGPU_DIR"
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.eval_tilegrid --device mps \
  --out /tmp/qwengpu-tilegrid-mps.json
```

This runs one row pair per workgroup. Add `--row-pairs 4` to run
all rows. The evaluator checks each pass independently and audits
all model calls for cache identity, one initial descriptor, two
bank switches, and one workgroup switch.

Add `--separate-outputs` to preload two distinct initial C tiles. The
model selects the second C tile after the workgroup transition and
emits its store addresses. The evaluator checks both physical output
regions and verifies that the second group leaves the first intact.

Add `--right-panel` to use different B inputs as well. With
`--row-pairs 4 --right-panel`, this runs an 8×16 output panel with
an inner dimension of 16. The evaluator checks the combined readout
against scalar multiplication.

Run the four-workgroup 16×16 case with one upload and one cache.
Use a custom 16×16 input by passing `--input` with the same JSON schema
as the dense fixture.

```sh
fixture=eval/frozen/matmul16_gate_v1/dense_seeded.json
"$QWENGPU_PYTHON" -m qwengpu.eval_tilegrid --grid-2x2 --row-pairs 4 \
  --input "$fixture" --device mps --out /tmp/qwengpu-tilegrid-2x2-mps.json
base=/path/to/Qwen3.5-9B
model=/path/to/Qwen3.5-9B-qwengpu-resident-tilegrid
"$QWENGPU_PYTHON" -m qwengpu.merge9b "$base" --resident-tilegrid --out "$model"
"$QWENGPU_PYTHON" -m qwengpu.eval_tilegrid --grid-2x2 --row-pairs 4 \
  --input "$fixture" --model "$model" --device cuda \
  --out /tmp/qwengpu-tilegrid-2x2-9b.json
```
