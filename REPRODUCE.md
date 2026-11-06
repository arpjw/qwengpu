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
