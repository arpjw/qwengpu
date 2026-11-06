# QwenGPU

QwenGPU is an experiment in compiling a small SIMT processor into stock
Qwen3.5 layers. It builds on the [QwenCPU](https://github.com/arpjw/qwencpu)
experiment and uses [Qwenputer](https://github.com/divagr18/qwenputer)'s
generic circuit compiler as a backend. The CPU, GPU, and compiler names here
refer to simulated machines inside model state, not physical hardware.
The processor weights are constructed by the compiler; this experiment does
not fine-tune the pretrained model.

The repository contains an independent eight-lane reference machine.
Each lane has an unsigned byte accumulator. The warp shares an instruction
stream and active mask. Structured `IF_LT`/`ELSE`/`ENDIF` executes divergent
paths under disjoint masks and then reconverges. Lane-indexed scratch memory,
addition, XOR, and reduction are specified and tested.

The compiled circuit performs eight masked byte additions or XORs in one
Qwen3.5 forward pass. Its inputs, lane mask, results, and lane-banked scratch memory
live in Gated DeltaNet state. Scratch has four byte slots per lane.
The model generates both per-lane result tokens and masks from `IF_LT`, `ELSE`,
and `ENDIF` operations. Result tokens write each lane back to recurrent state,
so later instructions use prior results. One-level divergent programs run
their then and else paths under disjoint masks and reconverge. The host supplies
raw instructions and read requests; it does not calculate the model's results.

| Check | Observed result |
| --- | --- |
| Standalone stock Qwen3.5 on Apple MPS | 8/8 adds, 8/8 generated-mask adds, 4/4 sequences, 4/4 divergent programs, 4/4 scratch programs, 4/4 ADD/XOR programs, and 4/4 mixed programs exact |
| Standalone stock Qwen3.5 on CUDA with optimized DeltaNet | The same seven suites exact; causal convolution used the reference kernel |
| Pretrained Qwen3.5-9B merge on CUDA | 12/12 divergent, 12/12 scratch, 12/12 ADD/XOR, and 12/12 mixed programs exact, including generated tokens and scratch-state inspection |
| Exploratory 9B mixed-program stress check | 100/100 programs exact, including 96 seeded cases, generated tokens, and scratch-state inspection |
| Exploratory MPS mixed-program stress check | The same 100 programs were exact, with identical generated tokens to the 9B CUDA run |
| Exploratory varied-program check on MPS | 32/32 programs across four shapes had exact generated tokens, scratch state, and final lane probes |
| Separate 16-slot ROM feasibility spike on MPS | One stock model fetched three different uploaded instruction sequences, generated PC updates, and retained exact ROM state |
| Exploratory ROM stress check on MPS | The same model fetched 100 16-slot sequences, including 97 seeded ones; 3,100 generated tokens and all recurrent ROM and PC snapshots matched |
| Example JSON kernel on MPS | A host-dispatched mixed program and final lane probe matched the independent interpreter |
| Frozen next-gate kernels on MPS | Four straight-line kernels passed on the current host-dispatched circuit and are ready for ROM integration checks |
| Integrated 16-slot ROM and ALU on MPS and optimized CUDA | Four frozen programs exact, including every generated PC/halt and lane token plus final recurrent ROM, PC, registers, and scratch |
| Integrated ROM inside pretrained Qwen3.5-9B on CUDA | The same four frozen programs passed with exact generated tokens and final recurrent machine state |
| Fixed text check on ROM 9B merge | 6/6 same next token; 5/6 identical first four greedy tokens; maximum next-logit difference 0.2734375 |
| Fixed text check on earlier merged 9B | 6/6 prompts kept the same next token; 5/6 kept the same first four greedy tokens |

These are small, fixed execution and integration checks. The earlier
host-dispatched circuit uses 27
Gated DeltaNet heads and 12 circuit layers. The 9B merge adds an exit layer,
preserves the source model's untied text output head, and was built with 4,864
hidden dimensions. [Validation details](VALIDATION.md) record the evidence and
the one text continuation that changed.

```sh
python3 -m unittest discover -s tests -v
```

The compiler checks require Qwenputer's `experimental/riscv` package on
`PYTHONPATH`. The standalone model check additionally requires PyTorch and
Transformers:

```sh
python3 -m qwengpu.eval_add --device mps --out results/standalone_mps.json
python3 -m qwengpu.eval_predicate --device mps --out results/predicate_mps.json
python3 -m qwengpu.eval_sequence --device mps --out results/sequence_mps.json
python3 -m qwengpu.eval_branch --device mps --out results/branch_mps.json
python3 -m qwengpu.eval_memory --device mps --out results/memory_mps.json
python3 -m qwengpu.eval_alu --device mps --out results/alu_mps.json
python3 -m qwengpu.eval_program --device mps --out results/program_mps.json
python3 -m qwengpu.eval_rom_spike --device mps --out results/rom_spike_mps.json
python3 -m qwengpu.run_kernel examples/mixed_kernel.json --device mps \
  --out results/example_kernel_mps.json
```

`run_kernel` accepts an initial eight-byte lane vector and up to 64
instructions in JSON. It validates the supported one-level branch and
four-slot scratch subset. The example ends with lane values
`[1, 2, 3, 4, 5, 6, 7, 8]`. Pass `--model` with a rebuilt 9B checkpoint to
use the pretrained merge. That CLI path has not been separately run; the 9B
checks above used the evaluator directly.
[`examples/no_else_kernel.json`](examples/no_else_kernel.json) shows a branch
that reconverges without an `ELSE`.
[`examples/masked_overwrite_kernel.json`](examples/masked_overwrite_kernel.json)
shows masked writes preserving inactive lanes in the same scratch slot.
[`examples/halt_only_kernel.json`](examples/halt_only_kernel.json) checks the
zero-work boundary.

A separate integrated ROM circuit now fetches and executes uploaded straight-line
programs from model state. Run `python3 -m qwengpu.eval_compiled_rom --device mps
--out results/compiled_rom_mps.json` for its four frozen fixtures. It uses 13
circuit layers and 19 heads; its 9B merge adds an exit layer and uses hidden
size 4,608. It passed the same frozen fixtures on CUDA. It retains fixed lane
read/feedback after every non-halt
TICK. See [ROM_DESIGN.md](ROM_DESIGN.md).

A separate [bounded writeback circuit](WRITEBACK_DESIGN.md) now runs with
only TICK and generated-state feedback after upload. Its seven frozen
programs pass on stock MPS, optimized CUDA, and the pretrained 9B merge,
including all 51 intermediate
register/scratch
snapshots. It uses 51 layers and 58 heads by unrolling 16 instruction stages
in depth. This removes lane feedback for bounded straight-line programs;
This earlier variant has no loops; the control variant below adds bounded loops. The 9B checkpoint adds an exit
layer and contains about 52.1 GB of weights. Its six-prompt text check kept
6/6 next tokens and 5/6 four-token continuations. Run a supported kernel with:

```sh
python3 -m qwengpu.eval_writeback --device mps \
  --kernel eval/frozen/writeback_gate_v1/overwrite.json --out result.json
```

The [control circuit](CONTROL_DESIGN.md) now supports nested IF/ELSE/ENDIF
and data-dependent WHILE_LT/ENDWHILE loops. It carries a separate execution
step and PC, so backward jumps use updated register state. Both branch and
loop stacks support depth two, including simultaneously. Fourteen fixed cases
pass on MPS, optimized CUDA, and inside pretrained Qwen3.5-9B: 13 halt and one deliberately exhausts the
16-step budget. Every instruction's machine snapshot is checked. It uses
69 circuit layers and zero lane feedback. Its 9B merge adds an exit layer,
contains about 61.95 GB of weights, and kept 6/6 next tokens and 5/6 short
continuations in the fixed text check. Try a loop kernel with:

```sh
python3 -m qwengpu.eval_control --device mps \
  --kernel eval/frozen/control_gate_v1/if_inside_loop.json --out control.json
```

The resume variant repeats 16-step windows on one fixed checkpoint. It
exports live warp state as tokens and reloads them into a fresh cache with
the original ROM. The host relays the tokens unchanged. Merged Qwen3.5-9B
ran a divergent scratch loop for 1,082 instructions across 68 windows and
matched the interpreter at every step.

The integer ISA variant adds a second byte register per lane, an indexed
gather from the 32-byte scratch space, and modulo-256 multiplication.
Its default kernel fills scratch, crosses a window boundary, and then
gathers and multiplies under a divergent mask. The complete 25-instruction
trace and handoff match on MPS and merged Qwen3.5-9B CUDA. Run it locally
with `python3 -m qwengpu.eval_resume --isa --device mps --out isa.json`.

The workgroup variant runs two eight-lane warps with independent registers
and control stacks over a shared 32-byte scratch space. It adds indexed
scatter, masked byte reduction, and barriers. A round-robin host launcher
relays model-generated state tokens after each warp instruction. Cooperative
reduction, interleaved loops, and an expected barrier deadlock match on
MPS and merged Qwen3.5-9B CUDA. Run the suite with
`python3 -m qwengpu.eval_workgroup --device mps --out workgroup.json`.

The tile variant moves warp selection and barrier release into the model.
Two warps multiply 4×4 byte matrices in two output tiles with 16-bit
accumulators and 64 shared bytes. The host relays generated scheduling and
state tokens. The matrix multiply, wide addition, and deadlock fixtures
match on MPS and merged Qwen3.5-9B. Run the demo with
`python3 -m qwengpu.eval_tile --device mps --out tile.json`.

The 8×8 tile variant expands shared memory to 256 bytes. The model
advances four row pairs, reloads C before accumulation, and emits
address-tagged writes for every output word. The seeded 8×8 multiply
and deliberate deadlock match on MPS and merged Qwen3.5-9B. Run them with
`python3 -m qwengpu.eval_tile8 --device mps --out tile8.json`.

The resident 8×8 variant uploads A, B, and C once and reuses one inference
cache across the whole launch. The host feeds the model's tagged writes
back into the same cache. On the dense fixture this reduced input tokens
from 73,434 to 14,694 while preserving the full trace. The multiply, store-and-reload, and
deadlock traces match on MPS and merged Qwen3.5-9B. Run the suite with
`python3 -m qwengpu.eval_tile8 --resident --device mps --out resident.json`.

The indexed resident variant gives each lane a 7-bit address register
and access to all 128 shared 16-bit words. The model gathers and
scatters from the same memory used for tiled multiply. The lowest lane
number wins same-instruction write collisions. The full address sweep
and cross-warp scatter and gather match on MPS and merged Qwen3.5-9B.
Run the suite with
`python3 -m qwengpu.eval_tile8 --indexed --device mps --out indexed.json`.

The masked resident variant lets each warp select active lanes from a
bit of each accumulator and reactivate all lanes explicitly. Inactive
lanes preserve registers and skip writes. Scatter priority follows the
lowest active lane targeting each word. All six fixtures match on MPS
and merged Qwen3.5-9B. Run the suite with
`python3 -m qwengpu.eval_tile8 --masked --device mps --out masked.json`.

The divergent resident variant adds two levels of `IF_BIT`, optional
`ELSE`, and `ENDIF` per warp. Saved predicates determine both paths,
and reconvergence restores the saved parent mask. All eight fixtures
match on MPS and merged Qwen3.5-9B. Run the suite with
`python3 -m qwengpu.eval_tile8 --divergent --device mps --out divergent.json`.

The looping resident variant adds model-selected back edges and exits.
`WHILE_BIT` uses a per-lane accumulator bit, and `ENDWHILE` restores the
parent mask before retesting. A single checkpoint supports two levels
each of branches and loops, 16-bit accumulation, and indexed shared
memory. All eleven fixtures match on MPS and merged Qwen3.5-9B. Run the
suite with
`python3 -m qwengpu.eval_tile8 --looping --device mps --out looping.json`.

Launch a two-warp kernel with the resident runner. The cooperative
example publishes tiled multiply rows and gathers them into a divergent
loop on the peer warp.

```sh
python3 -m qwengpu.run_resident eval/frozen/loop_gate_v1/cooperative_matmul.json
```

Add `--model /path/to/Qwen3.5-9B-qwengpu-looping --device cuda` for the
merged checkpoint. Pass `--resident-rom` to the standalone launcher or
use a merge built with `merge9b --resident-rom` to upload each warp program
once per launch. The model fetches instructions by its scheduled warp
and PC. Pass `--resident-state` or use a merge built with
`merge9b --resident-state` to keep each warp's accumulators, indices,
mask, and stacks in its own bank in the same cache. With
`--resident-delta`, or a merge built with `merge9b --resident-delta`,
the model emits a state update only when a nibble changes. The
`--resident-plan` checkpoint also tells the launcher which state
positions to read. With `--resident-io`, the model also selects shared
stores, and the launcher skips the follow-up when both sets are empty.
On a resident I/O checkpoint, choose `--transport fixed`,
`--transport state-plan`, or `--transport state-store-plan` to compare
the protocols on the same weights. The launcher reports the resulting
matrix, shared words, instruction count, and launch outcome as JSON.

Use `--resident-grid` or a merge built with `merge9b --resident-grid`
to set `"first_row_pair"` and `"row_pairs"` in the launch JSON.
The zero-based start defaults to 0, and the count defaults to the
remaining pairs. A window must fit inside the four row pairs. The model
schedules two rows per pair and emits completion after both warps halt
on the last pair. The default tiled kernel preserves the initial C values
outside the window. Explicit indexed stores can address any shared word.
The grid checkpoint supports all three transport modes.

Run the 16×16 modular matrix multiply on the same grid checkpoint:

```sh
python3 -m qwengpu.run_matmul16 eval/frozen/matmul16_gate_v1/dense_seeded.json
```

The runner packs four output tiles and carries each tile through two
accumulation passes. It accepts 16×16 byte matrices A and B and an
optional 16×16 unsigned 16-bit C. It returns both intermediate and final
matrices. Add `--resident-add32` to use the standalone checkpoint with
ADD32I. For CUDA, pass the path to either merged checkpoint with
`--model` and set `--device cuda`. The multiply accumulates modulo
65536 on both layouts.

With the resident ADD32 checkpoint, add `--retain-c` to run both K
slices for each output tile in the same inference cache. Each tile
starts with one upload of its initial C and keeps the partial result
in model state between the slices. The second launch refreshes A
and B. The 16×16 multiply uses four caches in total.

```sh
python3 -m qwengpu.run_matmul16 eval/frozen/matmul16_gate_v1/dense_seeded.json \
  --resident-add32 --retain-c --device mps
```

The isolated ADD32 prototype adds two unsigned 32-bit words per lane.
The model propagates carry between the 16-bit halves and keeps the
left operand in inactive lanes. Run the frozen checks with
`python3 -m qwengpu.eval_add32 --device mps --out results/add32_mps.json`.

The resident wide layout adds a high 16-bit register half to every lane.
Use `run_resident --resident-wide` or a merge built with
`merge9b --resident-wide`. The optional `initial_registers` field contains
two arrays of eight unsigned 32-bit words. Existing tile instructions
operate on the low half and preserve the high half. The launcher returns
both halves combined in `warp_registers`.

The resident ADD32 layout adds the `ADD32I` instruction. It accepts an
immediate from 0 through 255, adds it to each active lane's 32-bit
register, and wraps modulo 2³². The same checkpoint runs tiled multiply,
indexed shared-memory access, barriers, and divergent control.

```sh
python3 -m qwengpu.run_resident --resident-add32 \
  eval/frozen/resident_add32_gate_v1/cooperative_add32.json
```

The resident shared32 layout adds `LOAD_INDEX_HI` and
`STORE_INDEX_HI`. They use the same lane mask and scatter priority
as the indexed operations. Pair a low word at the saved index
with its high word at that index XOR 1. Use a barrier between a
producer's two stores and a consumer's loads.

```sh
python3 -m qwengpu.run_resident --resident-shared32 \
  eval/frozen/resident_shared32_gate_v1/cooperative_shared32.json
```

The resident addmem32 layout adds two arbitrary unsigned 32-bit
values from shared memory. Load the left pair with the shared32
instructions. `ADD_INDEX_LO` and `ADD_INDEX_HI` read the right pair
at the saved index XOR an even immediate from 0 through 126.
The low step propagates carry into the high register. The high
step completes the sum modulo 2³². Use the same mask and immediate
for both steps of a vector sum.

```sh
python3 -m qwengpu.run_resident --resident-addmem32 \
  eval/frozen/resident_addmem32_gate_v1/cooperative_addmem32.json
```

The resident branch32 layout extends `IF_BIT` to all 32 register
bits. It uses the existing mask stack and reconvergence rules.
`IF_BIT 31` can select lanes from the sign bit of a full-width sum.

```sh
python3 -m qwengpu.run_resident --resident-branch32 \
  eval/frozen/resident_branch32_gate_v1/cooperative_branch32.json
```

The resident loop32 layout extends `WHILE_BIT` to all 32
register bits. The model repeats the body for lanes whose bit
is set, restores the parent mask at `ENDWHILE`, and checks the
updated register on the next trip.

```sh
python3 -m qwengpu.run_resident --resident-loop32 \
  eval/frozen/resident_loop32_gate_v1/cooperative_loop32.json
```

The tilebanks layout holds both K-slice operand banks and partial
C in the same cache. The host selects the bank at launch. Both
banks are uploaded before the first slice, and the second slice
reuses all three memory images.

```sh
python3 -m qwengpu.eval_cached_tile --preload-operands --device mps \
  --out results/resident_tilebanks_cached_tile_mps.json
```

The tilepipe launcher runs both K slices for each 8×8 output tile
with one model-scheduled launch. The model advances to the second
operand bank and reuses C, the warp state, and the program.

```sh
python3 -m qwengpu.run_matmul16 eval/frozen/matmul16_gate_v1/dense_seeded.json \
  --schedule-k --device mps
```

The independent interpreter checks traces; it does not compute model outputs.
See [SCOPE.md](SCOPE.md), [VALIDATION.md](VALIDATION.md), and
[REPRODUCE.md](REPRODUCE.md) for limits, evidence, and pinned commands.
[CHECKPOINT.md](CHECKPOINT.md) records the sprint and continuations.

Model weights are not stored in this repository. The Python code is MIT
licensed; Qwen3.5 weights remain under their own license.
