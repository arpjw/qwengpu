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

The independent interpreter checks traces; it does not compute model outputs.
See [SCOPE.md](SCOPE.md), [VALIDATION.md](VALIDATION.md), and
[REPRODUCE.md](REPRODUCE.md) for limits, evidence, and pinned commands.
[CHECKPOINT.md](CHECKPOINT.md) records the sprint and continuations.

Model weights are not stored in this repository. The Python code is MIT
licensed; Qwen3.5 weights remain under their own license.
