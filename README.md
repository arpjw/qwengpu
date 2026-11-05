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
programs pass on stock MPS and optimized CUDA, including all 51 intermediate
register/scratch
snapshots. It uses 51 layers and 58 heads by unrolling 16 instruction stages
in depth. This removes lane feedback for bounded straight-line programs;
cyclic register reuse and loops remain open. Run a supported kernel with:

```sh
python3 -m qwengpu.eval_writeback --device mps \
  --kernel eval/frozen/writeback_gate_v1/overwrite.json --out result.json
```

The next gates are bounded divergent control flow, larger memory, and broader
program suites. The reference interpreter checks model traces and is not used
to compute model outputs. The original host-dispatched divergent circuit and
the two straight-line ROM variants have separate protocols and evidence.
See [SCOPE.md](SCOPE.md), [VALIDATION.md](VALIDATION.md), and
[REPRODUCE.md](REPRODUCE.md) for limits, evidence, and pinned commands.
[CHECKPOINT.md](CHECKPOINT.md) records the sprint and continuation.

Model weights are not stored in this repository. The Python code is MIT
licensed; Qwen3.5 weights remain under their own license.
