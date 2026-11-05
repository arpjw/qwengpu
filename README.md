# QwenGPU

QwenGPU is an experiment in compiling a small SIMT processor into stock
Qwen3.5 layers. It builds on the [QwenCPU](https://github.com/arpjw/qwencpu)
experiment and uses [Qwenputer](https://github.com/divagr18/qwenputer)'s
generic circuit compiler as a backend. The CPU, GPU, and compiler names here
refer to simulated machines inside model state, not physical hardware.

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
| Exploratory 9B mixed-program stress check | 100/100 seeded programs exact, including generated tokens and scratch-state inspection |
| Separate 16-slot ROM feasibility spike on MPS | One stock model fetched three different uploaded instruction sequences, generated PC updates, and retained exact ROM state |
| Example JSON kernel on MPS | A host-dispatched mixed program and final lane probe matched the independent interpreter |
| Fixed text check on merged 9B | 6/6 prompts kept the same next token; 5/6 kept the same first four greedy tokens |

These are small, fixed execution and integration checks. The circuit uses 27
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
use the pretrained merge.
[`examples/no_else_kernel.json`](examples/no_else_kernel.json) shows a branch
that reconverges without an `ELSE`.
[`examples/masked_overwrite_kernel.json`](examples/masked_overwrite_kernel.json)
shows masked writes preserving inactive lanes in the same scratch slot.

The next gates are program memory and instruction fetch, internal register
writeback, nested divergence, larger memory, and larger held-out
program suites. The reference interpreter checks model traces and is not used
to compute the model's outputs. The current one-level branch trace is driven by
host-supplied raw instruction tokens, so this is a SIMT circuit prototype and
not yet a self-contained programmable GPU. See [SCOPE.md](SCOPE.md) for the
next gates, [ROM_DESIGN.md](ROM_DESIGN.md) for the isolated fetch experiment,
and [REPRODUCE.md](REPRODUCE.md) for pinned inputs and commands.

Model weights are not stored in this repository. The Python code is MIT
licensed; Qwen3.5 weights remain under their own license.
