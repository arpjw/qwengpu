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

The compiled circuit performs eight masked byte additions in one Qwen3.5
forward pass. Its inputs, lane mask, and results live in Gated DeltaNet state.
The model generates both per-lane result tokens and masks from `IF_LT`, `ELSE`,
and `ENDIF` operations. Result tokens write each lane back to recurrent state,
so later instructions use prior results. One-level divergent programs run
their then and else paths under disjoint masks and reconverge. The host supplies
raw instructions and read requests; it does not calculate the model's results.

| Check | Observed result |
| --- | --- |
| Standalone stock Qwen3.5 on Apple MPS | 8/8 direct-mask adds, 8/8 model-generated comparison masks and adds, 4/4 two-instruction sequences, 4/4 divergent programs exact |
| Standalone stock Qwen3.5 on CUDA with optimized DeltaNet | The same four suites exact; causal convolution used the reference kernel |
| Pretrained Qwen3.5-9B merge on CUDA | 2/2 divergent programs exact, including all generated mask and lane-result tokens |
| Fixed text check on merged 9B | 6/6 prompts kept the same next token; 5/6 kept the same first four greedy tokens |

These are small, fixed execution and integration checks. The circuit uses 19
Gated DeltaNet heads and 11 circuit layers. The 9B merge adds an exit layer,
preserves the source model's untied text output head, and was built with 4,608
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
```

The next gates are compiled scratch memory, more than one ALU operation,
program memory and instruction fetch, nested divergence, and larger held-out
program suites. The reference interpreter checks model traces and is not used
to compute the model's outputs. The current one-level branch trace is driven by
host-supplied raw instruction tokens, so this is a SIMT circuit prototype and
not yet a self-contained programmable GPU. See [REPRODUCE.md](REPRODUCE.md)
for pinned inputs and commands.

Model weights are not stored in this repository. The Python code is MIT
licensed; Qwen3.5 weights remain under their own license.
