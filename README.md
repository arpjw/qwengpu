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

The first compiled circuit performs eight masked byte additions in one Qwen3.5
forward pass. Its inputs, lane mask, and results live in Gated DeltaNet state;
the host only supplies raw input and read-request tokens. This graph has 18
heads, 11 circuit layers, and 505 dimensions. A standalone stock
`Qwen3_5ForCausalLM` model with 576 hidden dimensions produced all 64 expected
result tokens across eight fixed cases on Apple MPS. This is a vector
arithmetic proof, not yet a programmable GPU or a pretrained-model merge.

```sh
python3 -m unittest discover -s tests -v
```

The compiler checks require Qwenputer's `experimental/riscv` package on
`PYTHONPATH`. The standalone model check additionally requires PyTorch and
Transformers:

```sh
python3 -m qwengpu.eval_add --device mps --out results/standalone_mps.json
```

The next gates are sequential vector instructions, model-generated active
masks and divergent control flow, scratch memory, program loading, and a
pretrained-model merge. The reference interpreter checks model traces and is
not used to compute the model's outputs.

No Qwen model weights or paid compute artifacts are stored in this repository.
