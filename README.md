# QwenGPU

QwenGPU is an experiment in compiling a small SIMT processor into stock
Qwen3.5 layers. It builds on the [QwenCPU](https://github.com/arpjw/qwencpu)
experiment and uses [Qwenputer](https://github.com/divagr18/qwenputer)'s
generic circuit compiler as a backend. The CPU, GPU, and compiler names here
refer to simulated machines inside model state, not physical hardware.

The current repository contains an independent eight-lane reference machine.
Each lane has an unsigned byte accumulator. The warp shares an instruction
stream and active mask. Structured `IF_LT`/`ELSE`/`ENDIF` executes divergent
paths under disjoint masks and then reconverges. Lane-indexed scratch memory,
addition, XOR, and reduction are specified and tested. A compiled Qwen3.5
circuit and checkpoint are still in progress.

```sh
python3 -m unittest discover -s tests -v
```

The intended first model gate is exact elementwise addition with masked lanes
and Gated DeltaNet state for input and output. Later gates cover divergent
control flow, scratch memory, program loading, and a pretrained-model merge.
The reference interpreter checks model traces and is not used to compute the
model's outputs.

No Qwen model weights or paid compute artifacts are stored in this repository.
