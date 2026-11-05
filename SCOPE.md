# QwenGPU scope and next gates

## What exists now

The current artifact is an eight-lane SIMT circuit compiled into stock
Qwen3.5 layers, then merged before a pretrained Qwen3.5-9B text model. Each
lane has one byte accumulator. The circuit implements masked ADD and XOR,
data-dependent comparison masks, one-level IF/ELSE/ENDIF, and four scratch
bytes per lane. Inputs and results are held in Gated DeltaNet state. One ADD
or XOR instruction computes all eight lane values in one model forward pass.
The caller still sends each instruction and requests eight output tokens to
write results back to lane state. This is a functional SIMT proof, not a
general-purpose or fast GPU.
The [JSON kernel runner](qwengpu/run_kernel.py) validates and dispatches
supported programs for local experimentation.
For a uniform ADD or XOR immediate, the current runner feeds eight operand
tokens, one execution token, eight read requests, and eight generated result
tokens back into the model. That is 25 one-token forward calls per vector
instruction after initialization. ROM operands and internal writeback target
this protocol cost.

The repository includes an independent interpreter with richer semantics.
Fixed checks on MPS and CUDA cover direct instructions, branch paths, scratch
state, and mixed programs. The 9B merge passed 12 cases in each of the
branch, scratch, ADD/XOR, and mixed program suites. These checks use fixed
and seeded cases; they do not establish correctness for arbitrary programs.
An exploratory 100-case run of the same mixed-program shape also passed on
both standalone MPS and merged 9B CUDA, with identical generated tokens.
An additional 32-case four-shape suite passed on standalone MPS only.
An isolated 16-slot ROM and PC circuit fetched three uploaded sequences
exactly on MPS. It has not yet been connected to the SIMT ALU.
An exploratory extension checked 100 uploaded sequences on the same model
object with fresh caches, also exactly.
See [VALIDATION.md](VALIDATION.md) for exact evidence and limits.

## Recommended build sequence

| Gate | Implementation | Acceptance test |
| --- | --- | --- |
| 1. Reusable kernel ROM | Upload a bounded straight-line program into recurrent state; add a PC and `TICK`-driven fetch/decode path | One checkpoint runs at least four different valid programs on separate fresh caches, with exact instruction and state traces on MPS and optimized CUDA |
| 2. Internal writeback | Feed ALU results directly into lane registers in the circuit | A multi-instruction program runs without eight host `READ` and output-token feedback pairs per instruction; compare every intermediate lane state with the interpreter |
| 3. Control flow | Add a bounded branch stack and loop/branch semantics | Nested divergence and reconvergence match the interpreter across empty, full, and mixed masks; invalid programs fail in the loader |
| 4. Memory and warp scale | Grow scratch capacity and add a second independent warp | Test bank independence, masked writes, address boundaries, and warp isolation; report model width, head, layer, token, and latency costs |
| 5. Pretrained model integration | Rebuild the 9B merge from pinned source and test a separate text path | Frozen SIMT suites pass, and text behavior is measured on a broader fixed corpus, including text after SIMT with cache handoff |

Gate 1 is the next experiment. QwenCPU's dynamic ROM shows a possible
Gated DeltaNet encoding, but the GPU circuit needs its own instruction format
and sequencer. Start with at most 16 straight-line ADD, XOR, STORE, LOAD, and
HALT instructions; add divergence in Gate 3. Program upload and launch may
remain host actions; individual instructions should come from model state
after launch. Keep the ROM size bounded until model-level traces are exact.
See [ROM_DESIGN.md](ROM_DESIGN.md) for the concrete first design.

## Measurement and decision rules

For each gate, freeze programs before the model run, compare generated tokens
and actual recurrent state to the independent interpreter, and test both the
standalone circuit and the pretrained merge. Record model revision, compiler
commit, kernel path, device, dimensions, latency, and approximate provider
cost. Test separate caches for separate uploaded programs. A failure should
be reported as a failure, with the failing case retained.

There is no acceleration claim. A 9B model forward pass per token makes the
current trace slow, and one vector operation currently needs extra reads and
feedback. The objective is to determine whether a reusable GPU-like machine can live
in model state while preserving ordinary text behavior. Throughput and
competitive performance would require separate measurements and substantial
architectural work.
