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

The repository includes an independent interpreter with richer semantics.
Fixed checks on MPS and CUDA cover direct instructions, branch paths, scratch
state, and mixed programs. The 9B merge passed 12 cases in each of the
branch, scratch, ADD/XOR, and mixed program suites. These checks use fixed
and seeded cases; they do not establish correctness for arbitrary programs.
See [VALIDATION.md](VALIDATION.md) for exact evidence and limits.

## Recommended build sequence

| Gate | Implementation | Acceptance test |
| --- | --- | --- |
| 1. Reusable kernel ROM | Upload a bounded program into recurrent state; add a PC and `TICK`-driven fetch/decode path | One checkpoint runs at least four different valid programs on separate fresh caches, with exact instruction and state traces on MPS and optimized CUDA |
| 2. Internal writeback | Feed ALU results directly into lane registers in the circuit | A multi-instruction program runs without eight host `READ` and output-token feedback pairs per instruction; compare every intermediate lane state with the interpreter |
| 3. Control flow | Add a bounded branch stack and loop/branch semantics | Nested divergence and reconvergence match the interpreter across empty, full, and mixed masks; invalid programs fail in the loader |
| 4. Memory and warp scale | Grow scratch capacity and add a second independent warp | Test bank independence, masked writes, address boundaries, and warp isolation; report model width, head, layer, token, and latency costs |
| 5. Pretrained model integration | Rebuild the 9B merge from pinned source and test a separate text path | Frozen SIMT suites pass, and text behavior is measured on a broader fixed corpus, including text after SIMT with cache handoff |

Gate 1 is the next experiment. QwenCPU's dynamic ROM shows a possible
Gated DeltaNet encoding, but the GPU circuit needs its own instruction format
and sequencer. A useful first kernel can be 16 instructions with ADD, XOR,
STORE, LOAD, IF/ELSE/ENDIF, and HALT. Program upload and launch may remain
host actions; individual instructions should come from model state after
launch. Keep the ROM size and branching depth bounded until the model-level
traces are exact.

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
