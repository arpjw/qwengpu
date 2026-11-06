# QwenGPU scope and next gates

## Current milestone and next gate

The resident 8×8 tile engine holds its 256 shared bytes in one inference
cache per launch. A, B, and initial C are uploaded once. The host relays
model-tagged writes into the same cache. MPS and merged Qwen3.5-9B agree
on the dense multiply, store-and-reload, and deliberate deadlock traces.

The indexed variant uses the same resident cache for tiled operations
and arbitrary 16-bit accesses. It covers 128 word addresses with
per-lane indices and model-computed scatter priority.

The masked variant adds per-warp lane masks. Inactive lanes preserve
their registers and issue no writes. The lowest active lane wins
scatter collisions, and a warp with an empty mask can reach a barrier.

The divergent variant supports two nested IF_BIT blocks per warp.
ELSE uses the saved predicate, and ENDIF restores the parent mask.
The fixtures exercise an empty branch, a partial parent mask, and
accumulator changes between IF_BIT and ELSE.

The looping variant chooses data-dependent back edges and exits in
the model. One checkpoint supports 16-bit arithmetic, 128 indexed
shared words, barriers, and two levels each of branches and loops.
Its host relays state packets and enforces a launch instruction budget.

The cooperative fixture runs a tiled multiply, barrier, indexed gather,
and divergent loop in one launch. The producer and consumer share one
inference cache. Their final rows agree with a separate scalar result
on MPS and merged Qwen3.5-9B.

The resident launcher exposes the unified checkpoint through a JSON
launch and result format. It runs the cooperative fixture on MPS and
merged Qwen3.5-9B and exposes halt, deadlock, and step-limit outcomes.

The resident-ROM checkpoint uploads both warp programs once. Its
scheduler ticket and PC select a row in the recurrent ROM table.
Each program has sixteen slots. The looping checkpoint remains
available for comparisons.

The resident-state checkpoint banks each warp's accumulator, index,
mask, and stacks in the launch cache. The model selects the active
bank and emits a tagged packet, which the host forwards unchanged.

The resident-delta checkpoint emits a tagged state update only when
a nibble changes. The host relays those tags. All state read request
positions still run, including the ones that emit a zero sentinel.

The resident-plan checkpoint emits a bitmap of changed state
positions. The launcher requests those positions and all 32
shared-store positions in a follow-up call. This saves tokens
and adds one forward call per instruction.

The resident I/O checkpoint adds a store-lane bitmap to the
state read plan and skips empty follow-up calls. The launcher
still relays the resulting tagged values.

On the tested A100 cooperative kernel, fixed requests give the
lowest launch latency. The state and store plan gives the lowest
token count and peak allocation. The benchmark uses the same
merged checkpoint for all modes. The launcher keeps its existing
layout-based defaults pending evidence from more workloads.

The launcher exposes explicit transport selection on one merged
checkpoint. Fixed requests had the lowest latency on both measured
A100 fixtures. The state and store plan had the lowest token count
and peak allocation. Existing layout-based defaults remain available.

The resident grid checkpoint takes a launch extent of one through four
row pairs. The scheduler reads the final pair from the cache and
decides when to finish. MPS and merged Qwen3.5-9B match on all 15
fixtures, including three partial launches with untouched later C rows
and 12 unchanged full launches.

The grid launcher accepts a nonzero first row pair on the same
circuit. Four windows preserve C outside their scheduled rows on
MPS and merged 9B. Two complementary windows reproduce the full tile.

The 16×16 modular multiply uses one loaded grid checkpoint. The
launcher packs four 8×8 output tiles and carries each output tile
across two K slices. The model computes the multiply-accumulates.
MPS and merged 9B match the scalar reference at both K slices.

An isolated 32-bit lane adder propagates carry from the low
16-bit limb to the high limb. Twelve frozen cases for carry, modular
wrap, and inactive lanes match on MPS, standalone CUDA, and merged 9B.

The next gate adds a high 16-bit register bank to each resident
warp. Seed distinct words and verify that scheduling and state
transport preserve both limbs and keep the warps isolated. Use that
layout to integrate the 32-bit instruction in the following gate.

Use RUNPOD_LEDGER.md to track the total approved $74 compute cap and
reconcile pending estimates before provisioning more resources.

## Model-scheduled tile variant

The tile checkpoint selects ready warps, releases barriers, and signals
completion or deadlock. Each warp owns eight 16-bit accumulators. The
64-byte shared layout holds two 4×4 byte input matrices and one 4×4
16-bit output matrix. The fixed 4×4 engine accepts different operands
and tile programs. Its three frozen fixtures match on MPS and merged CUDA.
The host relays model-generated control and data tokens between calls.

## General byte workgroup variant

A fixed Qwen3.5-9B checkpoint executes one instruction per scheduled warp
turn. Two warps carry separate PC, registers, masks, and stacks. The
round-robin host scheduler copies generated state tokens and releases
barriers after both arrivals. The model performs indexed loads and stores,
scatter collision selection, and masked modular reduction in shared
32-byte scratch. Three frozen workgroups pass on MPS and merged CUDA.
This variant uses the general byte workgroup ISA. See VALIDATION.md and
REPRODUCE.md for the evidence and commands.

## Bounded control variant

The new control builder supports two levels each of structured divergence
and data-dependent WHILE loops, including loops inside branches and branches
inside loops. It separates execution step from PC and carries machine state
forward into a new depth stage on every executed instruction. There are 16
ROM slots and a 16-step execution budget. Budget exhaustion is reported
separately from HALT. All 14 frozen cases pass on MPS, optimized CUDA, and the pretrained 9B merge;
see VALIDATION.md for exact evidence. CONTROL_DESIGN.md defines exact semantics.

## Original SIMT prototype

The original artifact is an eight-lane SIMT circuit compiled into stock
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
exactly on MPS. The separate integrated builder now connects ROM fetch to the straight-line
SIMT ALU and passes the four frozen gate programs on MPS and optimized CUDA.
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

Gate 1 passed on standalone MPS, optimized CUDA, and the pinned 9B merge. The bounded
writeback variant now passes seven frozen programs on MPS, using 16
instruction stages in model depth. It also passed the same seven programs
on optimized CUDA and inside the 9B merge. This removes lane feedback but does not
provide cyclic state reuse. See WRITEBACK_DESIGN.md for the architecture and
the next bounded-control-flow design constraint. QwenCPU's dynamic ROM shows a possible
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


## Bounded writeback gate

The new `compiled_writeback` variant handles at most 16 straight-line
instructions with two token calls per instruction and no lane feedback.
Each instruction position owns a snapshot in recurrent state. All 17 local
tests pass, including exhaustive compiler checks for the parallel byte adder.
Seven fixed programs compare 51 intermediate 40-byte snapshots to the
interpreter on MPS. CUDA and pretrained-merge status is in VALIDATION.md.

For bounded control flow, execution-stage identity must be distinct from
program address. Carry registers, scratch, masks, and branch-stack state to
the next depth stage even when a branch changes the PC. Simply jumping back
to a previous PC in this layout would read stale state. Measure the depth,
weight, cache, and latency costs before extending the execution budget.
