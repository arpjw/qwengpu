# QwenGPU scope and next gates

## Current checkpoint

The private [resident generation v3 release](RELEASE.md#restore-receipt)
merges the GPU circuit into Qwen3.5-9B. A launch uploads the program
and initial data. Within that launch, stock generation schedules warps,
executes instructions, and commits state. The model has two eight-lane
warps per workgroup, supports four sequential or two interleaved
workgroups, and has 32 ROM slots per warp and 512 bytes of programmable
global RAM alongside shared and private banks.

Merged CUDA evidence covers all 27 inherited tile fixtures, four
sequential and four interleaved workgroup fixtures, full-capacity vector
addition and two-pass reduction, two non-square tiled multiplies, and
the native-text handoff. A clean download passed all package checksums
and reproduced the seeded tile and native-text results. The complete
v3 CUDA replay and the standalone Apple MPS replay are tracked separately
in [VALIDATION.md](VALIDATION.md). The 86 GB merged checkpoint was
validated on CUDA.

The v3 16×16 launcher keeps one cache across its eight generation calls.
Earlier tilegrid evidence covers scheduling all eight panels inside one
launch. See the [reproduction commands](REPRODUCE.md#run-the-stock-generation-checkpoint).

## Earlier cross-backend baseline

The earlier `qwengpu-resident-memory512-9b-v1` layout runs on both
Apple MPS and merged CUDA. One fixed
constructed merged checkpoint runs uint32 vector addition up to 64 values,
reduction up to 128 values, and byte-input matrix multiply with uint32
accumulation for M <= 8, K <= 4, N <= 4. All 31 frozen common requests,
seven raw bank/barrier/collision probes, the separate raw matrix-extent
probe, and all 27 inherited cases match on standalone MPS and merged
Qwen3.5-9B CUDA.

The same checkpoint runs the specialized 16x16 modular uint16 multiply in
one cache, with four output tiles and two K slices. Both assembled scalar
pass matrices, all eight traces, transitions, and upload audits match.
Lane, thread, workgroup, and global IDs come from model state. One to four
sequential workgroups share 256 programmable uint16 words, or 512 bytes.
Dependent reduction launches retain that bank with zero memory re-upload.

Fresh text matches base on 24/24 first tokens and 21/24 short greedy
continuations. Four native-cache resets, a legacy full-width arithmetic
relaunch, and a reduction across native text reproduce their controls
exactly. The active expanded-bank handoff preserves all 8,192 indexed-port
nibbles. VALIDATION.md records the tests and initial native-text regression;
MEMORY512_DESIGN.md defines the supported interface.

The machine still uses host-relayed tokens, sixteen ROM slots per warp,
and sequential workgroups. Specialized matrix banks are a separate fixed
layout. The remaining gates below cover richer programs and arithmetic,
separate memory spaces, resident groups, and reduced host involvement.
These bounded results do not establish a complete general-purpose GPU.

## 32-slot ROM qualification

The `resident-rom32` variant doubles resident program capacity over the
512-byte layout. It preserves the old control format for PCs 0 through 15,
carries the fifth PC bit through barriers and resets, and decodes five-bit
forward and backward loop destinations. The merged allocation still uses
35 circuit layers and hidden width 7,424, with six additional live slots.

Two full-length programs test control flow above slot 15 and a workgroup
transition that restarts both PCs at zero while retaining global memory.
The ALU32 successor adds signed address offsets, immediate subtraction
and multiplication, full-width shifts, comparisons, and indexed bitwise
operations. The remaining arithmetic gate extends these operations to
arbitrary full-width operands. General runtime extents are also a separate
gate. See REPRODUCE.md for the
MPS and CUDA commands and VALIDATION.md for the supported evidence.

## Earlier gates

The standalone vector-grid circuit establishes the first grid scheduling
gate on MPS. A length descriptor drives up to two eight-lane workgroups
in one recurrent cache. The circuit selects each group, derives its
addresses, and commits 32-bit sums to resident memory on each step.
The follow-on grid layout needs to bring this boundary into tilepipe and
make global memory addressable by uploaded kernels.

The standalone tilegrid variant advances to a second workgroup from
inside tilepipe. It restarts the row scheduler and both PCs, resets
operand selection, and keeps the ROM and accumulator memory in the
same cache. The second workgroup can use a separately preloaded
output tile. Its stores carry the address of that output and preserve
the first tile. The model can also select different B panels for each
group, sharing the A panel across both. The 2×2 mode also selects the
A panel from the workgroup row. It stores four distinct C tiles in the
same cache. The A, B, and C images contain 1 KiB of logical matrix data
at fixed addresses. The full 16×16 multiply passed in one cache on standalone
MPS and the merged Qwen3.5-9B checkpoint on A100.
Tilegrid's matrix addresses are fixed. Globalgrid extends the same
circuit with a programmable launch of one to four sequential workgroups.
Each has two eight-lane warps with model-visible lane, thread, workgroup,
and global IDs. Global-mode indexed instructions address one resident
array of 128 unsigned 16-bit words. Adjacent even and odd words hold
the low and high limbs of a 32-bit value. Registers, indices, and masks
reset on each workgroup transition while global memory persists.

The vector add and two-pass reduction use the same tilegrid circuit
in single-workgroup mode. They exercise indexed 32-bit shared memory
and barriers between the two warps. The vector fixture encodes active
lanes in its launch data. The reduction's second launch reads the
partial sums from the same cache. Globalgrid adds a cross-workgroup
reduction path with independent partial sums in global memory.
Multiple simultaneously resident workgroups and larger address
spaces remain open.

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

The resident wide layout adds a high 16-bit register bank to each
warp. MPS and merged 9B preserve distinct seeded values through
scheduling, barriers, masked low-half writes, and the cooperative
tile kernel. The three state transports agree on the seeded kernel.

The resident ADD32 layout adds a ROM-encoded 32-bit immediate add.
MPS and merged 9B verify carry, modulo 2³² wrap, masked updates,
and exact high-half writeback. One mixed kernel runs ADD32I, tiled
multiply, shared-memory gather, and divergent loops in one launch.

The 16×16 runner accepts the resident ADD32 checkpoint. MPS and
merged 9B match the scalar reference after both accumulation passes
and match all eight earlier grid traces on their shared fields.
The composed multiply and the mixed ADD32I kernel passed from the
same merged directory.

The unified resident ADD32 layout passes all 23 frozen cases on
MPS and merged 9B. The case records match each other, the interpreter,
and the saved baselines, including row windows and terminal outcomes.

The fresh-cache text comparison on the unified merge matched base
Qwen3.5-9B on all 24 next tokens and 23 of 24 greedy continuations
of up to 16 tokens. The report records the logit differences and the
diverging continuation.

Direct cache reuse after the cooperative kernel keeps all four
first text tokens and three of four greedy continuations on the
same merged weights. Shifting positions in a fresh cache keeps
all four continuations. The handoff report records both comparisons.

Replacing the pretrained layers' cache entries starts text at
position zero while keeping the GPU layers in the same cache.
All four continuations and native next-token logits match the
fresh-cache control. The register and shared memory rows remain exact.

The GPU-to-text-to-GPU round trip uses one retained GPU cache.
The consumer copies sixteen producer-written values after text.
Its trace matches the interpreter and a fresh-cache control with
zero initial memory uploads.

Two K slices accumulate into one 8×8 output tile in one inference
cache on MPS and merged Qwen3.5-9B. The second slice refreshes A and
B and keeps C in the model. Both results match the scalar reference,
the prior tile traces, and the second slice's fresh-cache control.

The 16×16 launcher's `--retain-c` mode keeps C across both K slices
for each of the four output tiles, using one cache per output tile.
MPS and merged Qwen3.5-9B match both scalar pass matrices, all eight
prior traces, and all per-forward memory counts. The four second
launches omit 1,024 C upload tokens in total.

The shared32 layout moves both halves of a 32-bit value through
indexed shared memory. One warp loads both halves, adds with carry,
and stores both halves for a second warp to read after a barrier.
MPS and merged 9B match the independent reference for the handoff
and all 23 inherited ADD32 case records. Both reproduce all eight
retained 16×16 tile traces. Merged 9B also reproduces the single-tile
retention and GPU-to-text-to-GPU traces.

The addmem32 layout adds two arbitrary 32-bit operands from
indexed shared memory. Its low instruction adds with carry into
the upper half, and its high instruction completes the sum. The
MPS and merged 9B match the independent scalar result and all 24
inherited case records match the previous checkpoint. Both also
reproduce the retained 16×16 multiply, including all eight traces
and memory counts.

The addmem32 merge matches the earlier ADD32 merge on all measured
text fields on one H100. Both match base on 24 first tokens and 21
greedy continuations. Four cache resets preserve all inspected GPU
rows. A dependent kernel runs after text with zero initial shared
memory uploads and reproduces the inherited trace.

The addmem32 sum now serves as the producer in a GPU-to-text-to-GPU
round trip. A separate two-warp consumer copies eight full-width
values to disjoint shared pairs after text. MPS and merged 9B
match all GPU traces and all 128 shared words. Merged 9B matches
fresh-cache text exactly and skips the initial shared-memory upload.

The branch32 layout extends `IF_BIT` to bits 0 through 31.
Its fixture branches on the high bit of a 32-bit sum with
identical low halves. The then and else masks, reconvergence,
and the dependent warp match a scalar reference. MPS and merged
Qwen3.5-9B match all 26 cases and reproduce the eight retained
16×16 tile traces.

The loop32 layout extends `WHILE_BIT` to bits 0 through 31.
The first MPS fixture exits from carry into bit 16, with
different trip counts per lane. Masks, wraparound, and the
dependent warp match the scalar check. MPS matches the complete
27-case suite and all 26 inherited case records. Merged 9B
reproduces all 27 MPS case records. Both backends reproduce
the eight retained 16×16 tile traces.

The tilebanks layout preloads both K slices of one output tile in
the same cache. The host selects the second bank for the second
launch. MPS and merged 9B match both scalar pass matrices and
both prior traces, with zero initial memory uploads on the
second launch. The full MPS and merged replays match all 27
loop32 cases.

The tilepipe layout adds a model-selected K-slice transition
within one scheduled launch. MPS emits one transition after the
first slice, resets both PCs, selects bank one, and starts again
at row zero with the retained C and ROM. MPS and merged 9B
match both scalar matrices and all raw execution records. Both
also match the 27 inherited tilebanks case records.

The public 16×16 launcher now uses one scheduled launch per output
tile in `--schedule-k` mode. MPS and merged 9B match both scalar
matrices, all four model-generated transitions, and the upload
audits. On the same merged checkpoint, native text matches the
base first token on all 24 prompts and the greedy continuation
on 23. The four text resets and full-width GPU relaunch are exact
against their fresh-cache controls.

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

The next architectural gates after the qualified 512-byte checkpoint are:

| Gate | Required change | Evidence needed |
| --- | --- | --- |
| Separate address spaces | Add private and per-workgroup shared banks with explicit isolation from the 512-byte global bank | Boundary accesses, isolation, aliasing, guards, and cross-group visibility on both backends |
| More general programs | Add richer address arithmetic and runtime extents over the 32-slot ROM | Unseen uploaded programs and lengths on the same weights, with exact traces and bounded termination |
| Broader arithmetic | Full-width multiply, comparisons and bitwise operations; define signed and floating-point semantics separately | Overflow, carry, shift boundaries, signed extremes, and numerical edge cases where applicable |
| Multiple resident workgroups | Bank each group's PCs, registers, masks, and barrier state; schedule without resetting the only group bank | Interleaved producer/consumer traces, independent barriers, fairness, and explicit memory ordering |
| Less host involvement | Move tagged state and store writeback into recurrent model execution | Equivalent multi-instruction traces with measured reductions in forwarding and model calls |
| Larger end-to-end workloads | Tile arbitrary supported matrix shapes over the general memory/launch interface | Non-square shapes, tail tiles, multiple K panels, and scalar comparisons across seeded inputs |

The integrated integer layout now implements the first four gates and the
larger matrix workload. The merged 9B records include the new arithmetic,
runtime extents, independent workgroup barriers, scoped-memory isolation
and reset, and two non-square matrix shapes with tail tiles. The latest
cross-backend status is in VALIDATION.md. Its host transport coalesces
reads with scheduling and skips unchanged control tokens. The qualified
model-owned store variant removes programmable global and scoped store
tags, committing 2,086 words in the merged replay. General cyclic
register writeback through stock generation now reuses the same four
physical warp banks on interleaved launches. The model emits its own
schedule, control, register updates, and memory commit tokens from one
launch prompt. On merged CUDA, a two-pass 32-element reduction reuses
one cache with no global upload on the second launch. The 3×5 by 5×2
and 9×7 by 7×5 multiplies retain C across K panels. The v2 protocol
adds sequential workgroups. Standalone and merged 9B CUDA match all four
frozen sequential fixtures, including ROM reset, signed addressing, scoped
isolation, and runtime matrix extents. The inherited suite also
now has a legacy tile launch path in v3. Of its 27 fixtures, 22 use
`LOAD_TILE`, `MAC_K`, or `STORE_TILE`, four use legacy shared
memory with seeded registers and row windows, and one is a
barrier deadlock without a workgroup descriptor. The v3 merge passes
the native-text gate and all 27 inherited tile fixtures under autonomous
CUDA generation. Compiler and merged runs match 26,225 generated tokens
over 1,972 instructions. The merged run also matches the earlier
checkpoint's final tile memory and outcomes. The token-level writeback
edge remains visible in the generated stream.
See [MODEL_WRITEBACK_DESIGN.md](MODEL_WRITEBACK_DESIGN.md).
Atomics, floating-point execution, graphics,
CUDA compatibility, and useful acceleration have no current evidence.
The capacity step is qualified in MEMORY512_DESIGN.md. NEXT_MEMORY_DESIGN.md
retains its historical proposal and the original encoding hazards. Future
layout changes must preflight allocation and retain the native-text gate;
the validation pipeline checks text before the expensive GPU regressions.

For each gate, freeze programs before the model run, compare generated tokens
and actual recurrent state to the independent interpreter, and test both the
standalone circuit and the pretrained merge. Record model revision, compiler
commit, kernel path, device, dimensions, latency, and approximate provider
cost. Test separate caches for separate uploaded programs. A failure should
be reported as a failure, with the failing case retained.

The final checkpoint handoff also requires a private durable copy, a
clean restore with full-file integrity checks and GPU and native-text
runs, and a GitHub receipt before the final RunPod ends. The package,
license, and teardown requirements are in [RELEASE.md](RELEASE.md).

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
