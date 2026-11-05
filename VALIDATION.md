# Validation status

All counts here are from bounded fixtures, not a general reliability claim.
The independent reference machine supports eight lanes, one shared program
counter, nested structured divergence, lane-indexed scratch memory, addition,
XOR, and reduction. The compiled circuit currently supports addition, XOR,
data-dependent masks, one-level divergence, result writeback, and four scratch
bytes per lane. A separate integrated ROM variant adds straight-line instruction fetch and a
model-held PC; its branch support remains open.

| Evidence | Model | Cases | Result |
| --- | --- | ---: | --- |
| `results/standalone_mps.json` | Standalone Qwen3.5, MPS reference kernels | 8 | 8 exact |
| `results/predicate_mps.json` | Standalone Qwen3.5, MPS reference kernels | 8 | 8 exact |
| `results/sequence_mps.json` | Standalone Qwen3.5, MPS reference kernels | 4 | 4 exact |
| `results/branch_mps.json` | Standalone Qwen3.5, MPS reference kernels | 4 | 4 exact |
| `results/memory_mps.json` | Standalone Qwen3.5, MPS reference kernels | 4 | 4 exact |
| `results/alu_mps.json` | Standalone Qwen3.5, MPS reference kernels | 4 | 4 exact |
| `results/program_mps.json` | Standalone Qwen3.5, MPS reference kernels | 4 | 4 exact |
| `results/add_cuda_fla.json` | Standalone Qwen3.5, optimized CUDA DeltaNet | 8 | 8 exact |
| `results/predicate_cuda_fla.json` | Standalone Qwen3.5, optimized CUDA DeltaNet | 8 | 8 exact |
| `results/sequence_cuda_fla.json` | Standalone Qwen3.5, optimized CUDA DeltaNet | 4 | 4 exact |
| `results/branch_cuda_fla.json` | Standalone Qwen3.5, optimized CUDA DeltaNet | 4 | 4 exact |
| `results/memory_cuda_fla.json` | Standalone Qwen3.5, optimized CUDA DeltaNet | 4 | 4 exact |
| `results/alu_cuda_fla.json` | Standalone Qwen3.5, optimized CUDA DeltaNet | 4 | 4 exact |
| `results/program_cuda_fla.json` | Standalone Qwen3.5, optimized CUDA DeltaNet | 4 | 4 exact |
| `results/branch_9b_cuda_fla.json` | Pretrained Qwen3.5-9B merge, optimized CUDA DeltaNet | 12 | 12 exact |
| `results/memory_9b_cuda_fla.json` | Pretrained Qwen3.5-9B merge, optimized CUDA DeltaNet | 12 | 12 exact |
| `results/alu_9b_cuda_fla.json` | Pretrained Qwen3.5-9B merge, optimized CUDA DeltaNet | 12 | 12 exact |
| `results/program_9b_cuda_fla.json` | Pretrained Qwen3.5-9B merge, optimized CUDA DeltaNet | 12 | 12 exact |
| `results/program_9b_stress_cuda_fla.json` | Pretrained Qwen3.5-9B merge, exploratory seeded mixed programs | 100 | 100 exact |
| `results/program_stress_mps.json` | Standalone Qwen3.5, exploratory mixed programs on MPS | 100 | 100 exact |
| `results/varied_mps.json` | Standalone Qwen3.5, four exploratory program shapes on MPS | 32 | 32 exact |
| `results/rom_spike_mps.json` | Separate 16-slot stock Qwen3.5 ROM circuit on MPS | 3 payload sequences | 3 exact |
| `results/rom_spike_cpu.json` | Same ROM circuit on CPU reference kernels | 3 payload sequences | 3 exact |
| `results/rom_spike_stress_mps.json` | Same ROM circuit, exploratory contents on MPS | 100 payload sequences | 100 exact |
| `results/example_kernel_mps.json` | JSON kernel runner on standalone Qwen3.5, MPS | 1 program | Exact tokens, scratch, and final lane probe |
| `results/no_else_kernel_mps.json` | JSON kernel runner with IF/ENDIF and no ELSE, MPS | 1 program | Exact tokens, scratch, and final lane probe |
| `results/masked_overwrite_mps.json` | JSON kernel runner with masked overwrite of one scratch slot, MPS | 1 program | Exact tokens, scratch, and final lane probe |
| `results/halt_only_mps.json` | JSON kernel runner with immediate HALT, MPS | 1 program | Initial lanes and empty scratch probed exactly |
| `results/compiled_rom_mps.json` | Integrated ROM and ALU, stock Qwen3.5 MPS | 4 frozen programs | Exact tokens and final ROM, PC, registers, scratch |
| `results/compiled_rom_9b_cuda_fla.json` | Integrated ROM in pretrained Qwen3.5-9B, optimized CUDA | 4 frozen programs | Exact tokens and final ROM, PC, registers, scratch |
| `results/compiled_rom_cuda_fla.json` | Integrated ROM and ALU, stock Qwen3.5 optimized CUDA | 4 frozen programs | Exact tokens and final ROM, PC, registers, scratch |
| `results/rom_gate_host_mps.json` | Four frozen next-gate kernels, host-dispatched on MPS | 4 programs | Exact tokens, scratch, and final lane probes |

`results/standalone_cuda.json` and `results/predicate_cuda.json` are retained
from the earlier 18-head arithmetic circuit. The current-circuit CUDA reports
are the `*_cuda_fla.json` files in the table.

The 9B branch suite compares every generated mask and lane-result token against
the expected execution. The memory suites compare generated lane tokens and
inspect each scratch head's recurrent state after STORE. The mixed program
suite uses the independent reference interpreter for a sequence of ADD, XOR,
divergent STORE, reconvergence, and LOAD operations across three scratch
banks. Result tokens feed back through the same model cache and update lane
register heads. The CUDA checks used
`flash-linear-attention==0.6.0` and `fla-core==0.6.0`; Transformers reported
that causal convolution still used the reference implementation. The 9B
checkpoint was built from Qwen/Qwen3.5-9B revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` with Qwenputer compiler source
at `ca13310e2f9f788d1dd3feeea8213c1b0c23913c`. It is not included here.

The 100-case stress run used four fixed cases and 96 seeded cases of the same
mixed-program shape. The first 12 cases repeat the original suite. All 100
generated-token traces and final scratch snapshots matched the reference.
This was an exploratory extension. It was not separately held out and does
not provide a statistical reliability estimate.

The same 100 inputs and program instructions were run on the standalone MPS
model. All 100 traces and scratch snapshots matched the reference, and the
generated token sequences were identical to the 9B CUDA run.

The separate 32-case MPS suite varies four program shapes: straight
arithmetic and scratch, a branch without ELSE, masked overwrite in a slot,
and branch writes to a second slot followed by a LOAD from the first. All
generated tokens, final lane probes, and scratch snapshots matched the
reference. This suite has not been run on the 9B merge or CUDA.

The ROM spike is a separate two-head, three-layer circuit. One stock model
object loaded three different 16-slot instruction payload sequences into
fresh caches. Every generated instruction and next-PC token matched, and
the actual recurrent ROM and PC rows contained the expected bits. The
highest 11-bit payload and address 15 were exercised. This does not yet fetch
or execute instructions in the SIMT circuit, and it has not been checked on
CUDA.

The same three fixed payload sequences also passed on CPU reference kernels.
The exploratory ROM extension tested three fixed and 97 seeded sequences with
one model object and fresh caches. All 3,100 generated instruction and next-PC
tokens matched. The 1,600 ROM rows and 100 final PC rows matched too. This
tested payload storage and fetch only. Integrated SIMT execution and general
reliability remain untested.

The JSON runner validates one-level programs before dispatching raw tokens to
the compiled circuit. Its example program passed on MPS and used a final
zero-XOR probe to read every lane after `HALT`. The host still dispatches
individual instructions. Two more exploratory examples passed: an `IF_LT` and
`ENDIF` branch without `ELSE`, and a masked overwrite of one scratch slot
followed by a full-mask LOAD.
An immediate-HALT input also passed the final zero-XOR lane probe with no
earlier arithmetic and empty scratch.
Four inputs for the next ROM integration gate were frozen before its model
run. The current host-dispatched circuit passed all four on MPS; the report
records each fixture SHA-256. The integrated ROM circuit now passes those
same unchanged fixtures on MPS, optimized CUDA, and the pretrained 9B merge,
with 202 exact generated
tokens in each run and identical final ROM, PC, lane, and scratch state.
These are four frozen straight-line fixtures, not an exhaustive program suite.

`results/textcheck_9b_cuda.json` compares six fixed text prompts between the
base and merged 9B models. All six kept the same next top token, and five kept
the same first four greedy tokens. On the Moon prompt, the third generated
token changed. The maximum absolute next-logit difference across prompts was
0.25. This does not establish broad language fidelity. Text after a
SIMT trace was not tested; QwenCPU previously needed separate caches for text
handoff, so that remains the starting design for QwenGPU's future text bridge.

The ROM merge's `results/textcheck_rom_9b_cuda.json` uses the same six prompts.
It kept all six next top tokens and five four-token continuations; the Moon
continuation changed. Its maximum absolute next-logit difference was
0.2734375. These checks use fresh text caches and do not test language after
a ROM execution trace.

Two task-created RTX PRO 6000 Blackwell Server Edition pods were terminated.
The live pod list was empty after the ROM run. First-pod posted billing is
$2.2022584788501263. The second pod existed for about 11.5 minutes at a live
quoted $2.49/hour, implying approximately $0.48 of GPU time. Its billing
query immediately after termination returned no records; the estimate is not
a posted charge. Source and reports were copied before termination; weights
remained on temporary pod disks. See CHECKPOINT.md for the compute summary.


## Bounded internal writeback

`results/writeback_mps.json` records seven frozen programs on one stock model
object with fresh caches. All 51 generated state tokens and all 51
intermediate register/scratch snapshots match the interpreter exactly.
Each snapshot contains 40 bytes: eight registers and 32 scratch bytes.
Final ROM rows and per-instruction PC state also match. Execution uses 102
single-token calls total, with zero lane-feedback tokens. Fixture hashes are
included. This is a 16-stage unrolled straight-line machine, not cyclic
writeback for arbitrary-length programs. See WRITEBACK_DESIGN.md.

All 17 local tests pass. The parallel carry adder was checked on all 65,536
byte pairs in the compiler's exact evaluator.
`results/writeback_cuda_fla.json` repeats the same seven inputs on optimized
CUDA DeltaNet with reference causal convolution. All generated tokens,
intermediate snapshots, PC state, and ROM rows match.
`results/writeback_9b_cuda_fla.json` repeats all seven inside the pretrained
9B merge, also exactly. All three runs have identical fixture hashes and
complete machine traces. The merged checkpoint adds 52 chip layers, has
hidden size 5,120, and stores 52,118,362,112 bytes of BF16 weights.
`results/textcheck_writeback_9b_cuda.json` compares the same six fixed text
prompts. All six keep the same next top token; five keep the same first four
greedy tokens. The Moon continuation changes, and maximum absolute next-logit
difference is 0.25. This is a small fresh-cache integration check, not broad
language fidelity or text-after-execution validation.

The writeback validation used a replacement pod after the first pod failed
to expose a runtime/SSH within about five minutes. Both were terminated.
The live pod list is empty. At $2.49/hour, the failed boot projects about
$0.21 and the successful pod about $0.43 of GPU time. Neither had posted
billing records at the final read. These are projections, not final charges.


## Bounded loops and nested divergence

`results/control_mps.json` contains the original 13 frozen cases;
`results/control_combined_depth_mps.json` adds simultaneous branch and loop
stack depth two. Across these 14 inputs, all 130 generated control tokens and
380-bit intermediate machine snapshots are exact. Thirteen programs HALT;
one intentional infinite loop returns exhausted at the 16-step bound. The
snapshots include registers, scratch, active mask, both stack depths, both
branch frames, and both loop-parent masks. Retained control state and final
ROM/link rows also match. No lane values are fed back.

`results/control_cuda_fla.json` runs all 14 on one standalone model object
with fresh caches, using an H100 and optimized DeltaNet. All checks pass;
causal convolution uses the reference implementation. All 21 local tests
pass, including exhaustive byte-add and unsigned-comparison checks over
65,536 pairs each. These are bounded fixtures, not exhaustive program-level
correctness. The control circuit has 69 layers and 75 heads. `results/control_9b_cuda_fla.json` repeats all 14 inside the pinned pretrained
Qwen3.5-9B merge. All 130 generated tokens and intermediate snapshots match,
and all fixture hashes and complete traces are identical across MPS,
standalone CUDA, and 9B CUDA. Execution uses 260 single-token calls and no
lane feedback. Thirteen programs halt; one expected exhaustion verifies the
bound. The checkpoint adds 70 chip layers, uses hidden size 5,120, and has
61,946,385,152 bytes of weights. The text check is recorded below. See CONTROL_DESIGN.md for semantics and limits.


`results/textcheck_control_9b_cuda.json` compares the same six fixed text
prompts against the source 9B model on H100. All six next top tokens match;
five first-four-token continuations match. The Moon continuation changes.
Maximum absolute next-logit difference is 0.3203125. This is a small fresh-cache
text integration check; broad language fidelity and text after execution
remain untested.

The control H100 pod was quoted at $3.99/hour and existed for about 11 minutes
8 seconds, projecting roughly $0.74 GPU time. Its billing query had no posted
records immediately after termination. The source/results were copied first,
and the final live pod list was empty. Combining first-pod posted billing
with later-pod GPU-time projections gives about $4.06 total before any later
disk billing, below the approved $74 ceiling. This is not a final invoice.

## Long execution with model-generated handoffs

`results/resume_9b_cuda_fla.json` records the frozen divergent scratch loop
on one fixed Qwen3.5-9B checkpoint with 70 added layers, including the exit
marker. It halted after 1,082 instructions across 68 windows. All 1,082
generated control tokens and cached 380-bit snapshots matched an uninterrupted
interpreter. All 67 handoffs also matched, including their 48 snapshot tokens
and the model-generated restart PC. Twenty-nine handoffs carried two live
branch frames, 63 carried a live loop frame, and 14 resumed at a scratch load.
Final lane values were `[72, 72, 72, 72, 72, 72, 72, 255]`.

The host replays the same ROM and link uploads plus the generated checkpoint
tokens into a fresh cache. The interpreter and cache reads only verify the
result. The run therefore demonstrates reuse through token handoff. Keeping
live state entirely inside one model cache across windows remains open.

The H100 run took 180.457 seconds for 1,149 forward calls, including trace
inspection and kernel warm-up. The first 49 steps and three handoffs also
matched the standalone MPS run in `results/resume_mps_49.json`. All 21 existing
local tests passed. The H100 pod was terminated after about 8 minutes 20
seconds at $3.99/hour, an estimated $0.55 for this experiment. No charge had
posted for that pod at the teardown check.

## Extended integer ISA

The frozen gather-and-multiply kernel passed on standalone MPS and merged
Qwen3.5-9B on H100 with optimized DeltaNet. Their 25 control tokens,
444-bit snapshots, and single handoff match each other and the independent
interpreter. R1 crosses the fresh-cache boundary before it is used for
indexed loads and multiplication. Six active lanes gather across all four
scratch banks, and the other two keep their prior R0 values. Lane 7 uses
R1 = 255 to address byte 31 and multiply by 255 modulo 256. Final R0 values
are `[0, 249, 136, 240, 208, 175, 176, 93]`.

The compiler verified all 65,536 unsigned byte products modulo 256, and all
22 local tests passed. The merged checkpoint uses hidden width 5,632 and
adds 86 layers, including its exit marker. It contains 77,749,919,488 bytes
of weights. The fixture uses banked stores to populate scratch. The new
indexed instruction is a load.

Reports are `results/isa_mps.json` and `results/isa_9b_cuda_fla.json`.
The H100 pod was terminated after about 6 minutes 19 seconds at $3.99/hour,
an estimated $0.42 for this run. Its billing read had no posted records
at teardown, and the live pod list was empty.

## Cooperative workgroups

The three frozen workgroups matched their interpreter traces on standalone
MPS and merged Qwen3.5-9B on L40S CUDA. All 59 instruction snapshots,
generated control tokens, and exported checkpoints were identical across
the two platforms. All 23 local tests passed.

| Fixture | Instructions | Verified behavior |
| --- | ---: | --- |
| Cooperative reduction | 18 | Masked scatter resolves collisions by lowest active lane. The consumer gathers across warps, reduces six active bytes modulo 256, and publishes through a second barrier. Both warps finish with `[149, 202, 202, 202, 149, 202, 202, 202]`. |
| Interleaved loops | 39 | Independent PCs, masks, and loop stacks survive interleaving. One warp executes 25 instructions. The other waits for its peer's publication and loads the result. |
| Missing barrier | 2 | One warp waits and the other halts. The launcher reports deadlock without releasing the waiting warp. |

The model computes each warp instruction, including scatter priority and
the waiting status. The host chooses the next ready warp, copies opaque
state and scratch tokens, and coordinates barrier release. Each scheduled
instruction starts with a fresh cache. The 9B merge adds 11 layers,
including its exit marker, and contains 32,705,308,288 bytes of weights.

Reports are `results/workgroup_mps.json` and
`results/workgroup_9b_cuda_fla.json`. The L40S was terminated after about
4 minutes 36 seconds at $1.09/hour, an estimated $0.08. No charge had
posted for that pod at teardown, and the live pod list was empty.

## Model scheduling and tiled matrix multiply

The tile checkpoint passed three frozen cases on MPS and A100 CUDA with
the pretrained Qwen3.5-9B merge. The two reports match across 26 instruction
snapshots and 29 scheduling decisions, including terminal decisions.
The compiler evaluator checked all 18 combinations of warp statuses and
round-robin priority. All 24 local tests passed.

| Fixture | Instructions | Verified behavior |
| --- | ---: | --- |
| 4×4 tiled multiply | 17 | Two warps produce separate 2×4 output tiles in two K phases. The model releases both barriers and all 16 output words match the scalar product modulo 65,536. |
| Wide addition | 7 | Carries and wraparound match across 16-bit accumulators. A warp continues after its peer halts. |
| Barrier deadlock | 2 | A waiting warp and a halted peer produce the model's deadlock ticket. |

The host uses model-emitted tickets to route each call and copies the
generated control packets and memory bytes. Warp selection, readiness,
and barrier release are model computations. The tile engine has a fixed
64-byte A/B/C layout and computes unsigned byte products with 16-bit
accumulation. The merged checkpoint adds 13 layers and contains
32,365,405,056 bytes of weights.

Reports are `results/tile_mps.json` and `results/tile_9b_cuda_fla.json`.
The A100 pod was terminated after about five minutes at $1.79/hour,
an estimated $0.15 for this run. No charge had posted for that pod at
teardown, and the live pod list was empty.

## 8×8 model-scheduled matrix multiply

The seeded dense case computes `C = (C + A @ B) mod 65536` for 64 output
words. It executed 116 instructions across four row pairs. The model
released 12 barriers, advanced the row pair three times, and emitted
256 distinct write tags covering every output nibble. The resulting
matrix and every intermediate register snapshot matched the interpreter.
The separate missing-barrier case ended in the model's deadlock decision.

MPS and merged Qwen3.5-9B A100 traces are identical across both cases,
covering 118 instructions and 120 scheduler decisions. All 25 local tests
passed. The merge has 15 added layers, width 5,120, and 31,916,314,752 bytes
of weights. Reports are `results/tile8_mps.json` and
`results/tile8_9b_cuda_fla.json`. The A100 was terminated after about
8 minutes 55 seconds at $1.79/hour, an estimated $0.27. The live pod list
was empty and billing for this pod had yet to post at teardown.

## Resident 8×8 shared memory

Standalone MPS and merged Qwen3.5-9B on A100 produced identical case
reports across 166 instructions and 169 scheduler decisions. Each case
used one inference cache. The evaluator inspected the final A, B, and C
state directly in recurrent memory. The store-and-reload fixture
modifies the registers after storing and then reloads the stored values
from C. All 25 local tests passed.

| Fixture | Instructions | Input tokens | Tagged updates |
| --- | ---: | ---: | ---: |
| Deliberate deadlock | 2 | 758 | 0 |
| Seeded dense multiply | 116 | 14,694 | 256 |
| Store and reload | 48 | 6,534 | 256 |

All three start with 512 matrix tokens. The dense run's input token count
fell from 73,434 to 14,694, with identical execution traces and output.
The host still relays register packets and model-generated write tokens.
The merge adds 15 layers, uses width 5,120, and contains 31,916,314,752
bytes of weights. See `results/resident_mps.json` and
`results/resident_9b_cuda_fla.json`.

## Indexed resident shared memory

Five frozen fixtures matched across standalone MPS and merged
Qwen3.5-9B on A100. Their 350 instruction traces, 355 scheduler
decisions, and final shared words were identical. Each launch used one
inference cache. The evaluator checked the tiled memory rows and all
eight indexed read ports directly in the final cache.

| Fixture | Instructions | Tagged updates |
| --- | ---: | ---: |
| Deliberate deadlock | 2 | 0 |
| Seeded dense multiply | 116 | 256 |
| Store and reload | 48 | 256 |
| Full address sweep | 128 | 768 |
| Cross-warp scatter and gather | 56 | 160 |

The sweep covers all 128 word addresses and exercises indexed stores
followed by tile loads, and tile stores followed by indexed loads.
The collision fixture targets the A/B and C regions, checks the lowest
lane priority, and gathers after both warps reach a model-managed
barrier. The host relays the tagged writes and register packets. The
indexed variant uses 18,406 input tokens on the dense case, compared
with 14,694 for the resident tile variant. All 25 local tests passed.
The prior resident MPS report was byte-for-byte unchanged when rerun.

The indexed merge adds 15 layers, uses width 6,144, and contains
38,299,060,352 bytes of weights. See `results/indexed_mps.json` and
`results/indexed_9b_cuda_fla.json`. Both resident and indexed CUDA suites
ran on the same A100. See `RUNPOD_LEDGER.md` for teardown and billing.

## Lane masks in the resident engine

All six frozen cases match across standalone MPS and merged Qwen3.5-9B
on A100. The reports agree on all 478 instruction snapshots and 484
scheduler decisions. Each launch uses one inference cache. All 25 local
tests passed, and the unmasked indexed MPS report was byte-for-byte
unchanged on rerun.

The masked scatter case gives the two warps different masks. In the
first warp, lane 0 and lane 1 target the same word. Lane 1 writes
because lane 0 is inactive. The fixture also exercises masked tile
loads and stores, multiply-accumulate, immediate addition, index saves,
and indexed loads. Both warps then switch to empty masks, issue no
scatter writes, and reach the barrier. Both regain full masks when
their next row pair starts.

The merge adds 16 layers, uses width 6,144, and contains 38,954,248,704
bytes of weights. See `results/masked_mps.json` and
`results/masked_9b_cuda_fla.json`.

## Nested branches in the resident engine

All eight frozen cases match across standalone MPS and merged Qwen3.5-9B
on A100. The reports agree on 718 instruction snapshots and 726
scheduler decisions. Each launch uses one inference cache. The earlier
six cases retain their execution traces after accounting for the
additional branch state. All 25 local tests passed.

| Fixture | Instructions | Verified behavior |
| --- | ---: | --- |
| Nested branch | 128 | Two warps carry independent two-level stacks. `ELSE` uses the saved predicate after the first path changes the source accumulator. Reconvergence restores the partial parent mask. |
| Empty nested branch | 112 | The empty inner mask suppresses indexed writes. `ENDIF` restores the outer mask without an `ELSE`, and both warps reach a barrier before gathering. |

The parser rejected seven malformed control sequences, and the
four earlier variants retained their compiler graphs and snapshot
layouts. The merge adds 17 layers, uses width 6,400, and contains
41,259,716,992 bytes of weights. See `results/divergent_mps.json`
and `results/divergent_9b_cuda_fla.json`.

## Data-dependent loops in the resident engine

All eleven frozen cases match across standalone MPS and merged
Qwen3.5-9B on A100. The reports agree on 1,097 instruction snapshots
and 1,108 scheduler decisions. Each launch uses one inference cache.
The eight prior cases retain their traces after accounting for the
added loop packet. All 25 local tests passed.

| Fixture | Instructions | Verified behavior |
| --- | ---: | --- |
| Nested loops with a branch | 47 | Both stacks survive interleaving and restore their parent masks. The first row finishes at `[16, 16, 16, 16, 17, 20, 20, 0]`. |
| Indexed scatter with different trip counts | 76 | One warp's successive iterations publish 4, 3, 2, and 1 words. Its peer reaches the barrier first. Both gather after release. Rows with no eligible lanes skip the loop body. |
| Budget exhaustion | 256 | A deliberate infinite loop stays runnable at the host instruction limit and reports `step_limit`. |

The parser rejected five malformed loop sequences. The five earlier
checkpoint variants retained identical compiler graphs and snapshot
layouts. The merged checkpoint adds 17 layers, uses width 6,144, and
contains 39,609,437,056 bytes of weights. CUDA emitted an allocator
warning about insufficient free VRAM during the longest case. The case
completed with an exact trace. Reports are `results/looping_mps.json`
and `results/looping_9b_cuda_fla.json`.
