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

## Cooperative tiled multiply and gather

The looping checkpoint ran one cooperative kernel on standalone MPS
and merged Qwen3.5-9B on A100. The platforms produced identical traces
for 132 instructions and 133 scheduler decisions. Both matched the
separate scalar calculation for all 64 output words, including 30
producer sums that wrap at 16 bits.

Warp 0 published 32 tiled multiply words across four row pairs. The
scheduler released four barriers. Every one of warp 1's 32 gathers
read the updated peer word at its selected address. Its lanes took
zero, one, or two loop iterations according to the gathered values.
The launch used one cache, 512 initial memory tokens, and 256 tagged
writes. All 25 local tests passed.

Reports are `results/cooperative_matmul_mps.json` and
`results/cooperative_matmul_9b_cuda_fla.json`. The merge metadata
matches the checkpoint used for the data-dependent loop gate.

## Resident JSON launcher

The launcher ran the cooperative fixture on MPS and the merged 9B
checkpoint. The output matrix, shared words, transport counts,
instruction count, and outcome match the evaluator on both platforms.
With `--trace`, the merged launcher also reproduced all 132 instruction
snapshots and 133 scheduling decisions. See
`results/cooperative_launch_mps.json` and
`results/cooperative_launch_9b_cuda_fla.json`.

The MPS launcher returned exit code 3 with a deadlock result and
exit code 4 with a budget-exhaustion result at three instructions.
An invalid zero budget returned exit code 2. All 25 local tests passed.

## Resident warp programs

All twelve fixtures match across standalone MPS and merged Qwen3.5-9B
on A100, covering 1,229 instructions and 1,241 scheduler decisions.
Each fixture preserves its full execution trace relative to the
looping checkpoint. The evaluator checks both 16-slot ROMs and the
shared memory directly in the single launch cache. The JSON launcher
reproduced the cooperative fixture on both platforms. All 25 local
tests passed, and the six earlier compiler graphs were unchanged.

| Fixture | Prior input tokens | Resident-ROM input tokens |
| --- | ---: | ---: |
| Seeded dense multiply | 22,118 | 20,296 |
| Cooperative multiply and gather | 25,062 | 22,984 |
| 256-instruction budget exhaustion | 47,622 | 43,560 |
| Two-instruction deadlock | 886 | 888 |

Each launch sends 32 padded program tokens and two bank-select
tokens, replacing 16 program tokens per instruction. The merge adds
17 layers and uses width 6,144, with the same 39,609,437,056-byte
weight size as the looping merge. The longest A100 case completed
without the allocator warning observed in the prior run.

Reports are `results/resident_rom_mps.json`,
`results/resident_rom_9b_cuda_fla.json`,
`results/resident_rom_launch_mps.json`, and
`results/resident_rom_launch_9b_cuda_fla.json`.

## Resident warp register banks

All twelve fixtures match across standalone MPS and merged Qwen3.5-9B
on A100, covering 1,229 instructions and 1,241 scheduler decisions.
Each fixture preserves its complete canonical trace relative to the
resident-ROM checkpoint. Both warp banks, both ROMs, and shared
memory were checked directly in each launch's single cache. The JSON
launcher reproduced the cooperative fixture on both platforms. All
25 local tests passed, and the seven earlier compiler graphs were
unchanged.

The model tags each update with its selected bank. The host forwards
the full 64-nibble update packet and uses a separate copy for trace
comparison. Bank initialization adds 128 input tokens to each
launch. The cooperative kernel uses 23,112 input tokens and relays
8,448 state nibbles. Comparing consecutive states in its trace,
1,512 of those nibbles change. That count is a target for the next
optimization. The merge adds 17 layers, uses width 6,400, and contains
41,259,716,992 bytes of weights.

Reports are `results/resident_state_mps.json`,
`results/resident_state_9b_cuda_fla.json`,
`results/resident_state_launch_mps.json`, and
`results/resident_state_launch_9b_cuda_fla.json`.

## Changed-only warp state updates

All twelve fixtures match across standalone MPS and merged Qwen3.5-9B
on A100, covering 1,229 instructions and 1,241 scheduler decisions.
The canonical traces match the full-packet checkpoint. The verifier
checked each emitted delta against the model snapshot and checked the
final warp banks, ROMs, and shared memory. Independent counts from
the full-packet traces agree with the emitted update counts, including
1,238 changes to zero. The JSON launcher reproduced the cooperative
fixture on both platforms. All 25 local tests passed.

| Fixture | Prior state updates | Changed-only state updates | Prior input tokens | Changed-only input tokens |
| --- | ---: | ---: | ---: | ---: |
| Seeded dense multiply | 7,424 | 2,052 | 20,424 | 15,052 |
| Cooperative multiply and gather | 8,448 | 1,512 | 23,112 | 16,176 |
| 256-instruction budget exhaustion | 16,384 | 846 | 43,688 | 28,150 |

Each instruction still makes the fixed set of state read requests.
The merge adds 18 layers, uses width 6,400, and contains
41,942,202,112 bytes of weights. The eight earlier compiler graphs were
unchanged. Reports are `results/resident_delta_mps.json`,
`results/resident_delta_9b_cuda_fla.json`,
`results/resident_delta_launch_mps.json`, and
`results/resident_delta_launch_9b_cuda_fla.json`.

## Model-generated state read plans

The state read plan uses eight model-emitted bitmap bytes per
instruction. The launcher requests exactly the changed positions
and still requests all 32 shared-store positions. The verifier
checks the bitmap against the recurrent snapshot and checks the
final state, ROMs, and shared memory directly.

| Fixture | Fixed-request input tokens | Planned input tokens | Fixed-request forward calls | Planned forward calls |
| --- | ---: | ---: | ---: | ---: |
| Seeded dense multiply | 15,052 | 10,608 | 233 | 349 |
| Cooperative multiply and gather | 16,176 | 10,296 | 265 | 397 |
| 256-instruction budget exhaustion | 28,150 | 14,660 | 513 | 769 |

The fixed-request call counts follow from two calls per instruction
and one terminal schedule call. The planned counts are recorded in
the reports. These counts measure transport and model invocations.
Wall-clock throughput was not measured. The merge adds 18 layers,
uses width 6,400, and contains 41,942,202,112 bytes of weights.

All twelve fixtures match on MPS and merged Qwen3.5-9B, covering 1,229
instructions and 1,241 scheduler decisions. The cooperative launcher
also matches on both platforms. All 25 local tests passed. Reports are
`results/resident_plan_mps.json`, `results/resident_plan_9b_cuda_fla.json`,
`results/resident_plan_launch_mps.json`, and
`results/resident_plan_launch_9b_cuda_fla.json`.

## Model-generated state and store read plans

The resident I/O plan adds one store-lane bitmap byte per instruction.
It requests four nibbles for each selected lane and skips the
follow-up when both the state and store masks are empty. Independent
counts from the prior traces agree with every skipped call. Every
requested store nibble corresponds to an emitted write.

| Fixture | State-only plan input tokens | State and store plan input tokens | State-only forward calls | State and store forward calls |
| --- | ---: | ---: | ---: | ---: |
| Seeded dense multiply | 10,608 | 7,268 | 349 | 313 |
| Cooperative multiply and gather | 10,296 | 6,460 | 397 | 373 |
| 256-instruction budget exhaustion | 14,660 | 6,724 | 769 | 683 |

These counts measure transport and model invocations. The timing
comparison follows below. The merge adds 18 layers, uses width
6,400, and contains 41,942,202,112 bytes of weights. Short, single-token
follow-ups exercise the recurrent kernel. On MPS this uses the
PyTorch fallback.

All twelve fixtures match on MPS and merged Qwen3.5-9B, covering 1,229
instructions and 1,241 scheduler decisions. The cooperative launcher
also matches on both platforms. All 25 local tests passed, and the ten
earlier compiler graphs were unchanged. Reports are
`results/resident_io_mps.json`, `results/resident_io_9b_cuda_fla.json`,
`results/resident_io_launch_mps.json`, and
`results/resident_io_launch_9b_cuda_fla.json`.

## Same-checkpoint transport benchmark

Fixed requests gave the lowest latency for the cooperative kernel on
this A100. The state and store plan used the fewest tokens and the
lowest peak allocated memory. All 18 timed launches and three
profiled warmups matched the reference trace and frozen output
matrix. There were zero failures. The six timed samples per mode
covered every order permutation on the same loaded checkpoint.

| Transport | Median launch time | Range | Input tokens | Forward calls | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed requests | 27.179 s | 27.082–27.208 s | 16,176 | 265 | 39.917 |
| State-only plan | 39.599 s | 39.499–39.729 s | 10,296 | 397 | 39.638 |
| State and store plan | 37.201 s | 37.076–37.357 s | 6,460 | 373 | 39.462 |

Each peak allocation was identical across the six samples for its
mode. The profiled warmups spent about 88–89% of wall time inside
CUDA model-forward intervals. In the timed launches, elapsed time
per forward averaged about 0.10 seconds across modes. The increased
number of forwards explains the observed direction of the latency
tradeoff. The environment used FLA and reference causal convolution.
The timer includes host relay and cache verification and excludes
checkpoint loading and reference generation.

The result applies to the cooperative fixture on one A100 with this
software stack. Use fixed requests as the latency baseline for that
configuration. The launcher retains its layout-based defaults.
Reports are `results/resident_transport_benchmark_a100.json` and
`results/resident_transport_benchmark_host.txt`.

## Explicit transport selection and store-and-reload benchmark

All three explicit launcher choices matched the 48-instruction
store-and-reload trace on MPS and merged 9B. The implicit MPS I/O
launch produced the same result as the explicit state and store plan
after removing the mode label. An override on an earlier layout was
rejected. All 25 local tests passed. The selector evidence is
`results/resident_transport_launcher_store_reload.json`.

Fixed requests also gave the lowest latency on this write-heavier
fixture. It emits the same 256 store nibbles as the cooperative
fixture in 48 instructions rather than 132. All 18 timed launches
and three warmups matched exactly on the same loaded checkpoint.

| Transport | Median launch time | Range | Input tokens | Forward calls | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed requests | 10.958 s | 10.941–11.001 s | 6,451 | 97 | 39.468 |
| State-only plan | 15.479 s | 15.443–15.521 s | 4,286 | 145 | 39.363 |
| State and store plan | 14.743 s | 14.721–14.766 s | 3,054 | 137 | 39.304 |

Peak allocation was constant per mode. Profiled warmups spent
70–77% of their wall time inside CUDA model-forward intervals.
Timing trials were interleaved on one A100 with FLA and reference
causal convolution. The host reported a 12.39 one-minute load
average with eight available cores at launch. The benchmark
process was the only visible CPU-intensive process during sampling.

Use fixed requests as the latency baseline for both measured A100
fixtures. The state and store plan has the lowest token count and
peak allocation on both. Existing layout-based defaults are
preserved. Reports are `results/store_reload_transport_benchmark_a100.json`
and `results/store_reload_transport_benchmark_host.txt`.

## Programmable resident row-pair extent

The standalone resident grid checkpoint passed all 15 fixtures on
MPS: 1,403 instructions and 1,418 scheduler decisions matched the
independent interpreter. The 12 full-size fixture reports matched the
prior resident I/O reports exactly apart from the one launch token and
two grid transport fields. In every launch the verifier read the grid,
shared memory, program ROM, and warp registers directly from the same
inference cache.

| Row pairs | Rows scheduled | Instructions | Scheduler decisions |
| ---: | ---: | ---: | ---: |
| 1 | 2 | 29 | 30 |
| 2 | 4 | 58 | 59 |
| 3 | 6 | 87 | 88 |

The three partial fixtures also matched the scalar modular matrix
product for the scheduled rows and preserved all seeded values in
the later rows. The MPS graph uses 17 layers, 155 total heads, and
3,888 dimensions, compared with 17, 154, and 3,885 for resident I/O.

The initial schedule-only grid read failed the two-HALT diagnostic:
it ran seven instructions where the interpreter expected eight.
Both reset tokens also needed to read the grid. With that change,
all eight instructions and nine scheduler decisions matched.
Both reports and the diagnostic manifest are saved. All 25 unit
tests passed. The launcher rejected an extent below four on the
previous layout before loading a model.

The merged Qwen3.5-9B build passed the same 15 fixtures on one
A100 SXM 80 GB. All case records, including cache checks and
transport counters, are identical across MPS and CUDA. The merge has
18 chip layers, hidden size 6,400, and 41,942,202,112 tensor bytes.
The CUDA environment used Torch 2.14.1, Transformers 5.19.0, and FLA
0.6.0 with reference causal convolution. The model and compiler
revisions and device details are in `results/resident_grid_host_a100.txt`.

The standalone launcher matched the suite's one-pair result with its
implicit transport. All three explicit transports matched that same
canonical trace on merged 9B, with exact grid, memory, ROM, and state
cache checks. Fixed requests used 4,290 input tokens, the state plan
used 3,195, and the state and store plan used 2,360. These are token
counts for one fixture. See the two suite reports and four launcher
reports under `results/resident_grid*`, and run
`python3 scripts/check_resident_grid.py` to compare them.

## Resident row windows

The launcher seeds the start in the initial scheduling ticket and
puts the inclusive end in the existing grid token. The compiler
and merge recipe are unchanged. The model still chooses each
warp, releases barriers, advances the row pair, and signals halt.

The MPS suite matched the interpreter on the full tile and four
windows, covering 290 instructions and 295 scheduler decisions.
The full tile report matched the prior grid report exactly.
Each window matched the full tile in its scheduled rows and
preserved its initial C values everywhere else. Stitching the
front and back windows reproduced all 64 outputs of the full tile.
The 25 existing tests passed. The parser accepted all ten legal
windows and rejected out-of-range, fractional, and boolean inputs.

All five MPS case records are identical to the merged 9B records
on an A100 SXM 80 GB, including schedules and recurrent cache checks.
The final-pair launch through the JSON CLI ran 29 instructions and
matched the evaluator with the fixed transport. Omitting the length
resolved to one pair for that launch. The A100 environment and the
source hashes for the unchanged compiler and merge code are in
`results/resident_windows_host_a100.txt`. The exact reports can be
compared with `python3 scripts/check_resident_windows.py`.

## Composed 16×16 modular matrix multiply

The launcher uses one loaded grid checkpoint and eight separate
launch caches. It packs four 8×8 output tiles for each of two K slices
and uses the first slice's model outputs as C for the second.
The model performs all multiply-accumulates modulo 65536.

The MPS run matched all 256 scalar results after each slice.
In the frozen input, 256 entries cross the modular boundary in
the first pass and 253 in the second. Each tile ran 116 instructions
and 117 scheduler decisions, for 928 instructions and 936 decisions
in total. Each launch passed direct checks of its grid, shared
memory, program ROM, and warp state. The 25 existing tests passed.
The report stores both full matrices and the exact trace digest
and transport counters for each launch.

Merged Qwen3.5-9B on one A100 SXM 80 GB matched every entry after
each K slice. The two reports have identical per-tile trace digests
and transport records for all eight launches. The merge metadata
matches the earlier resident grid checkpoint. The model and compiler
revisions, host details, and source hashes are saved in
`results/resident_matmul16_host_a100.txt`. Run
`python3 scripts/check_matmul16.py` to reproduce the comparisons.

## Masked 32-bit lane addition

The standalone ADD32 prototype matched all 12 frozen cases on MPS.
The checks cover 384 emitted byte tokens and 96 cached lane results.
The vectors exercise carry across bit 15, wrap at bit 31, all lanes
inactive, alternating masks, and each single-lane mask. The verifier
also checked the loaded operands and mask directly in the cache.
Each case uses one inference cache, 98 input tokens, and two forwards.

The standalone circuit has 14 layers, 73 heads, and 1,722 dimensions.
The existing carry decoder now handles any whole number of nibbles.
All 25 unit tests passed. The dense resident 8×8 regression matched
the prior circuit statistics and its exact 116-instruction trace and
117 scheduler decisions. See `results/add32_resident_regression_mps.json`.

Standalone CUDA and merged Qwen3.5-9B on one A100 SXM 80 GB passed
all 12 cases. Case records are identical across the three runs.
The merge uses 15 chip layers, hidden size 4,864, and 29,683,094,144
tensor bytes. The pinned environment and source hashes are saved
in `results/add32_host_a100.txt`. The resident integration is tracked
separately in SCOPE.md.

## High register banks in the resident grid

The resident wide layout stores 32 more nibbles per warp. It uploads
192 state tokens for the two warps and expands the state read plan
from eight to twelve bytes. At each tick the model selects the high
half from the scheduled warp and includes it in the snapshot. The
current instruction set writes the low half and preserves the high
half. Both banks are verified directly in the inference cache.

MPS passed the three frozen cases, covering 274 instructions and
277 scheduler decisions. The cases include a seeded masked kernel
with barriers, a seeded cooperative multiply and gather with loops,
and the dense matrix multiply. The high halves in every snapshot
match the scheduled warp's seed. The dense and cooperative low
halves, schedules, and shared-memory results match the prior grid
reports. All 25 existing unit tests passed.

The first two wide builds exceeded the standalone FFN limit of
3,584 units per layer. Their exit layers required 3,776 and 3,603
units after compiler cleanup. The final layout emits control and
plan tokens in an earlier FFN and fits at 19 circuit layers, 189
heads, and 4,308 dimensions. The prior grid circuit's generated
nodes are identical to the previous build.

The merged Qwen3.5-9B checkpoint on one A100 SXM 80 GB passed the
same three cases. Their case records are identical to MPS. On both
devices, fixed requests and the state-only plan reproduced the
seeded kernel's schedules, snapshots, shared memory, and final
32-bit register words. The prior grid layout rejects the seeded
manifest before model load. The wide merge uses 20 chip layers,
hidden size 6,656, and 45,039,342,592 tensor bytes. See the host
record and run `python3 scripts/check_resident_wide.py` to compare
the reports.

## Resident ADD32I with cooperative tiled execution

The resident ADD32 layout accepts an unsigned byte immediate from
ROM and adds it to each active 32-bit register modulo 2³². The ROM
uses a 13-bit payload for the new opcode. The upload tokens use
reserved immediate encodings and preserve the existing ROM stride.
All 256 immediate values round-trip through the encoder and decoder.

MPS passed four frozen cases with 405 instructions and 409
scheduler decisions. The masked case covers carry, full-word wrap,
the zero immediate, preservation on inactive lanes, and an
interleaved 16-bit add. It emitted 18 high-register nibble updates.
The mixed cooperative case emitted 17 high-register updates in
the same launch as its tiled multiply, barrier, indexed gather,
and divergent loop. Its matrix matches the previous cooperative
result. Both inherited tile cases matched their earlier canonical
traces and emitted zero high-register updates. All 25 unit tests
passed. The prior wide layout's generated circuit nodes are
identical to the earlier build.

Merged Qwen3.5-9B on one A100 SXM 80 GB matched all four MPS case
records, including every snapshot and high-register update count.
Fixed requests, the state-only plan, and the state and store plan
all matched on the masked carry kernel. The verifier counted 18
changed high nibbles from its snapshots, and the model emitted
exactly 18 high-register update tokens in each transport. The
earlier wide layout rejects ADD32I before model load. The new merge
uses 22 chip layers, hidden size 6,656, and 46,458,906,368 tensor bytes.
Run `python3 scripts/check_resident_add32.py` to compare the saved
evidence.

## Composed 16×16 multiply on the resident ADD32 checkpoint

The 16×16 runner now accepts the resident ADD32 layout. On MPS,
both accumulation passes matched all 256 scalar outputs and the
earlier grid run. All eight tile traces matched the grid traces
after projecting the snapshots onto the shared fields. The report
also retains the full ADD32 trace digests and direct cache checks.
The eight launches ran 928 instructions and 936 scheduler decisions
with zero high-half writes. The multiply uses the existing modulo
65536 tile arithmetic.

Fixed requests on the ADD32 layout add 64 seed tokens and 32
high-half read requests per instruction, for 3,776 extra input
tokens per tile. All 25 existing tests passed. The CLI rejected an
explicit ADD32 request with grid-only metadata before model load.

Merged Qwen3.5-9B on one A100 SXM 80 GB matched the same scalar
and grid results. Its full per-tile trace hashes, cache checks,
and transport counts are identical to MPS. The 140-instruction
cooperative ADD32I fixture also passed on the exact merged directory
used for the multiply, with an identical case record to the previous
ADD32 verification. The three reports and host record are under
`results/resident_add32_matmul16*`.

## Full inherited qualification on the resident ADD32 checkpoint

One loaded ADD32 model ran all 23 frozen cases on MPS, with a fresh
cache per case. All 1,892 instructions and 1,915 scheduler decisions
matched the independent interpreter. The suite includes the dense
and cooperative tiles, indexed loads and collision-aware stores,
masks, nested branches and loops, three partial extents, four row
windows, and both new ADD32I kernels. It produced 21 halts, one
deliberate deadlock, and one deliberate step limit.

Nineteen grid and row-window cases match the prior traces after
projecting the snapshots onto the original fields. The two seeded
wide cases and two ADD32I cases match their earlier full snapshots.
The verifier also checked the transport counters and direct reads
of grid, memory, ROM, and warp state for every launch. See
`results/resident_add32_full_mps.json`.

Merged Qwen3.5-9B on one A100 SXM 80 GB matched all 23 MPS
case records, including schedules, register and shared-memory
snapshots, transport counters, and terminal outcomes. The merge
metadata matches the pinned resident ADD32 build. The CUDA report
and host record are under `results/resident_add32_full*`. Run
`python3 scripts/check_resident_add32_full.py` to verify the
cross-device and inherited comparisons.

## Fresh-cache text on the unified ADD32 merge

Base Qwen3.5-9B and the qualified ADD32 merge used the same greedy
generation settings on one A100 SXM 80 GB. The frozen corpus has
24 prompts from 21 to 691 input tokens. Both tokenizers returned
identical input IDs. The tokenizer data, configuration, and chat
template files were byte-identical. Transformers emitted a tokenizer
regex warning when loading the merged directory.

The next token matched on all 24 prompts. The greedy continuation
matched on 23 of 24 prompts at a limit of 16 new tokens. The Moon
prompt matched for its first two generated tokens and diverged at
token three. The merge emitted zero chip tokens. Over the original
vocabulary, the median per-prompt maximum absolute next-token logit
difference was 0.2265625 and the largest was 0.421875.

These results cover the frozen corpus with fresh caches. See
`results/resident_add32_text_9b_cuda_fla.json` for both token sequences,
per-prompt logit differences, and the generation configuration.
The GPU-to-text cache comparison is the next gate.

## Direct GPU-to-text cache handoff

The unified merge replayed the cooperative ADD32I kernel four times
on one A100 SXM 80 GB. Each replay matched the prior 140-instruction
trace and used 7,549 input tokens in one cache. The harness appended
one frozen text prompt to each final cache. It also evaluated every
prompt in a fresh cache at position zero and at position 7,549.
All four fresh continuations match the earlier merged text report.

All four native next tokens matched across the three conditions.
Direct reuse matched three of four greedy continuations. The Moon
prompt matched for two generated tokens and diverged on token three.
Shifting positions alone kept all four continuations. No chip tokens
appeared in any of the twelve continuations.

| Frozen prompt | GPU cache vs fresh maximum logit difference | Shifted positions vs fresh maximum logit difference |
| --- | ---: | ---: |
| Moon fact | 0.375 | 0.1328125 |
| Python list reversal | 0.78125 | 0.1640625 |
| Multilingual copy | 0.404296875 | 0.25 |
| Long register log | 1.125 | 0.2841796875 |

The logit differences use the original vocabulary. The four cases
cover direct reuse of this kernel's cache. See
`results/resident_add32_handoff_9b_cuda_fla.json` for all cache
lengths, token sequences, and comparison records.

## Text cache reset with resident GPU state

The reset replaced the pretrained layers' cache entries and kept
the GPU layers in the same dynamic cache. Text began at position
zero. All four cooperative kernel replays matched the direct
handoff report, including the 7,549 input tokens and trace digest.

All four greedy continuations and all native next-token logits
matched the fresh-cache control exactly. This includes the Moon
prompt that changed with direct cache reuse. The check compared
128 register nibbles, 4,096 indexed memory nibbles across the lane
banks, and 512 nibbles in the tiled memory view per prompt. All
18,944 comparisons matched after text generation. The report
records the hashes before and after for each bank.

The measured boundary starts a new text context while preserving
the tested GPU rows. See `results/resident_add32_handoff_reset*`
for the A100 report and host details.

## Relaunch a consumer on retained GPU memory

The round-trip harness runs the cooperative ADD32I producer, starts
a new text context on the same cache, and runs a consumer kernel
after text generation. The consumer uploads control, ROM, and warp
state. It reads the first two produced rows through indexed memory
and stores them into the final row pair. A forward hook checks the
cache identity and counts the memory tokens actually sent.

A standalone MPS smoke check passed the two GPU launches without
text between them. Its ten consumer instructions matched the
interpreter, with zero initial memory tokens and zero new caches
on the second launch. All 25 existing tests passed.

The merged 9B round trip on one A100 reproduced the prior producer
trace and exact text. The ten consumer instructions and eleven
scheduler decisions matched the interpreter and the fresh-cache
control. All sixteen copied values came from producer-written
addresses and differed from their destination seeds. The audit
checked the same cache across all 27 consumer calls. It counted
zero initial memory tokens and 64 memory tokens for the consumer's
own stores. The retained launch used 761 input tokens, compared
with 1,273 for the fresh-cache control.

See `results/resident_add32_cache_relaunch*` for the trace, audited
token counts, merge metadata, and host details.

## Two K slices with resident C

The standalone MPS check accumulated the two K slices of the
top-left 8×8 output tile in one cache. The first launch uploaded
A, B, and C. The second launch reused the same cache and uploaded
only new A and B. Both passes matched the scalar reference and
the tile trace digests from the earlier composed 16×16 multiply.
Each pass ran 116 instructions and 117 scheduler decisions.

The forward audit counted 256 operand tokens and zero initial C
tokens on the second pass. Its 256 C tokens were the model's own
stores. The retained pass used 7,677 input tokens, compared with
7,933 for the exact fresh-cache control. All 25 existing tests
passed. The MPS report includes each forward's memory token counts.

The merged Qwen3.5-9B A100 run matched both MPS pass records and
the fresh-cache control exactly, including all forward token counts
and trace digests. Both devices used a single cache for the retained
pair. The control used a separate cache and reproduced the second
slice's result and trace. See `results/resident_add32_cached_tile*`
for the reports and host details, and `REPRODUCE.md` for the commands.

## Retained C across the full 16×16 multiply

On standalone MPS, the ADD32 launcher's `--retain-c` mode matched
the independent scalar reference at both K slices. The eight
launches reproduced the original 928 instructions, 936 scheduler
decisions, and all eight trace digests. Each pair used one new
cache for its first launch and reused it for its second launch.
The forward audit counted zero C upload tokens in all four second
launches, omitting 1,024 C upload tokens across the multiply.

The retained mode used 61,340 input tokens and the previous
fixed-transport mode used 150,626. The retained mode also uses
model-generated read plans, so this total difference includes
changes beyond the C uploads. See
`results/resident_add32_matmul16_retained_mps.json` for the audited
launches.

The merged ADD32 checkpoint on one A100 matched the complete MPS
pass records, including all cache and transport counters. The
runner finished in 299 seconds from model load through report
write on this host. That elapsed time is for the audited run.
The fixed-transport report comes from an earlier run. See
`results/resident_add32_matmul16_retained_9b_cuda_fla.json` and
`results/resident_add32_matmul16_retained_host_a100.txt` for the
CUDA records and `REPRODUCE.md` for commands.

## 32-bit indexed exchange between warps

The shared32 layout pairs the existing low indexed operation at
`index` with a high-half operation at `index XOR 1`. Each limb is
one 16-bit memory operation. The producer executes both stores
before the shared barrier, then the consumer reads both halves.
The layout still addresses 128 shared words.

The frozen case runs 20 instructions and 21 scheduler decisions.
It exercises carry, 32-bit wrap, both shared-memory regions, the
odd address 127, and one inactive producer lane. Two active lanes
compute different values for the same pair. The lower-numbered
lane wins both stores, and both consumer lanes read its result.
The checker independently verifies both final register banks,
all 128 words, and the two six-lane scatter grants.

On MPS, all 24 cases match the interpreter. The 23 inherited
case records match the original ADD32 MPS suite in full. All 25
local tests passed.

The pinned merged Qwen3.5-9B on A100 reproduces all 24 MPS case
records, including 1,912 instructions, 1,936 scheduler decisions,
and exact recurrent ROM, shared memory, warp and grid state.
The merge uses 22 chip layers and hidden width 6,656, matching
the previous resident ADD32 dimensions. See the two
`results/resident_shared32_full_*` reports and `REPRODUCE.md`.

The public MPS launcher reproduced the handoff's complete trace.
The retained 16×16 launcher passed on both backends with all 928
instructions, 936 decisions, and memory audits identical to the
prior ADD32 runs. It still uses four caches across eight launches
and omits 1,024 initial C tokens. The merged shared32 checkpoint
also reproduced the two-slice tile and GPU-to-text-to-GPU cache
replays. The reset text output matched that checkpoint's own
fresh-cache control in the relaunch fixture. See the saved
`results/resident_shared32_*` reports and A100 host record.

## Full-width addition from indexed shared memory

The addmem32 layout uses the same shared words, warps and register
banks as shared32. The left pair is loaded at the saved index. The
two new instructions add the right pair at the saved index XOR an
even immediate. Carry from the low addition increments the high
register before the high addition. The high addition wraps modulo
2¹⁶, completing the sum modulo 2³².

The 15-instruction fixture uses arbitrary full-width operands,
a pair mask of 126, an odd base, and an inactive producer lane.
It checks both register banks and all 128 shared words against
independent scalar sums. The consumer loads the result after the
producer's barrier, including the original value in the masked
lane. All 128 valid opcode/immediate encodings checked out, and
five invalid mask values were rejected.

On MPS the 25-case run matches the interpreter, with 1,927
instructions and 1,952 scheduler decisions. All 24 inherited
records match the shared32 baseline. All 25 local tests passed.
The pinned merged Qwen3.5-9B on A100 reproduced all 25 MPS
records exactly. The checkpoint has 28 chip layers at hidden
width 6,656, compared with 22 layers at the same width for
shared32. The public MPS launcher reproduced the cooperative
arithmetic trace. On both backends, the retained 16×16 launcher
reproduced all 928 instructions, 936 decisions, both scalar
matrices, and all audited memory counts from the ADD32 baseline.
It still uses four caches and omits 1,024 C upload tokens. See
`results/resident_addmem32_*` and the A100 host record.

## Native text and cache handoff on addmem32

One H100 ran the 28-layer addmem32 merge and the earlier 22-layer
ADD32 control against the same pinned base. Both merge reports
are identical across the 24 frozen prompts, including all
generated tokens and reported logit metrics. Both match all 24
base first tokens and 21 of 24 greedy continuations. The three
differences start after 3, 3, and 11 matching generated tokens.
The median per-prompt maximum next-token logit difference from
base is 0.2265625 and the maximum is 0.46875. Both corpus runs
emit zero chip tokens. The prior A100 ADD32 run
matched 23 of 24 continuations against its base run.

On addmem32, all four text resets reproduced fresh-cache logits
and continuations exactly and preserved all 18,944 inspected GPU
nibbles. The GPU-to-text-to-GPU replay reproduced the inherited
producer and consumer traces. The ten-instruction consumer
copied 16 producer-written values without uploading initial
shared memory, saving 512 tokens against the cold control.

See `results/resident_addmem32_text_*`,
`results/resident_addmem32_handoff_reset_*`,
`results/resident_addmem32_cache_relaunch_*`, and
`results/resident_add32_text_9b_cuda_h100_control.json`.

## Full-width arithmetic through a text handoff

The same addmem32 checkpoint runs the 15-instruction arithmetic
producer and a separate 20-instruction consumer. The consumer's
two warps copy eight full-width values to disjoint shared pairs.
Seven values are producer sums. The eighth is the original pair
from the masked producer lane. A scalar check validates the
producer and all 128 final shared words, including sources and
destination pairs.

Standalone MPS runs the two GPU launches. Merged Qwen3.5-9B on
A100 generates text between them and matches its fresh-cache
logits and greedy continuation exactly. The GPU records match
MPS and the cold control. The retained consumer uses 1,003 input
tokens against 1,515 for the cold control. The forward audit
counts zero initial shared-memory tokens and 64 store tokens.
The legacy relaunch also matches its prior GPU records.
All 25 local tests passed.

See `results/resident_addmem32_arithmetic_relaunch_*` and
`results/resident_addmem32_cache_relaunch_legacy_replay_9b_cuda_fla.json`.

## Divergence on the high bit of a 32-bit sum

The branch32 layout extends `IF_BIT` to all 32 accumulator bits
and uses the existing branch stack. The frozen producer adds two
32-bit operands per lane. The active lanes have identical low
halves at the branch. Bit 31 selects the then and else masks, and
the next instruction after reconvergence executes under the
original parent mask. A second warp reads both halves after the
barrier, including the original word in the inactive lane.

The 21-instruction case checks the branch and reconvergence masks,
both register banks, and all 128 shared words against independent
scalar results. All 32 encodings were checked, and the earlier
layout accepts only bits 0 through 15. The standalone MPS run
matches all 26 cases and all 25 inherited case records, with
1,948 instructions and 1,974 scheduler decisions. All 25 local
tests passed.

The pinned merged Qwen3.5-9B on H100 NVL reproduces every MPS
case record, including the generated masks, register snapshots,
scheduler decisions, and final memory. The merge uses 28 chip
layers and hidden width 6,656, matching the addmem32 dimensions.
The public MPS launcher reproduces the high-bit branch trace.
Both retained 16×16 runs match the prior ADD32 matrices, all 928
instructions, all 936 scheduler decisions, and all memory counts.
They use four caches and omit 1,024 C upload tokens. See
`results/resident_branch32_*` and the H100 NVL host record.

## Upper-half loops driven by lane-specific carry

The loop32 layout extends `WHILE_BIT` to all 32 register bits and
uses the existing loop stack and PC link rules. The frozen case
tests bit 16, increments with `ADD32I 1`, and exits after carry
clears that bit. The active lanes take zero to three trips.
One lane wraps from 0xffffffff to zero. A masked lane retains
its seed, and a second warp reads both halves after the barrier.

The independent checker verifies each trip count and execution
mask, both register banks, and all 128 shared words. All 224
high-half bit and exit-PC combinations decode to their expected
payloads. MPS passes all 27 cases and matches all 26 inherited
records, with 1,972 instructions and 1,999 scheduler decisions.
All 25 local tests passed.

The pinned merged Qwen3.5-9B on H100 reproduces all 27 MPS
records exactly. The merge keeps 28 chip layers and hidden
width 6,656. The public MPS launcher reproduces the upper-half
loop trace. The retained 16×16 runs on both backends match the
prior ADD32 matrices, 928 instructions, 936 scheduler decisions,
and all audited memory counts. They use four caches and omit
1,024 C upload tokens. See `results/resident_loop32_*` and the
H100 host record.

## Two preloaded operand banks in one cache

The tilebanks layout keeps both A/B operand pairs and partial C
in one cache. Both banks are written before the first K slice.
A bit in the launch grid record selects the bank for tile MACs.
Indexed shared-memory accesses keep their original address map.

MPS and merged Qwen3.5-9B on H100 match the scalar result and the
prior trace digest for both slices. The physical cache check
verifies both banks and the selector after each pass. The first
launch uploads 768 memory tokens, 256 for each operand bank and
256 for C. The second launch uploads zero initial memory tokens.
Both launches still relay the model's C stores.

The second launch uses 7,421 input tokens against 7,933 for the
fresh-cache control. Preloading moves 256 operand tokens from the
second launch to the first, so the pair uses the same total
number of tokens as the earlier retained-C experiment. See
`results/resident_tilebanks_cached_tile_*` for the audited calls.
All 27 inherited case records match loop32 on MPS and merged
9B, with 1,972 instructions and 1,999 scheduler decisions. The
legacy retained-C launcher reproduced both passes and the cold
control. All 25 local tests passed. The new merge keeps 28 chip
layers and hidden width 6,656. See the full reports and H100
host record under `results/resident_tilebanks_*`.

## One scheduled launch for both K slices

The tilepipe layout emits a distinct ticket after both warps halt
on the last row pair of slice zero. Relaying that ticket selects
bank one inside the model and restarts execution at row zero.
The same ROM, memory, and warp state remain in the cache.

MPS produces both exact scalar matrices across 232 instructions
and 233 scheduler decisions. The audit records one grid setup,
one relayed transition, and no later operand uploads. The first
slice's normalized trace matches the previous run. In the second
slice, the initial no-op on warp one retains its previous
accumulator until `LOAD_TILE` replaces it. This changes one
instruction record's register and snapshot fields relative to
a separate launch. All actual records match the continuous
interpreter reference.

Merged Qwen3.5-9B on A100 reproduces the raw trace, transition,
physical cache checks, and every audit entry. The one-launch
path uses 15,233 input tokens against 15,462 for two launches
with the same two preloaded banks. The shared memory upload
count is the same. The one-launch path uploads the grid, ROM,
and warp setup once. Both backends reproduce all 27 inherited
case records from tilebanks. All 25 local tests passed.
See `results/resident_tilepipe_*` for the reports and host record.

## Four model-scheduled launches for the full 16×16 multiply

The public `--schedule-k` mode uses one launch and cache for each
8×8 output tile. The model transitions between preloaded banks
inside each launch. The MPS run matches both scalar matrices. The
checker verifies all four transition tickets and the grid, ROM,
warp and memory upload counts for each tile. The four launches
execute 928 instructions and 932 scheduler decisions.

The MPS and merged 9B reports match both pass matrices and every
tile record. They use 60,408 input tokens, compared with 61,340
in the earlier eight-launch retained-C mode. The legacy
single-tile evaluator reproduces its saved MPS report after
the launcher change. All 25 local tests passed.

## Native text and cache handoff on the final tilepipe merge

The pinned 28-layer tilepipe merge and base model were compared on
one A100 across 24 frozen prompts. The merge matches all 24 first
tokens and 23 greedy continuations, with zero chip tokens. The
diverging continuation matches its first two generated tokens.
The median per-prompt maximum next-token logit difference is
0.2265625 and the maximum is 0.421875.

All four text resets match their fresh-cache logits and
continuations exactly, preserving all 18,944 inspected GPU
nibbles. The full-width arithmetic relaunch also reproduces
the inherited GPU trace, copies eight pairs after text, and
uploads zero initial shared-memory tokens. These checks use
the same tilepipe merge as the public multiply and the 27-case
GPU suite. See `results/resident_tilepipe_*` and `REPRODUCE.md`.

## Two model-selected vector workgroups in one cache

Standalone Qwen3.5 on MPS passed all six lengths (0, 1, 8, 9, 13, and
16) using the pinned compiler. Each case started with one upload of
its two inputs, output guard image, and length descriptor. The model
emitted the workgroup tickets, selected addresses from the group bit
and lane index, and stored modular 32-bit sums in the same forward call
that computed them. The host fed each generated ticket back unchanged
with a fixed step token.

The output tokens and all 192 recurrent memory bytes matched the scalar
reference per case. This includes wraparound, carry out of the low
16 bits, and unchanged inactive output words. A step after the terminal
ticket left memory intact. The 13-word case used five forward calls,
including prefill, the terminal probe, and readout. This isolated check
uses a fixed vector operation. Tilepipe integration and merged 9B
qualification are separate gates. See `results/vector_grid_mps.json`.

The tilepipe v1 assembler also launched a two-warp 32-bit producer and
consumer example on MPS. The example branches on bit 16, stores indexed
pairs, synchronizes, and reads them from the other warp. All 16
instructions and every scheduler decision and final state field matched
the independent interpreter. The launch used one cache and passed all
four physical state checks. See `results/assembled_cooperative_pair_mps.json`.

## Workgroup scheduling in the two-warp tile circuit

The standalone tilegrid circuit on MPS ran 116 instructions and 117
scheduling decisions in a single cache. The model emitted both bank
changes and the workgroup transition. At the workgroup boundary the
model reset the operand bank, row pair, and both warp PCs, then reused
the uploaded operands, ROM, and resident output tile. The four pass
traces and matrices matched the interpreter. The host audited one
initial memory upload, two bank tickets, and one workgroup ticket.
See `results/tilegrid_row_pair_mps.json`.

The full eight-row run also matched on MPS with 464 instructions
and 465 scheduler decisions in one cache. See
`results/tilegrid_full_mps.json`.

The separate-output variant passed the 116-instruction row-pair
check on MPS. One prefill supplied both initial C images. The
second workgroup loaded and wrote its own C image and left the
first group's result unchanged. Both physical output regions
matched the interpreter at each boundary. See
`results/tilegrid_separate_outputs_mps.json`.

With different B panels, the row-pair check matched the scalar
2×16 active output and kept the six inactive rows intact. The
model read both K slices from the correct panel and wrote only
the selected output region. The host uploaded both B panels once.
See `results/tilegrid_right_panel_row_pair_mps.json`.

The full 8×16 panel also matched on MPS. Four passes executed
464 instructions and 465 decisions in one cache, with one initial
upload and no operand reloads. See
`results/tilegrid_right_panel_full_mps.json`.

The 2×2 variant passed the MPS row-pair check across all four
workgroups. All 232 instructions matched, and the host audited four
bank switches and three workgroup switches in one cache. Both A panels,
both B panels, and the four C images were uploaded in the first call.
The interpreter and scalar check agreed with the model's active
rows and all inactive output rows. See
`results/tilegrid_2x2_row_pair_mps.json`.

The full 16×16 case passed on MPS with 928 instructions and 929
scheduling decisions in one cache. Both K-pass matrices matched
the four-cache baseline. The model loaded the full operand image
once and emitted its three workgroup transitions and four bank
transitions. Inputs fell from 60,408 to 58,693 tokens and forward
calls fell from 2,498 to 2,495. This counts work on the same fixture.
See `results/tilegrid_2x2_full_mps.json`.

The 2,495 calls consist of 929 scheduling calls, 928 instruction
and plan calls, and 638 state and store reads. The runner skipped
290 reads for instructions that produced no state or store packet.
The grid eliminated redundant setup. Per-instruction relay still
accounts for most model calls.

The merged Qwen3.5-9B tilegrid checkpoint on one A100 SXM 80 GB
matched the full MPS result, all eight pass records, transition
tickets, and each audited model call. The run took 6 minutes
45 seconds including model load. CUDA retried three allocations
and completed with exact output. See
`results/tilegrid_2x2_full_9b_cuda_fla.json`.

The public MPS launcher reproduced the qualification report's matrices,
passes, transitions, and call audit. Its result SHA-256 is
`31d7f01824d6b90be7ebd39b74b7912b9410fd5ec109df87286399eff60d4e4b`.

On the same tilegrid merge, the assembled two-warp example matched the
standalone trace and final memory across all 16 instructions. The
24-prompt corpus matched 24 first tokens and 22 greedy continuations,
with zero chip tokens. The Moon and multiplication prompts diverged
after two and three tokens, respectively. Four cache resets matched
their fresh-cache logits and continuations exactly. The 20-instruction arithmetic
consumer relaunched after text with zero shared-memory uploads.
See `results/tilegrid_assembled_9b_cuda_fla.json` and the tilegrid text,
handoff, and relaunch reports.

The same standalone tilegrid circuit matched six seeded programs on
MPS, covering 70 instructions. Two carry loops, two diverging
scatter collisions, and two deliberate barrier deadlocks agreed with
the reference, including physical recurrent state. See
`results/tilegrid_generated_mps.json`.

The tilegrid reduction executed two assembled launches of 32
instructions each on MPS. Two warps reduced 16 unsigned 32-bit inputs
to `0x5f049522` modulo `2^32`. The second launch retained the cache
and uploaded zero input or shared-memory tokens. Each partial sum,
guard, and instruction trace matched the reference. See
`results/tilegrid_reduction_mps.json`.

An independently assembled 13-element vector add ran on the same
standalone tilegrid circuit. All 20 instructions and the complete
shared-memory image matched the interpreter and scalar additions.
The right operand and three inactive output guards stayed intact.
See `results/tilegrid_vector_mps.json`.

The standalone MPS circuit and merged tilegrid checkpoint replayed
all 27 inherited cases exactly. Both sets of records match the
previous suite, including 1,972 instructions, 1,999 scheduling
decisions, and the memory and transport checks. See
`results/tilegrid_inherited_mps.json` and
`results/tilegrid_inherited_9b_cuda_fla.json`.

The assembled vector add, two-pass reduction, and six seeded
programs also matched on the same 9B merge. The second reduction
pass again uploaded zero shared-memory tokens. CUDA matched the
MPS traces, output words, guards, and physical state records.
See the corresponding `results/tilegrid_*_9b_cuda_fla.json` reports
and `results/tilegrid_2x2_host_a100.txt`.

## Programmable workgroup IDs and global memory

The MPS globalgrid circuit matched the independent interpreter for
one, two, three, and four workgroups. It emitted correct lane, thread,
workgroup, and global IDs and stored the 64 possible global IDs at
their computed addresses. The odd-numbered groups read values stored
by the preceding groups. Full recurrent memory, guard words, and
the zero-initialized warp state at each group boundary matched.
See `results/globalgrid_ids_mps.json`.

The MPS reduction summed 32 unsigned 32-bit inputs modulo
`2^32`. Two producer workgroups executed 64 instructions and
stored partial sums `0xadc9c3bd` and `0xddd1129b` at distinct
global addresses. The 12-instruction consumer read both partials
and produced `0x8b9ad658`. It reused the producer cache with zero
global-memory uploads. The scalar reference matched the partials,
final result, and every guard word. The interpreter matched each
instruction and scheduling decision. See
`results/globalgrid_reduction_mps.json`.

The merged globalgrid Qwen3.5-9B on A100 reproduced all four ID
probes and both reduction passes exactly, including recurrent
memory and the zero-upload consumer handoff. The public resident
launcher reproduced the two-workgroup probe on both backends.
The new merge also matched the existing single-workgroup assembly
trace. See `results/globalgrid_reduction_9b_cuda_fla.json`,
the public launcher reports, and `results/globalgrid_host_a100.txt`.
