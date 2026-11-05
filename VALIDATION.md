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
