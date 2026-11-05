# Validation status

All counts here are from bounded fixtures, not a general reliability claim.
The independent reference machine supports eight lanes, one shared program
counter, nested structured divergence, lane-indexed scratch memory, addition,
XOR, and reduction. The compiled circuit currently supports addition, XOR,
data-dependent masks, one-level divergence, result writeback, and four scratch
bytes per lane. Integrated instruction fetch remains open.

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
| `results/rom_spike_mps.json` | Separate 16-slot stock Qwen3.5 ROM circuit on MPS | 3 programs | 3 exact |
| `results/rom_spike_stress_mps.json` | Same ROM circuit, exploratory contents on MPS | 100 programs | 100 exact |
| `results/example_kernel_mps.json` | JSON kernel runner on standalone Qwen3.5, MPS | 1 program | Exact tokens, scratch, and final lane probe |
| `results/no_else_kernel_mps.json` | JSON kernel runner with IF/ENDIF and no ELSE, MPS | 1 program | Exact tokens, scratch, and final lane probe |
| `results/masked_overwrite_mps.json` | JSON kernel runner with masked overwrite of one scratch slot, MPS | 1 program | Exact tokens, scratch, and final lane probe |

The 9B branch suite compares every generated mask and lane-result token against
the expected execution. The memory suites compare generated lane tokens and
inspect each scratch head's recurrent state after STORE. The mixed program
suite uses the independent reference interpreter for a sequence of ADD, XOR,
divergent STORE, reconvergence, and LOAD operations across three scratch
banks. Result tokens feed
back through the same model cache and update lane register heads. The CUDA checks used
`flash-linear-attention==0.6.0` and `fla-core==0.6.0`; Transformers reported
that causal convolution still used the reference implementation. The 9B
checkpoint was built from Qwen/Qwen3.5-9B revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` with Qwenputer compiler source
at `ca13310e2f9f788d1dd3feeea8213c1b0c23913c`. It is not included here.

The 100-case stress run used the same mixed-program shape with 96 additional
seeded inputs and operands after the 12-case suite. All 100 generated-token
traces and final scratch snapshots matched the reference. This was an
exploratory extension, not a separately held-out test or a statistical
reliability estimate.

The ROM spike is a separate two-head, three-layer circuit. One stock model
object loaded three different 16-slot instruction payload sequences into
fresh caches. Every generated instruction and next-PC token matched, and
the actual recurrent ROM and PC rows contained the expected bits. The
highest 11-bit payload and address 15 were exercised. This does not yet fetch
or execute instructions in the SIMT circuit, and it has not been checked on
CUDA.
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

`results/textcheck_9b_cuda.json` compares six fixed text prompts between the
base and merged 9B models. All six kept the same next top token, and five kept
the same first four greedy tokens. On the Moon prompt, the third generated
token changed. This does not establish broad language fidelity. Text after a
SIMT trace was not tested; QwenCPU previously needed separate caches for text
handoff, so that remains the starting design for QwenGPU's future text bridge.

The Runpod check used one RTX PRO 6000 Blackwell Server Edition pod. The model
outputs were copied into this repository; model weights stayed on the
temporary pod. The pod was terminated and the pod list was empty. Its live
quoted rate was $2.49/hour and it existed for about 53 minutes, implying
roughly $2.20 of GPU time before any billing adjustments. The billing query
returned no records immediately after teardown, so the final charge was not
yet available.
