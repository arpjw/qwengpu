# Validation status

All counts here are from bounded fixtures, not a general reliability claim.
The independent reference machine supports eight lanes, one shared program
counter, nested structured divergence, lane-indexed scratch memory, addition,
XOR, and reduction. The compiled circuit currently supports addition, XOR,
data-dependent masks, one-level divergence, result writeback, and four scratch
bytes per lane. Instruction fetch remains open.

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

`results/textcheck_9b_cuda.json` compares six fixed text prompts between the
base and merged 9B models. All six kept the same next top token, and five kept
the same first four greedy tokens. On the Moon prompt, the third generated
token changed. This does not establish broad language fidelity. Text after a
SIMT trace was not tested; QwenCPU previously needed separate caches for text
handoff, so that remains the starting design for QwenGPU's future text bridge.

The Runpod check used one 96 GB RTX PRO 6000 Blackwell Server Edition pod.
The model outputs were copied into this repository; model weights stayed on
the temporary pod. The pod's live quoted rate was $2.49/hour. Final billed
cost should be checked after teardown.
