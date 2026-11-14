---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
library_name: transformers
tags:
- safetensors
- qwengpu
---

# QwenGPU resident generation v3

This research checkpoint combines the Qwen3.5-9B text tower with a
compiled SIMT processor. The GPU executes uploaded instructions in
Qwen’s recurrent state. Use the QwenGPU launchers to encode kernel
inputs and validate generated control tokens and memory.

## Changes from Qwen3.5-9B

- Added 53 compiled circuit layers ahead of the text tower and widened
  the hidden size from 4,096 to 7,936.
- Reserved 212,992 model token IDs starting at 248,320 for the SIMT
  instruction and state protocol. The text tokenizer is from the
  pinned base revision.
- Exported a text-only causal language model with the pretrained untied
  output head and a separate GPU output path.

The model supports two eight-lane warps per workgroup, up to four
sequential workgroups or two interleaved workgroups, and 32 resident
instructions per warp. Its programmable global memory has 512 bytes.
Kernels run bounded integer arithmetic and report halt, deadlock,
or a launcher-imposed instruction limit. Each emitted token takes a
model forward pass.

## Revisions and usage

`build-manifest.json` pins the QwenGPU build, Qwenputer compiler, and
Qwen3.5-9B source revisions and records the shard digests.
`environment.txt` records the validation runtime. Use the QwenGPU
source and commands named in the private project’s release receipt.

```bash
python -m qwengpu.run_resident eval/frozen/tile8_gate_v1/dense_seeded.json \
  --model "$checkpoint" --device cuda --check-reference --out result.json
```

The merged v3 checkpoint matched all 27 inherited tile fixtures and all
eight sequential and interleaved workgroup fixtures on CUDA. The
frozen native-text corpus matched 24 of 24 first tokens and 21 of 24
full greedy continuations of up to 16 tokens, with zero GPU IDs emitted.
Consult the project’s release receipt for validation of the restored
snapshot at its exact revision.

## Attribution

The underlying Qwen3.5-9B model is copyright 2026 Alibaba Cloud and is
provided under the Apache License 2.0. This checkpoint includes the
modifications described above. See the packaged `LICENSE` and any
accompanying notices.
