---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
library_name: transformers
tags:
- safetensors
- qwengpu
---

# Qwen-GPU

This research checkpoint combines the Qwen3.5-9B text tower with a
compiled SIMT processor. The GPU executes uploaded instructions in
Qwen’s recurrent state. Use the QwenGPU launchers to encode kernel
inputs and validate generated control tokens and memory.

![Execution flow from a host launch through Qwen generation to recurrent registers and memory](docs/media/architecture.png)

## Verified on merged Qwen3.5-9B

![Merged v3 CUDA results: 27 of 27 inherited tile fixtures, 8 of 8 workgroup fixtures, 24 of 24 first tokens, and 21 of 24 full native text continuations](docs/media/validation.png)

The CUDA replay matched 26,225 generated tokens over 1,972 instructions
in the tile suite. The same checkpoint passes 64-element vector add,
128-element two-pass reduction, and 8×4 by 4×4 matrix multiply. The
second reduction launch reuses global RAM with zero re-upload.

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
`environment.txt` records the validation runtime. Set up the compiler
environment from the [Qwen-GPU source repository](https://github.com/arpjw/qwengpu/blob/main/REPRODUCE.md)
and set `checkpoint` to the downloaded snapshot. The merged weights
occupy about 86 GB and were qualified on a 96 GB CUDA GPU.
For a checksum-verified download, use the [pinned checkpoint revision](https://huggingface.co/arpjw/Qwen-GPU/tree/fe671778a2409a218dde757871f33f5ca46d9219).

```bash
python -m qwengpu.run_resident eval/frozen/tile8_gate_v1/dense_seeded.json \
  --model "$checkpoint" --device cuda --check-reference --out result.json
```

The model emitted zero GPU IDs on the frozen native-text corpus.
The v3 Apple MPS replay is pending. The
[release receipt](https://github.com/arpjw/qwengpu/blob/main/RELEASE.md)
identifies the pinned weights and the restore checks.

## Attribution

The underlying Qwen3.5-9B model is copyright 2026 Alibaba Cloud and is
provided under the Apache License 2.0. This checkpoint includes the
modifications described above. See the packaged `LICENSE` and any
accompanying notices.
