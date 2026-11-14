---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
library_name: transformers
tags:
- safetensors
- qwengpu
---

# Qwen-GPU v4

Qwen-GPU combines the Qwen3.5-9B text tower with a bounded programmable
SIMT processor compiled into the same weights. A host uploads a kernel
and its initial data. Qwen generation selects warps, executes the
uploaded instructions, and updates registers and memory in its cache.

![Kernel launch, Qwen generation, and recurrent GPU state](architecture.png)

## Verified behavior

![Frozen CUDA qualification results](validation-v4.png)

| Capability | This checkpoint |
| --- | --- |
| Execution | Two eight-lane warps per workgroup; up to four sequential or two interleaved workgroups |
| Programs | 32 instructions per warp, barriers, nested divergence, loops, and bounded termination |
| Memory | 512 bytes of programmable global RAM, with separate workgroup shared and warp private banks |
| Arithmetic | 32-bit integer and address operations, indexed loads and stores, and tiled multiply |
| Native text | The same weights generate ordinary Qwen text; GPU-to-text handoffs retain GPU state |

The rebuilt v4 merge passed 27 inherited tile fixtures, eight workgroup
fixtures, a 64-element vector addition, a 128-element two-pass
reduction, and non-square tiled matrix multiplies on CUDA. Native text
matched 24 of 24 first tokens and 21 of 24 short greedy continuations
against the pinned base. The same merged checkpoint completed the full
16×16 matrix in one generation call: 928 instructions, 20,167 exact
generated tokens, seven panel transitions, and 256 correct output cells.
An independent clean restore reproduced the seeded tile, native text,
and all matrix tokens, panel transitions, and output cells on a second
96 GB CUDA GPU.

[Validation reports](https://github.com/arpjw/qwengpu/tree/main/results/resident_generation_v4_cuda)
and [reproduction commands](https://github.com/arpjw/qwengpu/blob/main/REPRODUCE.md#run-the-stock-generation-checkpoint)
are in the source repository.

## Download and run

The merged weights occupy about 86 GB. CUDA qualification and an
independent clean restore used a 96 GB GPU with the pinned runtime
recorded in `environment.txt`. The v4 package is currently stored in
a private GitHub draft release. The Hugging Face repository still
contains the older v3 checkpoint because the v4 upload hit the
account's private storage limit. Set up the pinned environment in the
[reproduction guide](https://github.com/arpjw/qwengpu/blob/main/REPRODUCE.md#run-the-stock-generation-checkpoint).
From an authenticated QwenGPU source checkout, restore into a
destination outside the checkout that does not already exist:

```bash
checkpoint=../Qwen-GPU-v4
export QWENPUTER_DIR=/path/to/qwenputer/experimental/riscv
export PYTHONPATH="$QWENPUTER_DIR:$PWD"
gh auth token | python3 scripts/restore_private_release.py \
  arpjw/qwengpu v4-private-backup-2026-11-14 445244518 \
  f20d0565c37fb766ce0eb80544c6693fcb20c6075b4987dcd41d35708cd7d213 \
  "$checkpoint"
python3 scripts/check_checkpoint_package.py "$checkpoint" \
  --pinned-manifest results/resident_generation_v4_cuda/manifest.json \
  --expected-checksum-digest f20d0565c37fb766ce0eb80544c6693fcb20c6075b4987dcd41d35708cd7d213
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
  eval/frozen/tile8_gate_v1/dense_seeded.json \
  --model "$checkpoint" --device cuda --check-reference --out tile.json
```

`build-manifest.json` pins the QwenGPU source, Qwenputer compiler, and
Qwen3.5-9B base revisions and hashes every weight shard. `SHA256SUMS`
checks every packaged file. The
[restore receipt](https://github.com/arpjw/qwengpu/blob/main/RELEASE.md#qualified-v4-private-draft-backup)
records the verified backup and CUDA replay. A future Hugging Face
revision must be pinned separately after upload and restore validation.

## Limits

This is a bounded experimental processor. Each generated token needs a
model forward pass, so these programs are slow. The checkpoint has no
demonstrated floating-point GPU arithmetic, atomics, graphics pipeline,
CUDA compatibility, or acceleration. Apple MPS qualification is tracked
separately from the merged CUDA result.

## Changes and attribution

The merge adds 53 compiled circuit layers ahead of the pretrained text
tower, widens hidden size from 4,096 to 7,936, and reserves 212,992
model token IDs starting at 248,320 for the SIMT protocol. It retains
the base text tokenizer and untied output head, with a separate GPU
output path.

The underlying Qwen3.5-9B model is copyright 2026 Alibaba Cloud and
provided under Apache 2.0. This checkpoint includes the modifications
above. Its GPU circuit comes from the MIT-licensed QwenGPU project and
was compiled with the MIT-licensed Qwenputer compiler. See the packaged
[Apache 2.0 license](LICENSE) and the
[source attribution](https://github.com/arpjw/qwengpu/blob/main/RELEASE.md#license-and-attribution).
