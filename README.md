# Qwen-GPU

A bounded, programmable SIMT processor compiled into Qwen3.5. The host
uploads a kernel and its inputs. Qwen selects warps, fetches instructions,
and updates registers and memory through its generated tokens.

![Execution flow from a host launch through Qwen generation to recurrent registers and memory](docs/media/architecture.png)

## What runs today

| Capability | Resident generation v4 |
| --- | --- |
| Execution | Two eight-lane warps per workgroup; four sequential or two interleaved workgroups |
| Programs | 32 instructions per warp, loops, divergent branches, barriers, and deadlock detection |
| Memory | 512 bytes of programmable global RAM, plus scoped shared and private memory |
| Arithmetic | 32-bit integer and address operations, indexed loads and stores, and tiled multiply |
| Matrix | Full 16×16 multiply across eight model-scheduled panels in one generation call |
| Text | The same merged 9B weights produce ordinary text and preserve audited GPU state across text handoffs |

## Validation

![Fixed v4 CUDA results: 27 inherited fixtures, eight workgroup fixtures, 24 first text tokens, 21 continuations, and 256 matrix cells](docs/media/validation-v4.png)

The fixed v4 merge passed the complete CUDA qualification: all 27 inherited
tile fixtures, eight workgroup fixtures, 31 common compute requests, seven
raw memory cases, full-width integer programs, and the native-text gate.
The 16×16 matrix emitted 20,167 exact tokens over 928 instructions and
produced all 256 correct cells in one launch. Native text matched 24 of
24 first tokens and 21 of 24 short greedy continuations, with no GPU
token IDs. [Reports and reproduction commands](VALIDATION.md)

Each generated token requires a model forward pass, so the current
machine is slow. Apple MPS qualification of v4 remains pending and is
separate from the CUDA-qualified v4 release scope. The
private v4 checkpoint package passed its 48-file checksum pass and is
backed up in a private GitHub draft release. A clean download of the frozen
execution package passed
all file and package checks, then reproduced the seeded tile, native text,
and full 16×16 matrix on a separate 96 GB CUDA GPU. The
[release receipt](RELEASE.md) and [restored-run reports](results/resident_generation_v4_cuda/restored-reports)
record the comparison. The previously verified v3 checkpoint remains
available at its pinned revision. The v4 Hugging Face upload is blocked by
the account's private storage limit. A separate clean restore of the
corrected model-card package passed all 49 file checks and the pinned
build-manifest check.
The v4 checkpoint does not implement 32-bit atomic memory operations;
the [atomic instruction gate](ATOMIC32_DESIGN.md) is follow-on work.

## Run the verified v4 checkpoint

The merged weights occupy about 86 GB and were restored and run on a
96 GB CUDA GPU. The corrected v4 package is in a private GitHub draft
release. Set up the [pinned environment](REPRODUCE.md#run-the-stock-generation-checkpoint),
then run these commands from a QwenGPU checkout with `gh` authenticated
for this repository. Choose a destination outside the checkout that
does not already exist:

```bash
checkpoint=../Qwen-GPU-v4
export QWENPUTER_DIR=/path/to/qwenputer/experimental/riscv
export PYTHONPATH="$QWENPUTER_DIR:$PWD"
gh auth token | python3 scripts/restore_private_release.py \
  arpjw/qwengpu v4-corrected-package-2026-11-14 445423220 \
  67e063bedb52e3f55fd5a298962e8a111c1de92ade2632225f9cc5a6e9d73913 \
  "$checkpoint" --backup-manifest corrected-github-backup-manifest.json
python3 scripts/check_checkpoint_package.py "$checkpoint" \
  --pinned-manifest results/resident_generation_v4_cuda/manifest.json \
  --expected-checksum-digest 67e063bedb52e3f55fd5a298962e8a111c1de92ade2632225f9cc5a6e9d73913
"$QWENPUTER_DIR/.venv/bin/python" -m qwengpu.run_resident \
  eval/frozen/tile8_gate_v1/dense_seeded.json \
  --model "$checkpoint" --device cuda --check-reference --out tile.json
```

The [release guide](RELEASE.md#corrected-v4-private-draft-package)
records the package checks and the frozen CUDA matrix and native-text replay.

## Apple MPS candidate

A 12.05 GB standalone export of the same v4 circuit passed exact CPU
smoke traces for deadlock and cooperative 32-bit shared memory. It
contains no pretrained text weights. The [standalone artifact card](release/standalone-v4-card.md)
records how to run the full Apple MPS qualification; that run is pending.

## Older v3 Hugging Face checkpoint

The verified v3 release remains at its immutable Hugging Face revision.
Sign in with `hf auth login` if the private model repository requires
access.

```bash
checkpoint=./Qwen-GPU-v3
hf download arpjw/Qwen-GPU \
  --revision fe671778a2409a218dde757871f33f5ca46d9219 \
  --exclude .gitattributes --local-dir "$checkpoint"

python -m qwengpu.run_resident eval/frozen/tile8_gate_v1/dense_seeded.json \
  --model "$checkpoint" --device cuda --check-reference --out tile.json

python -m qwengpu.run_compute examples/memory512_reduce_128.json \
  --model "$checkpoint" --device cuda --out reduction.json
```

The [reproduction guide](REPRODUCE.md#run-the-stock-generation-checkpoint)
includes the pinned environment, build path, and v4 qualification command.

## Project details

- [Scope and remaining gates](SCOPE.md)
- [Checkpoint release and restore](RELEASE.md)
- [Stock-generation protocol](MODEL_WRITEBACK_DESIGN.md)
- [Development history and earlier launchers](docs/DEVELOPMENT_HISTORY.md)

Qwen-GPU builds on [QwenCPU](https://github.com/arpjw/qwencpu) and uses
[Qwenputer](https://github.com/divagr18/qwenputer) to compile the circuit.
The source is MIT-licensed. The derived checkpoint carries the
Qwen3.5-9B Apache 2.0 license. [License and attribution](RELEASE.md#license-and-attribution)
