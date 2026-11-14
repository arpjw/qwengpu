# Private checkpoint handoff

The final RunPod teardown requires a verified, recoverable private copy
of the qualified merged checkpoint. Keep the build pod until the
restore checks and receipt below are complete. A public release is a
separate decision.

## License and attribution

QwenGPU source is distributed under the [MIT License](LICENSE).
The derived model package carries the Apache 2.0 license from its
pinned Qwen3.5-9B base. The model card describes the modifications.
The pinned checkpoint's `LICENSE` was compared byte for byte with the
upstream license. The pinned base file listing contains `LICENSE`.
[Qwen3.5-9B upstream license](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/LICENSE)

The merge code credits the MIT-licensed QwenCPU implementation. The
checkpoint build uses the MIT-licensed Qwenputer compiler.
[QwenCPU license](https://github.com/arpjw/qwencpu/blob/main/LICENSE) ·
[Pinned Qwenputer license](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/LICENSE)

## Public release review

Keep both repositories private through qualification and teardown. The
GitHub history currently includes the author's commit email and RunPod
cost records, including provider pod IDs, usage, and spend. A visibility
flip exposes that history alongside the source and reports. Decide
whether to retain it before flipping. If a clean history is needed,
preserve the private original and prepare a reviewed public snapshot.
The build and compiler revisions referenced by the checkpoint must stay
available to readers of either release path.
[GitHub visibility guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility)

The final review should cover every GitHub branch and tag, model
repository references and package files, tracked logs and report
metadata, and license and source links. Before the flip, refresh the
secret scan against the final refs, verify the model revision and
checksums, and confirm the documentation matches the completed
qualification. After a flip, check both READMEs, images, source pins,
and the pinned model download in a signed-out session.

## Pinned checkpoint

The verified v3 package is stored in the Hugging Face model repository
[`arpjw/Qwen-GPU`](https://huggingface.co/arpjw/Qwen-GPU). Hugging Face
listed all 50 package files at revision
[`fe671778a2409a218dde757871f33f5ca46d9219`](https://huggingface.co/arpjw/Qwen-GPU/tree/fe671778a2409a218dde757871f33f5ca46d9219).

## Restore receipt

| Item | Verified value |
| --- | --- |
| Hugging Face revision | `fe671778a2409a218dde757871f33f5ca46d9219` |
| QwenGPU checkpoint build | `b5f4898cb2e54ac559cd0fa81d93c2901ccb9222` |
| Qwenputer compiler | `ca13310e2f9f788d1dd3feeea8213c1b0c23913c` |
| Qwen3.5-9B base | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Restore runner source | `fab58f0` |
| SHA-256 of SHA256SUMS | `f85fcb86f80668c57bfbc241aa8d9ca08378d50623eea405add56d499dcbd415` |

On November 14, 2026 UTC, a separate RunPod fetched all 50 package
files into an empty directory. The 49 content hashes, including all
35 SafeTensors shards, matched the original package. With the Hub
disabled during inference, the restored dense tile kernel matched 116
instructions, 2,519 generated tokens, ROM, control, registers, and
mirrored RAM. Native text matched 24 of 24 first tokens and 21 of 24
continuations with zero chip IDs.

[Build manifest](results/resident_generation_v3_cuda/manifest.json) ·
[Package checksums](results/resident_generation_v3_cuda/package-SHA256SUMS) ·
[GPU restore report](results/resident_generation_v3_cuda/restored-tile.json) ·
[Native-text restore report](results/resident_generation_v3_cuda/restored-text.json) ·
[Restore verifier output](results/resident_generation_v3_cuda/restore-verification.txt)

The model card and charts are also available on the repository's
default branch. The revision above is the checksum-pinned package
used for restore verification.

## Prepare the artifact

Use the qualified merged directory on RunPod. Preserve all SafeTensors
shards and their index, the model and generation configs, all tokenizer
files, and `qwengpu9b.json`. Include the qualified build manifest,
which pins the QwenGPU build commit, Qwenputer compiler commit, and
Qwen3.5-9B base revision. Record the validation-tools commit and runtime
versions in the package as well.

Copy the pinned Qwen Apache 2.0 `LICENSE` and any upstream notices
that apply. Preserve QwenGPU and Qwenputer notices for any material
copied into the package. Write a model card that credits the base and
describes the added circuit layers, widened text tower, reserved SIMT
token range, text-only merge, and tested GPU limits. Include tested run
commands and link to the source and validation evidence. Preserve the
upstream attribution and mark the derived checkpoint as modified. Start
from [the v3 model card](release/model-card.md) and update its evidence
section after qualification.
[Pinned Qwen license](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/LICENSE)

Generate `SHA256SUMS` over every shipped file except the checksum file
itself. Confirm the shard and configuration digests against the
qualification manifest before uploading. Keep the checksum file's
digest in the GitHub receipt so the restored copy can be verified
independently.

```bash
original_manifest=results/resident_generation_v3_cuda/manifest.json
python scripts/check_checkpoint_package.py "$package" \
  --pinned-manifest "$original_manifest" --write
# Save the printed digest in the receipt before downloading.
python scripts/check_checkpoint_package.py "$restored" \
  --pinned-manifest "$original_manifest" --expected-checksum-digest "$digest" \
  > "$reports/restore-checksum.txt"
```

## Store and restore

1. Use a user-controlled private Hugging Face model repository and verify
   its visibility before uploading. Upload directly from RunPod with an
   account credential scoped to the destination. Use the Hub's resumable
   folder upload with an explicit file list or a clean staging
   directory. Keep credentials, caches, and unrelated artifacts outside
   the upload. Record the commit SHA once the whole package is uploaded.
   [Hugging Face upload guide](https://huggingface.co/docs/huggingface_hub/main/guides/upload)
2. If Hub access or quota prevents the copy, use a user-controlled
   private object-storage bucket. Use a versioned or immutable prefix
   and record the provider, bucket, prefix, and object versions. Keep
   storage access independent of the build pod's lifetime.
3. Fetch the pinned revision or exact object versions into an empty
   directory with a fresh download cache. Use a separate GPU workspace
   if the build pod lacks disk for two copies. Verify every packaged
   file against `SHA256SUMS`, verify its digest against the original,
   and confirm the weight index references all and only the packaged
   shards. Exclude Hub-generated Git metadata such as `.gitattributes`
   if it was absent from the original package.
4. Load the restored directory with the pinned inference environment.
   Run one frozen GPU kernel with exact token and final-state checks,
   and run the native text gate against the pinned base. Save both
   reports alongside the manifest and GitHub receipt.
5. Push the receipt to the private QwenGPU source repository before
   terminating the final task pod. Include the durable locator and
   exact revision or object versions, checkpoint and checksum-manifest
   digests, build and verification revisions, and report paths. If a
   restore check fails, keep the source artifact available and record
   the failure before choosing another destination.

From the QwenGPU checkout, run the restore checks using the same pinned
base and compiler as the original qualification. Set `restored` to the
clean download, `base` to the pinned Qwen3.5-9B directory, and
`reports` to an empty report directory.

```bash
mkdir -p "$reports"
python -m qwengpu.run_resident eval/frozen/tile8_gate_v1/dense_seeded.json \
  --model "$restored" --device cuda --check-reference --trace \
  --out "$reports/restored-tile.json"
python -m qwengpu.textcheck_9b "$base" "$restored" --device cuda \
  --corpus eval/frozen/text_add32_gate_v1/prompts.json --max-new-tokens 16 \
  --out "$reports/restored-text.json"
python scripts/check_restored_checkpoint.py "$reports"
```

Require a halted tile with exact tokens, ROM, registers, control and
mirrored memory. Compare its tokens and output with the saved v3
`dense_seeded.json` case. Require 24 of 24 first tokens, at least 21 of
24 full continuations, and zero chip IDs in the restored text report.
