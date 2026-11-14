# Private checkpoint handoff

The final RunPod teardown requires a verified, recoverable private copy
of the qualified merged checkpoint. Keep the build pod until the
restore checks and receipt below are complete. A public release is a
separate decision.

The v3 package is staged on the build pod. Its 49 file digests are
recorded in `results/resident_generation_v3_cuda/package-SHA256SUMS`.
The upload destination is the private Hugging Face model repository
[`arpjw/Qwen-GPU`](https://huggingface.co/arpjw/Qwen-GPU). Hugging Face
reports it as private and lists all 50 package files at revision
[`fe671778a2409a218dde757871f33f5ca46d9219`](https://huggingface.co/arpjw/Qwen-GPU/tree/fe671778a2409a218dde757871f33f5ca46d9219).
The clean restore reports are pending.

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
