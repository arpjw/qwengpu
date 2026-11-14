# Private checkpoint handoff

The final RunPod teardown required a verified, recoverable private copy
of the qualified merged checkpoint. The private GitHub backup and clean
file restore passed before the task pod was terminated. A public release
is a separate decision.

## V4 handoff status

| Gate | Status | Evidence |
| --- | --- | --- |
| Fixed merged 9B CUDA qualification | Passed | [Qualified reports](results/resident_generation_v4_cuda) and [verifier output](results/resident_generation_v4_cuda/verification.txt) |
| Independent clean GPU replay of the frozen package | Passed | [Restored reports](results/resident_generation_v4_cuda/restored-reports) and [verifier output](results/resident_generation_v4_cuda/restored-reports/restored-verifier.txt) |
| Corrected model-card package | Passed clean file restore and pinned checksum check; CUDA execution evidence is inherited from byte-identical weights | [Clean restore receipt](results/resident_generation_v4_cuda/corrected-package/clean-restore-receipt.json) and [live asset audit](results/public-readiness/corrected-v4-live-asset-audit-2026-11-14.json) |
| Durable private backup | Available as a private GitHub draft; 90 assets match the manifest | [Backup receipt](results/resident_generation_v4_cuda/corrected-package/github-backup-receipt.json) |
| Hugging Face v4 | Not uploaded because the account reached its private storage limit; the private Hub repo still holds v3 | [Quota and revision record](#qualified-v4-private-draft-backup) |
| Public visibility | Awaiting an explicit publication decision and a final source scan | [Public release review](#public-release-review) |

Apple MPS qualification and 32-bit atomics are separate follow-on work.
Neither is a gate for the fixed v4 CUDA handoff.

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

The [history exposure audit](results/public-readiness/history-exposure-audit-2026-11-14.json)
at source commit `988b757` found nine remote branches, with all eight
non-main branches fully merged into `main`, and no Git tags. The history
contains one distinct commit-author email that is not a GitHub noreply
address. Ten tracked files matched RunPod cost or pod-ID keywords. The
receipt lists those files without exposing the author address. Decide
whether this existing history and usage metadata should be public before
changing visibility.

The final review should cover every GitHub branch and tag, model
repository references and package files, tracked logs and report
metadata, and license and source links. Before the flip, refresh the
secret scan against the final refs, verify the model revision and
checksums, and confirm the documentation matches the completed
qualification. After a flip, check both READMEs, images, source pins,
and the pinned model download in a signed-out session.

The November 14 all-ref source scan at `6df7da9` covered 555 commits,
nine remote branches, and no tags. Gitleaks 8.30.1 returned 27 generic
API-key matches. Review of the matched source lines found 25
tokenized-prompt SHA-256 fields, one matrix-token SHA-256 constant, and
one Transformers import. None of these matches is a credential. The
[redacted scan summary](results/public-readiness/source-scan-2026-11-14.json)
records the command, refs, classifications, and finding locations without
secret values. Rerun the scan if refs change before publication.
The [latest saved all-ref rescan](results/public-readiness/source-scan-current-2026-11-14.json)
at source commit `d892ecb` covered 574 reachable commits. Gitleaks
reported 557 scanned commits and the same 27 reviewed locations, with
no new scanner matches. Publication still requires a fresh scan against
its final refs.

The [card and source audit](results/public-readiness/card-and-source-audit-2026-11-14.json)
confirms that the v4 model card and both graphics in this checkout match
the corrected package's checksums. Its simple local-link check found no
missing targets in the main README, scope, or release guide. The pinned
QwenCPU repository, Qwenputer commit, and public Qwen3.5 base revision
resolve; the base license digest matches the corrected package. This
does not verify signed-out access to the still-private repositories.

## Pinned checkpoint

The verified v3 package is stored in the Hugging Face model repository
[`arpjw/Qwen-GPU`](https://huggingface.co/arpjw/Qwen-GPU). Hugging Face
listed all 50 package files at revision
[`fe671778a2409a218dde757871f33f5ca46d9219`](https://huggingface.co/arpjw/Qwen-GPU/tree/fe671778a2409a218dde757871f33f5ca46d9219).

## Qualified v4 private draft backup

The fixed v4 merge passed the complete CUDA qualifier at source commit
`702a6f9`. The 35 weight shards and every other packaged file matched
the [qualified manifest](results/resident_generation_v4_cuda/manifest.json).
The 48 packaged content files are listed in
[SHA256SUMS](results/resident_generation_v4_cuda/package-SHA256SUMS),
whose SHA-256 is
`f20d0565c37fb766ce0eb80544c6693fcb20c6075b4987dcd41d35708cd7d213`.
The [CUDA verifier output](results/resident_generation_v4_cuda/verification.txt)
exited zero. The restored copy also passed its own CUDA execution checks
on a separate 96 GB GPU.

The authorized private upload to `arpjw/Qwen-GPU` failed on November
14 with Hugging Face's response: `Private storage limit reached for
user arpjw`. A read after the failure confirmed the repository remained
private and at revision `7182ddec3349cace59850f706af11dbe0e7232dc`,
with its 53 existing files unchanged. All 35 v4 shard checksums differ
from v3. Hugging Face documents 100 GB of private storage for a free
account and 1 TB for PRO; storage limits apply across repositories.
[Storage limits](https://huggingface.co/docs/hub/storage-limits)

The v4 package is backed up on the private GitHub [releases page](https://github.com/arpjw/qwengpu/releases)
as draft tag `v4-private-backup-2026-11-14`, release ID `445244518`.
GitHub reported 91 uploaded assets totaling
86,321,158,445 bytes, including 89 package assets, the backup manifest,
and an extra copy of the checksum list. Each package asset's reported
size and SHA-256 matched the [backup manifest](results/resident_generation_v4_cuda/github-backup-manifest.json).
The backup manifest's SHA-256 is
`5150eab9a807b44feb7e46e6bb4edd43b66cf3063f45a499b39662100ffe79e5`.
The [asset receipt](results/resident_generation_v4_cuda/github-backup-receipt.json)
records the private and draft status. The [upload transcript](results/resident_generation_v4_cuda/github-backup-upload.txt)
lists the verified assets.

The restore helper downloaded the draft release into an empty directory
on RunPod. Its [transcript](results/resident_generation_v4_cuda/github-backup-restore.txt)
records 49 reconstructed files, including all 35 weight shards, with
matching part and full-file digests. The separate package checker
confirmed the frozen build manifest, exact shard index, and the original
SHA256SUMS digest in [restore-checksum.txt](results/resident_generation_v4_cuda/restore-checksum.txt).
This is a clean byte-for-byte file restore. On November 14 a new RunPod
network volume was mounted on a separate RTX PRO 6000 Blackwell Server
Edition GPU. The [restored reports](results/resident_generation_v4_cuda/restored-reports)
record a 116-instruction seeded tile with exact generated tokens and
state, 24 of 24 matching first text tokens and 21 of 24 matching short
continuations, and a 928-instruction 16×16 matrix with all 20,167
generated tokens, panel transitions, and 256 output cells identical to
the qualified original. The matrix used one generation call and one
fresh cache with zero host execution tokens. The independent
[v4 restore verifier](results/resident_generation_v4_cuda/restored-reports/restored-verifier.txt)
exited zero. [Environment](results/resident_generation_v4_cuda/restored-reports/restored-environment.txt)
and [full-file restore transcript](results/resident_generation_v4_cuda/restored-reports/restore-transcript.txt)
are saved with the reports.

The earlier RunPod migration could not place the retained volume on a GPU:
its migration dialog reported `There are no instances currently
available`. Pod `gmgmirb2mz2hyy` was terminated after the restore
evidence was pushed. Deletion returned HTTP 204, a follow-up lookup
returned 404, and the pod list was empty. A post-teardown download of
the private release's `SHA256SUMS` matched the pinned digest. The
[teardown receipt](results/resident_generation_v4_cuda/teardown-receipt.json)
records the earlier checks and billing boundary. The later independent
network-volume restore passed GPU execution as recorded above. The
temporary GPU pod and network volume were deleted after the reports were
copied and verified locally. A full v4 Hugging Face upload still needs
private storage capacity. The existing v3 revision remains the verified
Hugging Face download below.

The frozen v4 backup's packaged `README.md` predates the Hugging Face
quota failure and still shows a direct v4 Hub download that is not
available. The corrected package below includes the revised
[v4 model card](release/model-card-v4.md). Preserve the frozen backup
and its original checksum as the CUDA execution baseline.

Reproduce this private file restore with GitHub access and the release
ID above. The helper reads a token from standard input and requires a
new destination directory:

```bash
gh auth token | python scripts/restore_private_release.py \
  arpjw/qwengpu v4-private-backup-2026-11-14 445244518 \
  f20d0565c37fb766ce0eb80544c6693fcb20c6075b4987dcd41d35708cd7d213 \
  "$restored"
python scripts/check_checkpoint_package.py "$restored" \
  --pinned-manifest results/resident_generation_v4_cuda/manifest.json \
  --expected-checksum-digest \
  f20d0565c37fb766ce0eb80544c6693fcb20c6075b4987dcd41d35708cd7d213
```

## Corrected v4 private draft package

A separate private GitHub draft, tag `v4-corrected-package-2026-11-14`
and release ID `445423220`, contains the corrected model card and a new
`SHA256SUMS`. The [repackage receipt](results/resident_generation_v4_cuda/corrected-package/repackage-receipt.json)
records 47 unchanged content files and the one changed `README.md`.
Every weight shard is byte-identical to the CUDA-qualified frozen backup.
The revised checksum list has SHA-256
`67e063bedb52e3f55fd5a298962e8a111c1de92ade2632225f9cc5a6e9d73913`.
The corrected card has SHA-256
`c214c59bdfd9bb7cbeb4ab854fc0c5eecea8ad3c61ec5c14862afc07c7d5616f`.

GitHub reported 90 assets totaling 86,321,154,200 bytes. The
[asset receipt](results/resident_generation_v4_cuda/corrected-package/github-backup-receipt.json)
records private and draft status and matching asset sizes and SHA-256
values against the [backup manifest](results/resident_generation_v4_cuda/corrected-package/github-backup-manifest.json).
That manifest has SHA-256
`3c6c39c345967b0bdc2352edd248384f92dcfdf8ab0e885a1df76cf961b9ea5c`.
The corrected package passed a full file and pinned-manifest check
before upload. This package revision changes documentation, so its CUDA
execution evidence is inherited by byte identity from the frozen
checkpoint; it is not a new GPU replay.

The [latest live asset audit](results/public-readiness/corrected-v4-live-asset-audit-2026-11-14.json)
checked all 90 GitHub asset names, sizes, and SHA-256 metadata against
the pinned backup manifest. Reproduce this lightweight metadata check
without downloading weights:

```bash
python3 scripts/check_release_assets.py \
  results/resident_generation_v4_cuda/corrected-package/github-backup-manifest.json
```

A separate clean download reconstructed all 49 files from this corrected
draft. The first transfer stopped after shard 16 when the temporary
volume reached a write quota. With redundant package trees removed,
`--resume` verified 25 existing files and restored the remaining 24.
The independent pinned package checker then exited zero and returned
the corrected checksum digest. The [clean restore receipt](results/resident_generation_v4_cuda/corrected-package/clean-restore-receipt.json),
[full transcript](results/resident_generation_v4_cuda/corrected-package/restore-resume.txt),
and [package check](results/resident_generation_v4_cuda/corrected-package/restore-package-check.txt)
record the exact result. No new CUDA execution run was performed.
The 35 weight shards are among the 47 unchanged content files, which
are byte-identical to the qualified and independently replayed frozen
package.
The temporary CPU pod and network volume were deleted after the
transcripts were copied and matched locally. The [teardown receipt](results/resident_generation_v4_cuda/corrected-package/teardown-receipt.json)
records their 204 deletion responses and empty follow-up lists.

Restore the corrected package into a nonexistent directory with GitHub
access:

```bash
gh auth token | python3 scripts/restore_private_release.py \
  arpjw/qwengpu v4-corrected-package-2026-11-14 445423220 \
  67e063bedb52e3f55fd5a298962e8a111c1de92ade2632225f9cc5a6e9d73913 \
  "$restored" --backup-manifest corrected-github-backup-manifest.json
python3 scripts/check_checkpoint_package.py "$restored" \
  --pinned-manifest results/resident_generation_v4_cuda/manifest.json \
  --expected-checksum-digest \
  67e063bedb52e3f55fd5a298962e8a111c1de92ade2632225f9cc5a6e9d73913
```

If the transfer stops, repeat the restore command with `--resume`. It
rehashes existing expected files, skips complete files, and replaces
only incomplete ones. It rejects unexpected entries and symlinks in
the destination.

If this repository and the corrected release are later made public,
run the same restore command without the `gh auth token |` prefix and
with `--public`. The helper then requires a published release in a
public repository while preserving the release ID and checksum checks.
Run `scripts/check_release_assets.py` with `--public` too. Publication
still requires an explicit visibility decision, a fresh source scan,
and a pinned v4 Hugging Face revision if that destination is used.

The Hugging Face upload helper now verifies the pinned build manifest,
all package file hashes, and the `SHA256SUMS` digest before sending any
files. It cannot change repository visibility. After an explicit public
release decision, make the model repository public through its settings,
then upload the verified corrected package from a machine with enough
disk space:

```bash
printf '%s\n' "$HF_TOKEN" | python scripts/upload_private_model.py \
  "$restored" arpjw/Qwen-GPU --public \
  --pinned-manifest results/resident_generation_v4_cuda/manifest.json \
  --expected-checksum-digest \
  67e063bedb52e3f55fd5a298962e8a111c1de92ade2632225f9cc5a6e9d73913 \
  --message "Upload qualified Qwen-GPU v4"
```

Record the returned Hub revision. Clean-download that revision and run
the package checker, a frozen GPU kernel, and the native-text gate
before updating the public model card or claiming a Hub restore.

## Standalone v4 Apple MPS candidate

The standalone v4 circuit was exported separately from the 86 GB merged
checkpoint. Its 12,048,670,904-byte SafeTensors file has SHA-256
`67bcb12dee641c6605a93e2ba79bcf4b3da7949baf9cfb6727ed4a2b02ee6e31`.
The package's `SHA256SUMS` has SHA-256
`98e29de12f2896aa755076a3ba54e773aa3bed3582724a029333f0f288dd8687`.
It contains the v4 SIMT circuit but no pretrained Qwen3.5 text tower.
Two exact CPU smoke traces passed; full Apple MPS qualification is pending.
The [artifact card](release/standalone-v4-card.md),
[export manifest](results/resident_generation_v4_cuda/standalone-export/export-manifest.json),
and [CPU reports](results/resident_generation_v4_cuda/standalone-export)
record the build and checks.
The [export transcript](results/resident_generation_v4_cuda/standalone-export/export-transcript.txt)
and [upload transcript](results/resident_generation_v4_cuda/standalone-export/upload-transcript.txt)
record the artifact creation and verified assets.

The private draft on the GitHub [releases page](https://github.com/arpjw/qwengpu/releases)
has tag `v4-standalone-mps-candidate-2026-11-14` and release ID
`445397359`. GitHub reported 17 assets totaling
12,048,693,460 bytes, all with SHA-256 digests. The [asset receipt](results/resident_generation_v4_cuda/standalone-export/github-backup-receipt.json)
records the private and draft status, and the [backup manifest](results/resident_generation_v4_cuda/standalone-export/standalone-backup-manifest.json)
maps every file to its verified assets. Restore into a nonexistent directory using
an authenticated GitHub CLI. The helper verifies the package checksum,
each asset part, and every reconstructed file:

```bash
gh auth token | python3 scripts/restore_private_release.py \
  arpjw/qwengpu v4-standalone-mps-candidate-2026-11-14 445397359 \
  98e29de12f2896aa755076a3ba54e773aa3bed3582724a029333f0f288dd8687 \
  "$standalone" --backup-manifest standalone-backup-manifest.json
```

On November 14, a clean download reconstructed and verified all nine
package files, including the full SafeTensors file. The restored copy
again matched the 15-instruction cooperative 32-bit shared-memory
fixture on CPU, with 205 exact generated tokens and exact machine state.
The [restore transcript](results/resident_generation_v4_cuda/standalone-export/restore-transcript.txt),
[restored run report](results/resident_generation_v4_cuda/standalone-export/restored-addmem32.json),
and [pod teardown receipt](results/resident_generation_v4_cuda/standalone-export/teardown-receipt.json)
are saved. The pod was deleted after these checks.

Set `QWENGPU_STANDALONE_WEIGHTS="$standalone"` before running the
[v4 MPS qualification script](scripts/qualify_resident_generation_mps.sh).
The standalone circuit is a candidate for that script, not an MPS
qualified checkpoint.

GitHub can change a draft release's `untagged-*` URL when its notes are
edited. Resolve the current URL from its stable release ID:

```sh
gh api repos/arpjw/qwengpu/releases/445244518 --jq .html_url
gh api repos/arpjw/qwengpu/releases/445397359 --jq .html_url
```

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
   account credential scoped to the destination. The upload helper reads
   the token from standard input, confirms the repository is already
   private, checks the pinned package, uploads the flat package, and
   verifies the resulting revision and visibility. It removes stale files
   from the prior private revision and checks the final file list while
   retaining Hub `.gitattributes`.
   Keep credentials, caches, and unrelated artifacts outside the package.
   Record the commit SHA once the whole package is uploaded.
   [Hugging Face upload guide](https://huggingface.co/docs/huggingface_hub/main/guides/upload)

   ```bash
   printf '%s\n' "$HF_TOKEN" | python scripts/upload_private_model.py \
     "$package" arpjw/Qwen-GPU \
     --pinned-manifest "$original_manifest" \
     --expected-checksum-digest "$digest" \
     --message "Upload qualified Qwen-GPU checkpoint"
   ```
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

For a qualified v4 package, also reproduce its single-launch 16×16
matrix on the clean download. Set `qualified_v4` to the complete merged
v4 CUDA report directory. Use `$qualified_v4/manifest.json` as the
`--pinned-manifest` and the recorded v4 checksum-manifest digest when
checking the package. Save the matrix report under the filename the
restore verifier expects:

```bash
python -m qwengpu.run_matmul16 eval/frozen/matmul16_gate_v1/dense_seeded.json \
  --model "$restored" --device cuda --out "$reports/restored-matmul16.json"
python scripts/check_restored_checkpoint.py "$reports" \
  --qualified "$qualified_v4" --v4
```

The v4 verifier requires the restored matrix tokens, panel transitions,
output, and one-call count to match the qualified merged report. It also
compares the restored text token sequences and the seeded tile trace.
