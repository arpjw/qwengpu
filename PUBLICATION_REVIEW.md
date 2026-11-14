# Qwen-GPU private release review

The fixed v4 CUDA checkpoint is qualified and recoverable. Both the
GitHub source repository and Hugging Face model repository remain
private, as requested for review. This page records what a future
public release would expose and which checks remain at that boundary.

| Item | Current state | Evidence |
| --- | --- | --- |
| Fixed v4 execution | Passed the merged 9B CUDA qualifier and an independent clean GPU replay of the frozen package | [Validation](VALIDATION.md), [restore reports](results/resident_generation_v4_cuda/restored-reports) |
| Corrected v4 package | Clean file restore and pinned package check passed; only `README.md` changed from the GPU-replayed package | [Equivalence check](results/public-readiness/v4-package-equivalence-2026-11-14.json), [restore receipt](results/resident_generation_v4_cuda/corrected-package/clean-restore-receipt.json) |
| Private backup | Corrected 86 GB package is a 90-asset GitHub draft release; live asset sizes and digests match the manifest | [Latest asset audit](results/public-readiness/corrected-v4-live-asset-audit-latest-2026-11-14.json) |
| Hugging Face | Private repository still serves v3; v4 private upload hit the account storage limit | [Release guide](RELEASE.md#qualified-v4-private-draft-backup) |
| Public visibility | Neither repository has been made public | [Release guide](RELEASE.md#public-release-review) |

## GitHub history to review

The source repository has nine remote branches. All eight branches other
than `main` are merged into `main`; there are no tags. Its commit history
contains one author email that is not a GitHub noreply address. Tracked
reports and `RUNPOD_LEDGER.md` contain provider pod IDs and usage costs.
Making this repository public would expose that history and those files.
The [history inventory](results/public-readiness/history-exposure-audit-2026-11-14.json)
lists the earlier affected paths without printing the email.

A fresh [all-ref secret scan](results/public-readiness/source-scan-review-2026-11-14.json)
at commit `56d7f69` found the same 27 locations previously reviewed as
non-credentials and no new locations. A targeted search at that commit
found no matches for the narrow local-home and email patterns recorded
in the scan receipt. It did not inspect deleted historical blobs. These
checks do not establish that every item in the history is suitable to
publish.

If the existing history and cost records are acceptable, the current
repository can be considered for a visibility change after a final
source scan. If they are not, preserve this private original and prepare
a separate reviewed public snapshot. That path must preserve links to
the exact Qwen-GPU build, Qwenputer compiler, and base model revisions
used for the checkpoint.

## Model publication boundary

The corrected package has a verified private GitHub backup. The model
repository still has v3 on `main`. The [staged Hub procedure](RELEASE.md#corrected-v4-private-draft-package)
would upload v4 to a non-main branch, clean-restore and qualify that
immutable revision, then promote its files to `main` in one commit. A
small live private probe verified branch-to-branch copy and deletion;
the full 86 GB Hub upload and promotion have not run. Public Hub storage
capacity for that upload is not guaranteed.

After any future visibility change, check the public READMEs and images,
the model file list, a signed-out clean download, and the pinned package
digest before announcing availability. [Release commands and limits](RELEASE.md)

Apple MPS qualification and 32-bit atomics remain separate follow-on
work. They are outside the fixed v4 CUDA release gate.
