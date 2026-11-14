# Qwen-GPU publication review

The fixed v4 CUDA checkpoint is qualified and recoverable. The GitHub
source repository and Hugging Face model repository were made public
after the user authorized publication. GitHub distributes the corrected
v4 package; Hugging Face currently serves the verified v3 checkpoint.
This page records the release checks and the remaining v4 Hub upload.
The [claim audit](results/public-readiness/v4-claim-audit-2026-11-14.json)
compares the README and model card's key numeric statements with the
saved CUDA reports and two checksum-verified package metadata assets.

| Item | Current state | Evidence |
| --- | --- | --- |
| Fixed v4 execution | Passed the merged 9B CUDA qualifier and an independent clean GPU replay of the frozen package | [Validation](VALIDATION.md), [restore reports](results/resident_generation_v4_cuda/restored-reports) |
| Corrected v4 package | Clean file restore and pinned package check passed; only `README.md` changed from the GPU-replayed package | [Equivalence check](results/public-readiness/v4-package-equivalence-2026-11-14.json), [restore receipt](results/resident_generation_v4_cuda/corrected-package/clean-restore-receipt.json) |
| V4 distribution | Corrected 86 GB package is a 90-asset public GitHub release; asset sizes and digests match the manifest | [Latest asset audit](results/public-readiness/corrected-v4-live-asset-audit-latest-2026-11-14.json) |
| Hugging Face | Public repository serves verified v3; the v4 private upload hit the account storage limit | [Release guide](RELEASE.md#qualified-v4-private-draft-backup) |
| Public visibility | Both repositories are public; v4 Hub upload remains open | [Release guide](RELEASE.md#public-release-review) |

## GitHub history to review

The source repository has nine remote branches. All eight branches other
than `main` are merged into `main`; the public release adds one tag. Its
commit history
contains one author email that is not a GitHub noreply address. Tracked
reports and `RUNPOD_LEDGER.md` contain provider pod IDs and usage costs.
The public repository exposes that history and those files.
The [history inventory](results/public-readiness/history-exposure-audit-2026-11-14.json)
lists the earlier affected paths without printing the email.

A fresh [all-ref secret scan](results/public-readiness/source-scan-review-2026-11-14.json)
at commit `56d7f69` found the same 27 locations previously reviewed as
non-credentials and no new locations. A targeted search at that commit
found no matches for the narrow local-home and email patterns recorded
in the scan receipt. It did not inspect deleted historical blobs. These
checks do not establish that every item in the history is suitable to
publish.

The user authorized publication after this exposure was identified.
The final all-ref scan matched the same reviewed false positives and
found no new scanner locations. The
[prepublication receipt](results/public-readiness/publication-preflight-2026-11-14.json)
records the source head, release assets, and model revision checked at
the visibility boundary.

## Model publication boundary

The corrected package has a verified public GitHub release. The model
repository still has v3 on `main`. The [staged Hub procedure](RELEASE.md#corrected-v4-public-package)
would upload v4 to a non-main branch, clean-restore and qualify that
immutable revision, then promote its files to `main` in one commit. A
small live private probe verified branch-to-branch copy and deletion;
the full 86 GB Hub upload and promotion have not run. Public Hub storage
capacity for that upload is not guaranteed.

The public READMEs, images, file lists, and signed-out access were
checked after publication. The corrected v4 package had a full clean
restore and pinned checksum check before publication; the v4 Hub upload
still needs those checks if attempted later. [Release commands and limits](RELEASE.md)

Apple MPS qualification and 32-bit atomics remain separate follow-on
work. They are outside the fixed v4 CUDA release gate.
