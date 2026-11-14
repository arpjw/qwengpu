# Qwen-GPU v4 corrected package

Private draft backup of the fixed Qwen3.5-9B CUDA-qualified checkpoint.
The package updates the model card after the private Hugging Face upload
hit the account's storage limit. All 35 weight shards and 47 of 48
content files are byte-identical to the frozen qualified backup.

The revised `SHA256SUMS` has SHA-256
`67e063bedb52e3f55fd5a298962e8a111c1de92ade2632225f9cc5a6e9d73913`.
The backup manifest is `corrected-github-backup-manifest.json` and has
SHA-256 `3c6c39c345967b0bdc2352edd248384f92dcfdf8ab0e885a1df76cf961b9ea5c`.
The private draft has 90 assets totaling 86,321,154,200 bytes, all
checked against that manifest.
A separate clean download reconstructed all 49 files. After an
interrupted transfer, resume verified 25 existing files and downloaded
the other 24; the independent pinned package checker exited zero.
The receipt and full transcript are linked from `RELEASE.md`.

Restore with the pinned release ID and verification commands in
[`RELEASE.md`](https://github.com/arpjw/qwengpu/blob/main/RELEASE.md#corrected-v4-private-draft-package).
CUDA execution evidence comes from the byte-identical frozen checkpoint
and its independent 96 GB GPU restore. Full Apple MPS qualification
remains pending and is separate from this CUDA release. Atomics are
follow-on work. Keep this draft and the source repository private until
the public release review is complete.
