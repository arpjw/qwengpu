# Qwen-GPU v4 merged checkpoint

Private draft backup of the CUDA-qualified Qwen3.5-9B merge. It contains
the pretrained text tower and the compiled SIMT circuit. This draft has
not been published.

## Verified state

- 91 GitHub assets, 86,321,158,445 bytes total, with verified size and
  SHA-256 for every asset part.
- Package `SHA256SUMS` SHA-256:
  `f20d0565c37fb766ce0eb80544c6693fcb20c6075b4987dcd41d35708cd7d213`.
- A clean download reconstructed and verified all 49 package files.
- A separate 96 GB CUDA GPU replayed the seeded tile, native text, and
  16×16 matrix. The matrix matched all 20,167 generated tokens and all
  256 output cells in one model generation call.

The [source release guide](https://github.com/arpjw/qwengpu/blob/9424997bcf4fa440a17fe6cb65429388ab7deb1d/RELEASE.md#qualified-v4-private-draft-backup)
links the asset receipt, restore transcript, comparison reports, and
reproduction commands.

## Restore

From an authenticated QwenGPU checkout, choose a destination that does
not already exist:

```sh
gh auth token | python3 scripts/restore_private_release.py \
  arpjw/qwengpu v4-private-backup-2026-11-14 445244518 \
  f20d0565c37fb766ce0eb80544c6693fcb20c6075b4987dcd41d35708cd7d213 \
  "$checkpoint"
```

The frozen package's `README.md` predates the private Hugging Face
storage-limit failure and incorrectly advertises a direct v4 Hub
download. Use the [corrected model card](https://github.com/arpjw/qwengpu/blob/9424997bcf4fa440a17fe6cb65429388ab7deb1d/release/model-card-v4.md)
and the restore command above. The package must be rebuilt with that
card and verified under a new checksum before any publication.

Full Apple MPS qualification and a v4 Hugging Face upload remain open.
The current Hugging Face revision is the earlier v3 checkpoint.
