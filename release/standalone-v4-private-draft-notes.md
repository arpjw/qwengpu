# Qwen-GPU v4 standalone circuit

Private draft backup of the 12.05 GB v4 SIMT circuit without the
pretrained Qwen3.5-9B text tower. This is an Apple MPS candidate, not
an MPS-qualified or merged text checkpoint.

## Verified state

- 17 private draft assets, 12,048,693,460 bytes total, each with a
  verified SHA-256.
- Package `SHA256SUMS` SHA-256:
  `98e29de12f2896aa755076a3ba54e773aa3bed3582724a029333f0f288dd8687`.
- A clean download reconstructed and verified all nine package files.
- The restored file matched a 15-instruction cooperative 32-bit
  shared-memory trace on CPU, including all 205 generated tokens and
  the checked machine state. A second CPU smoke case checked deadlock.

The [artifact card](https://github.com/arpjw/qwengpu/blob/9424997bcf4fa440a17fe6cb65429388ab7deb1d/release/standalone-v4-card.md)
and [evidence](https://github.com/arpjw/qwengpu/tree/9424997bcf4fa440a17fe6cb65429388ab7deb1d/results/resident_generation_v4_cuda/standalone-export)
give the source pins, model hash, CPU reports, and Apple MPS command.

## Restore

From an authenticated QwenGPU checkout, choose a destination that does
not already exist:

```sh
gh auth token | python3 scripts/restore_private_release.py \
  arpjw/qwengpu v4-standalone-mps-candidate-2026-11-14 445397359 \
  98e29de12f2896aa755076a3ba54e773aa3bed3582724a029333f0f288dd8687 \
  "$standalone" --backup-manifest standalone-backup-manifest.json
```

Full Apple MPS replay remains pending. The original merged v4 model
passed its complete CUDA qualification and an independent CUDA replay;
this standalone export has only the stated CPU smoke evidence.
