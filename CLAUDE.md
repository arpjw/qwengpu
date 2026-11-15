# Qwen-GPU project brief

This is the shared QwenGPU checkout. Read `README.md`, `SCOPE.md`, and
`VALIDATION.md` before changing execution code. Use `REPRODUCE.md` for
pinned commands and `RELEASE.md` for checkpoint provenance.

## Current state

- The fixed v4 Qwen3.5-9B merge is CUDA-qualified. It runs bounded SIMT
  kernels with two eight-lane warps per workgroup, 512 bytes of global
  memory, 32-bit integer operations, and a full 16 by 16 matrix multiply
  in one generation call. The exact limits and evidence are in the docs.
- The corrected 86 GB v4 package is the public GitHub release
  `v4-corrected-package-2026-11-14`. The public Hugging Face repository
  `arpjw/Qwen-GPU` still serves the separately verified v3 weights.
- Full v4 Apple MPS qualification and an integrated 32-bit atomic
  instruction remain open. `ATOMIC32_DESIGN.md` distinguishes reference
  and isolated circuit work from a qualified checkpoint.

## Working in this checkout

- Inspect `git status` before editing. Preserve existing work and keep
  changes small and reviewable. Use the existing source and test layout.
- Do not claim a test, model run, Hub upload, or checkpoint qualification
  passed unless the matching command and output were observed. Keep
  proposed work separate from verified results.
- The Qwenputer compiler checkout is a sibling at
  `../qwenputer/experimental/riscv`; use its pinned environment as
  described in `REPRODUCE.md`.
- The merged v4 weights are not in this checkout. Follow `README.md` and
  `RELEASE.md` to restore the public release and verify its checksums.
- Avoid sustained CPU or memory-heavy jobs on Arya's Mac. Check current
  authorization and cost before using paid compute; do not assume the
  earlier RunPod budget has remaining capacity.
- Keep the GitHub source and Hugging Face model repositories aligned with
  their actual published versions. Do not overwrite the pinned v3 Hub
  revision or describe the v4 weights as available there.
