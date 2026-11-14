# Qwen-GPU v4 standalone circuit

This checkpoint contains the compiled v4 SIMT circuit without the
pretrained Qwen3.5-9B text tower. It is an intermediate artifact for
Apple MPS qualification. It is not the merged Qwen-GPU text model.

The circuit was exported from QwenGPU revision
`efd0111a1c971af2edaeb740446125b480a5bd17` with Qwenputer revision
`ca13310e2f9f788d1dd3feeea8213c1b0c23913c` using:

```sh
python scripts/export_resident_generation_standalone.py \
  --generated-tilegrid /path/to/standalone-v4
```

Its configuration has 52 compiled circuit layers plus one inert
full-attention layer, a 3,840-dimensional hidden state, and 212,992
SIMT token IDs. The single SafeTensors file is about 12 GB. The
`SHA256SUMS` file checks every packaged file, and `export-manifest.json`
records the source and compiler revisions.

The exported checkpoint loaded on CPU and matched the frozen
two-instruction deadlock trace and a 15-instruction cooperative indexed
32-bit shared-memory trace, including all 205 generated tokens in the
latter case. Both reports are packaged with the weights. These smoke
checks do not qualify Apple MPS. The full MPS replay remains open.

From a QwenGPU checkout with the pinned Qwenputer environment, set
`QWENGPU_STANDALONE_WEIGHTS` to the restored directory. The complete
MPS qualification command is:

```sh
export QWENGPU_STANDALONE_WEIGHTS=/path/to/standalone-v4
export QWENPUTER_DIR=/path/to/qwenputer/experimental/riscv
export QWENGPU_PYTHON="$QWENPUTER_DIR/.venv/bin/python"
export PYTHONPATH="$QWENPUTER_DIR:$PWD"
bash scripts/qualify_resident_generation_mps.sh --v4 \
  /path/to/mps-reports results/resident_generation_v4_cuda
```

The standalone circuit has no demonstrated floating-point GPU
arithmetic, atomics, graphics pipeline, CUDA compatibility, or
acceleration. QwenGPU source is MIT-licensed; the compiler is
MIT-licensed Qwenputer. This artifact contains no pretrained Qwen3.5
text weights.
