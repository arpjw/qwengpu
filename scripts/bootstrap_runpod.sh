#!/usr/bin/env bash
# Run inside an ephemeral CUDA pod. No credentials or private repo access needed.
set -euo pipefail
cd /workspace
if [[ ! -d qwenputer/.git ]]; then
  git clone --filter=blob:none https://github.com/divagr18/qwenputer.git
fi
git -C qwenputer checkout ca13310e2f9f788d1dd3feeea8213c1b0c23913c
cd qwenputer/experimental/riscv
uv sync --extra hf --no-dev
uv pip install --python .venv/bin/python flash-linear-attention==0.6.0 fla-core==0.6.0 einops
.venv/bin/hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir /workspace/Qwen3.5-9B
