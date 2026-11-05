# Two-hour build checkpoint

This checkpoint records what the QwenGPU sprint established and the next
decisive experiment. Exact evidence and reproduction commands are in
[VALIDATION.md](VALIDATION.md) and [REPRODUCE.md](REPRODUCE.md).

## Built

- An independent eight-lane byte SIMT interpreter with structured divergence,
  scratch memory, addition, XOR, and reduction semantics.
- A circuit compiled into stock Qwen3.5 layers with eight masked ADD/XOR
  lanes, model-generated comparison masks, one-level divergence, result-token
  writeback, and four scratch bytes per lane. Its Gated DeltaNet state holds
  registers, masks, results, and memory.
- A merge ahead of pinned pretrained Qwen3.5-9B text weights. The final
  circuit uses 27 Gated DeltaNet heads and 12 circuit layers; the merge adds
  one exit layer and widens hidden size to 4,864.
- A JSON kernel runner for the supported instruction subset, plus a separate
  16-slot ROM and PC circuit that fetches uploaded instruction payloads from
  recurrent state.

## Observed checks

- Four 12-case 9B suites passed for divergence, scratch, ADD/XOR, and mixed
  programs. Generated tokens and inspected scratch rows were exact in the
  cases that check them.
- The exploratory mixed-program expansion passed 100/100 on both standalone
  MPS and merged 9B CUDA. All generated token sequences matched across the
  two runs, and all final scratch snapshots matched the interpreter. The
  100 cases share one program shape and are not a general reliability claim.
- A separate exploratory 32-case suite varied four program shapes on MPS.
  Its generated tokens, scratch state, and final lane probes were exact. It
  has not yet been run on CUDA or the 9B merge.
- The separate ROM spike passed 100/100 payload sequences on MPS with one
  model object and fresh caches. It generated 3,100 exact instruction/PC
  tokens and retained 1,600 exact ROM rows.
- Six fixed text prompts kept the same next top token after the 9B merge;
  five kept the same first four greedy tokens. One continuation changed, and
  the maximum absolute next-logit difference across prompts was 0.25.

## Current boundary

The host still supplies each SIMT instruction and eight lane read/feedback
pairs after lane-changing operations. The ROM spike is not connected to the
SIMT ALU. The compiled circuit has one warp, one-level divergence, byte
arithmetic, and 32 scratch bytes. This is a functional proof; acceleration
has not been measured. Text after a SIMT trace and broad language fidelity
have not been established. Model weights are excluded from GitHub; the pinned
source revision and merge builder reproduce them.

## Next decisive gate

Integrate the 16-slot ROM and model-held PC with the eight-lane ALU in a
separate circuit builder. Require one checkpoint to execute at least four
different valid straight-line kernels from fresh caches without the host
sending individual instructions. Compare generated PC/halt and lane tokens,
plus actual ROM, register, and scratch state, against the interpreter on MPS
and optimized CUDA. Rebuild the 9B merge only after the standalone model
passes. Then pursue internal lane writeback, followed by nested control flow
and multi-warp scaling. [SCOPE.md](SCOPE.md) and [ROM_DESIGN.md](ROM_DESIGN.md)
give the detailed sequence.

## Compute

One Runpod pod created for this sprint was terminated. It was quoted at
$2.49/hour and existed for about 53 minutes, giving an approximate $2.20 GPU
time projection. The per-pod billing read at 08:05 UTC posted $0.8027 across
two hourly records. The data available did not explain that difference, and
no final invoice was checked. No pod remains running; both figures are below
the $74 approved ceiling.
