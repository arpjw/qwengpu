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

## ROM continuation

The integrated builder `qwengpu/compiled_rom.py` now executes four frozen
straight-line programs from uploaded recurrent ROM state. One standalone
model object with fresh caches passed all four on MPS and optimized CUDA:
202 generated PC/halt and lane tokens per run, plus exact final ROM, PC,
register, and scratch state. The rebuilt 9B merge passed the same four
fixtures, also exactly. All 15 local tests pass.

The ROM variant uses 19 heads and 13 circuit layers. Its separate 9B merge
uses 14 chip layers including the exit marker, hidden size 4,608, and a
65,536-row chip vocabulary. The temporary checkpoint occupies about 27 GiB.
It preserves the source untied text output head.

## Bounded writeback continuation

The `compiled_writeback` builder removes all lane READ/result-feedback tokens.
It unrolls 16 instruction stages in model depth. The standalone circuit has
51 layers and 58 heads. A parallel carry adder reduced the initial 147-layer
design to 51 and passed all 65,536 byte pairs in the exact compiler evaluator.
All 17 local tests pass. The seven frozen writeback programs passed on MPS:
51 generated state tokens, 51 intermediate 40-byte register/scratch snapshots,
retained PC state, and final ROM rows all match the independent interpreter.
Execution uses two token calls per instruction after upload. All seven also
passed on optimized CUDA and inside the pretrained 9B merge, with identical
complete traces across the three runs. The merge adds 52 chip layers, uses
hidden size 5,120, and has 52,118,362,112 bytes of weights. This removes token
feedback by increasing model depth; it does not establish acceleration.

## Current boundary

The original divergent circuit still receives raw instructions from the host.
The ROM circuits fetch straight-line instructions from recurrent model state.
The newest variant writes results internally, using a separate snapshot for
each instruction position. It supports up to 16 instructions; it cannot
revisit an earlier stage with updated registers. There is one warp, byte
arithmetic, and 32 scratch bytes. No acceleration has been established.
Text after an execution trace and broad language fidelity remain open.

## Next decisive gate

Separate execution-stage identity from program address, then add bounded
control flow. Each stage must carry registers, scratch, active mask, and
branch-stack state forward even when a branch changes the PC. Freeze nested
empty/full/mixed-mask cases and inspect every intermediate state. Compare
standalone MPS, optimized CUDA, and the pinned 9B merge. Measure the depth,
weight, cache, token, and latency costs before increasing the execution budget.
Simply making the current PC jump backwards would read stale state.
See WRITEBACK_DESIGN.md for the construction and its limit.

## Text and compute after ROM integration

The new ROM merge kept the next top token on 6/6 fixed text prompts and the
first four greedy tokens on 5/6. The Moon continuation changed. Maximum
absolute next-logit difference was 0.2734375. This is a small integration
check; text fidelity and text after an execution trace remain open.

## Writeback text and final compute state

The writeback merge kept the next top token on 6/6 fixed prompts and the
first four greedy tokens on 5/6. The Moon continuation changed. Maximum
absolute next-logit difference was 0.25. Text used fresh caches; language
after machine execution and broad language fidelity remain open.

All four task-created Runpod pods are terminated. The final live pod list was
empty. First-pod posted billing is $2.2022584788501263. Later-pod GPU-time
projections at the quoted $2.49/hour are:

| Stage | Pacific time on November 5 | GPU-time projection |
| --- | --- | ---: |
| ROM integration | 10:50:42 to about 11:02:15 | $0.48 |
| Failed writeback pod boot | 11:09:55 to about 11:15:00 | $0.21 |
| Successful writeback validation | 11:15:10 to about 11:25:27 | $0.43 |

The later pods had no posted billing records at their latest reads. Adding
these rate projections to first-pod posted billing gives about $3.32 before
any later disk billing, below the approved $74 ceiling. This is not a final
invoice. Weights were temporary; source, pinned revisions, fixture hashes,
and result reports are saved in the private repository.
