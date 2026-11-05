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

## Current boundary

The original divergent circuit still receives raw instructions from the host.
The separate ROM circuit fetches its own straight-line instructions but still
uses eight lane READ/result-feedback pairs after each non-halt TICK. Neither
variant combines autonomous ROM with divergent branches yet. Both have one
warp, byte arithmetic, and 32 scratch bytes. No acceleration has been measured.
Text after an execution trace and broad language fidelity remain open.

## Next decisive gate

Implement internal writeback in the ROM variant. Freeze multi-instruction
programs and require the same lane/scratch traces without eight host READ
and output-feedback pairs after every instruction. Keep program upload and
launch in the host, then use a fixed TICK/state-feedback protocol. Inspect
intermediate registers to catch errors hidden by later overwrites. Repeat on
standalone MPS, optimized CUDA, and the pinned 9B merge before adding control
flow. Then add bounded divergence and finally warp/memory scaling.

## Text and compute after ROM integration

The new ROM merge kept the next top token on 6/6 fixed text prompts and the
first four greedy tokens on 5/6. The Moon continuation changed. Maximum
absolute next-logit difference was 0.2734375. This is a small integration
check; text fidelity and text after an execution trace remain open.

Both task-created Runpod pods are terminated, and the live pod list is empty.
The first pod has posted billing of $2.2022584788501263, including GPU and disk.
The ROM continuation pod existed from November 5, 10:50:42 to about 11:02:15
Pacific at the quoted $2.49/hour, approximately $0.48 of GPU time. Its billing
query had no records immediately after termination, so $0.48 is an estimate,
not a posted charge. Combined posted first-pod billing plus that estimate is
about $2.68, below the approved $74 ceiling. Model weights were temporary;
source, pinned revisions, and result reports are saved in the private repo.
