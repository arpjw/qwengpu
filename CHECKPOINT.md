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

## Loops and divergent control continuation

The separate `compiled_control` variant adds data-dependent WHILE loops and
two levels each of IF and loop stacks. A model-generated token carries PC
and execution step separately. A backward PC jump therefore reads updated
state from the previous execution stage. Registers, scratch, masks, and both
stacks use internal writeback; no lane tokens are fed back. The control
circuit has 69 layers and 75 heads. Fourteen frozen cases pass on MPS and
optimized H100 CUDA, covering 130 executed instructions. Thirteen HALT and
one intentionally exhausts the 16-step budget. All intermediate machine
snapshots, generated control tokens, and retained ROM/link rows match.
All 21 local tests pass, including exhaustive byte-add and comparison checks.
The same 14 cases also passed inside the pinned pretrained 9B merge, with
identical complete traces across all three platforms. The merge adds 70
chip layers, uses hidden size 5,120, and has 61,946,385,152 bytes of weights.
Current text and compute evidence is recorded below.

## Current boundary

The latest variant supports a 16-slot ROM, at most 16 executed instructions,
eight byte lanes, 32 scratch bytes, and two frames in each control stack.
Branches and data-dependent loops can nest and combine. Exhaustion is a
separate outcome, not successful HALT. The state pipeline remains bounded in
model depth; increasing the step limit adds depth and requires more step
bits. There are no arbitrary jumps, reductions, or multiple warps. No
acceleration or broad language fidelity has been established. Text after
an execution trace remains untested.

## Next decisive gate

Increase the execution budget only after measuring weight/cache/latency
costs. Add larger frozen and exploratory program suites with independently
specified outcomes, including longer divergent loops. Then pursue warp and
memory scaling and a measured text-cache handoff. The current bounded loop
and nested-control construction is specified in CONTROL_DESIGN.md.

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

All five task-created Runpod pods are terminated. The final live pod list was
empty. First-pod posted billing is $2.2022584788501263. Later-pod GPU-time
projections at the quoted $2.49/hour are:

| Stage | Pacific time on November 5 | GPU-time projection |
| --- | --- | ---: |
| ROM integration | 10:50:42 to about 11:02:15 | $0.48 |
| Failed writeback pod boot | 11:09:55 to about 11:15:00 | $0.21 |
| Successful writeback validation | 11:15:10 to about 11:25:27 | $0.43 |
| Control H100 validation ($3.99/hour) | 11:37:27 to about 11:48:36 | $0.74 |

The later pods had no posted billing records at their latest reads. Adding
these rate projections to first-pod posted billing gives about $4.06 before
any later disk billing, below the approved $74 ceiling. This is not a final
invoice. Weights were temporary; source, pinned revisions, fixture hashes,
and result reports are saved in the private repository.


## Control text check

The control merge kept the next top token on 6/6 fixed prompts and the first
four greedy tokens on 5/6. The Moon continuation changed. Maximum absolute
next-logit difference was 0.3203125. These use fresh text caches. Complete
control traces and fixture hashes match across MPS, standalone H100 CUDA,
and merged Qwen3.5-9B H100 CUDA. GPU resources are terminated and all source
and evidence are saved in the private repository.
