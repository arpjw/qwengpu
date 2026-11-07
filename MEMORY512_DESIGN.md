# 512-byte programmable memory

The new layout is `qwengpu-resident-memory512-9b-v1`. This document defines
the candidate; qualification results belong in VALIDATION.md.

## Memory and address semantics

Programmable launches have 256 uint16 words, or 512 bytes. Two limbs form
one uint32 word, little endian. Two eight-lane warps share the bank, and
one to four sequential workgroups keep it in one inference cache.
The first launch uploads 1024 tagged nibbles. A dependent launch resets
warp state and loads its ROM while retaining memory with zero re-upload.

`SAVE_INDEX` takes register bits 0 through 7. Indexed high-limb operations
toggle bit zero. `ADD_INDEX_LO/HI` XOR an even immediate into the address;
all even masks 0 through 254 are supported. Distinct addresses with equal
low seven bits remain distinct for both reads and write arbitration.
Within a warp, the lowest active lane wins a same-address store.
A uint32 pair is not an atomic store: explicit barriers or launch boundaries
are needed to publish both limbs before a cooperating consumer reads them.

New memory tokens occupy `[0x20000, 0x24000)`. This range is disjoint from
the older ROM, state, request, and specialized tile tags. Each token packs
an eight-bit word address, a two-bit nibble address, and a nibble value.
The chip vocabulary is 147456 tokens. Text uses the original model's
namespace, with the existing chip offset applied at the transport boundary.

## Banked reads and compatible state

Each indexed port uses two 128-row attention heads. Both banks reuse the
low seven query bits; address bit seven selects the returned value.
Loading keys share word-row, nibble, and bank features instead of allocating
a separate feature for every token address. No attention head needs a
larger row count than the qualified 128-row limit.

The original 388 snapshot bits retain their positions. Eight additional
index bits, the programmable-mode bit, and three padding bits precede the
dirty flags. The existing 96-nibble warp packet already had a spare index
bit, so its size stays unchanged. Legacy tiled launches require those
extra twelve snapshot bits to be zero and omit them from public traces.
Their original register, mask, branch, loop, and grant fields stay intact.

Legacy ADD_INDEX encodings stay unchanged. Extended masks use explicitly
mapped, previously unused BARRIER/HALT immediates. This mapping applies only
to the new layout; the old layout's entire ROM decoder remains identical.

The old compute graph remains selectable. Preflight compares its complete
IR and ROM domain to the pinned previous source. The new graph adds bank
heads and eight-bit index logic only when the new layout is selected.
Specialized tiled kernels retain their original separate memory layout.

## Common kernels

The common request runner supports uint32 vector addition with 0 through
64 elements, and modular uint32 reduction with 0 through 128 inputs.
Vector A occupies words 0 through 127 and B occupies words 128 through 255;
tail guards and B remain unchanged.

Reduction uses at most 64 threads. For more than 64 inputs, each active
thread conditionally adds its input from the second half before the warp
sum. The producer has 15 ROM slots including HALT. A second launch reads
up to eight warp partials and reduces them without uploading memory again.
The sum wraps modulo 2^32. Empty reductions return zero.

Byte-input matrix multiply retains M <= 8, K <= 4, N <= 4 with uint32 C.
The specialized 16x16 modular uint16 tiled runner is a separate supported
kernel on the same checkpoint. Increasing memory does not enlarge its ROM
or remove its fixed scheduling structure.

## Qualification boundary

The frozen set includes seven raw bank/barrier/collision probes and 31
common requests, including 64-element vector addition and a 128-element
all-maximum reduction. A separate raw matrix-extent probe covers all 64
lanes across four workgroups. Unit checks cover every memory address and every
supported vector and reduction length against independent scalar results.
Actual inference must verify both MPS and merged 9B, all inherited traces,
the long tiled matrix kernel, and merged-model text/cache handoffs.

This remains a bounded constructed SIMT processor. It uses host-relayed
state tokens, sequential workgroups, and 16 ROM slots per warp. The work
does not establish arbitrary-program correctness, GPU acceleration,
floating-point support, or a general CUDA-compatible GPU.
