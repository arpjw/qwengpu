# Next memory gate: 256 programmable words

This is an unimplemented follow-up to the compute checkpoint. It proposes
512 bytes of programmable memory. The current bank remains 128 uint16
words, or 256 bytes. Larger specialized tile banks are separate.

The next useful step is to double indexed capacity while keeping the
existing checkpoint reproducible. Introduce a new layout identifier and
retain all current graph paths when that layout is not selected.

## Constraints found in the current source

`compiled_tile8.py` currently reads seven index bits per lane: four from
the first state nibble and three from the second. `SAVE_INDEX` takes seven
register bits. `_state_snapshot_digits` reconstructs the eighth index bit
as zero. The packed snapshot reserves 56 index bits followed by eight
scatter grants. Increasing only `INDEX_WORDS` would corrupt this ABI.

Both memory halves use 64 word slots, each with four tagged nibbles.
Doubling each half needs a larger token range. Extending the operand range
at `INDEX_OPERAND` would overlap the existing warp-state tag range.
Allocate distinct ranges in a larger chip vocabulary; verify every tag
range against ROM, state, tile-bank, request, and sentinel tokens.

`ADD_INDEX_LO` and `ADD_INDEX_HI` currently encode an even XOR address mask
from 0 through 126. The reserved ROM encoding also assumes that bound.
An eighth address bit needs an explicit encoding decision and loader checks;
changing the range alone would collide with other instruction encodings.

The merge allocator uses circuit live slots to select widened hidden size.
More memory and address queries can therefore grow pretrained weights and
cache cost. Linear extrapolation from logical memory bytes is insufficient.

## Implementation order

1. Define one geometry description for index bits, memory halves, nibble
   tags, snapshot offsets, and address-argument encoding. Preserve the
   existing seven-bit geometry and saved reports exactly.
2. Build the eight-bit variant and print graph statistics, live slots,
   selected hidden size, and estimated checkpoint storage before compiling
   model weights. Detect token-range collisions mechanically.
3. Freeze boundary and aliasing programs for addresses 0, 63, 64, 127,
   128, 191, 192, and 255. Include masked collisions, both 32-bit limbs,
   cross-warp barriers, and cross-workgroup retained memory.
4. Validate actual memory and every instruction on standalone MPS. A
   widened snapshot needs a deliberate semantic projection onto previous
   traces, rather than claiming old packed snapshot strings are unchanged.
5. Merge into the pinned 9B base and run the same cases. Record model,
   cache, token, latency, disk, and billing costs. Qualify inherited kernels
   and text handoffs before treating the variant as the new baseline.
6. Extend vector addition to 64 values and investigate reduction of 128
   values with two inputs per thread. These are proposed workloads, not
   currently supported requests. Keep explicit tail masks and memory guards.

Separate private and per-workgroup shared state comes after this capacity
gate. Multiple resident groups need separate PCs, masks, register banks,
barrier state, and an explicit global-memory ordering contract. Enlarging
the shared array does not implement concurrent workgroups or atomics.
