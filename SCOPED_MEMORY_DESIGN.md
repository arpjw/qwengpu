# Scoped memory on the resident processor

Use explicit shared and private instructions on programmable workgroup
launches. The 256 global words keep their existing addresses and lifetime
across dependent launches. The proposed first layout adds 16 shared words
per workgroup and one private word per thread. Shared addresses use the low
four bits of the lane's saved word index. Both warps in a workgroup can
address the same words. Private addresses come from the model's workgroup,
warp, and lane IDs, so a program has no way to select a peer's private word.

The 128 words of scoped storage fit the indexed heads currently used by
legacy tiled launches. Reuse those heads with a dedicated token range
`[0x18000, 0x1a000)` in programmable launches. Each token carries a physical
scoped word, nibble offset, and nibble value. Shared rows 0–63 use
`16 * workgroup + (index & 15)`. Private rows 64–127 use
`64 + 16 * workgroup + 8 * warp + lane`. Every scoped launch initializes
those rows, including a dependent launch that keeps its global RAM. Separate
token ranges keep scoped feedback away from global and tiled output writes.

The scheduler already exposes the active workgroup and warp. The model
computes the physical address, arbitrates scatter collisions using the
existing lowest active lane rule, and tags each store. Shared writes become
visible to a peer after the existing barrier. Private stores have one
physical destination per lane. The host continues to forward emitted tags
and the interpreter independently checks the physical words.

Freeze a fixture that starts with distinct seeds, writes the same shared
index in each group, publishes across a barrier, and puts both warp-private
values and the shared result in global memory. Check wrapped shared indices,
inactive lanes, the last physical word, all untouched words, and a second
launch with scoped reset and retained global RAM. Compare both MPS and the
merged checkpoint to the independent interpreter and replay the ALU32
qualification suite.
