# Common compute interface

This document describes the original 256-byte compute layout. The expanded
512-byte layout and its larger request bounds are specified in
[MEMORY512_DESIGN.md](MEMORY512_DESIGN.md).

The `qwengpu-resident-compute-9b-v1` layout extends globalgrid with two
operations. The same constructed weights accept different uploaded programs,
operands, and supported shapes. Shape selection does not rebuild the model.

`MASK_GLOBAL_LT n` intersects the current mask with `GLOBAL_ID < n`.
It accepts 0 through 64 and lets a partial final workgroup preserve unrelated
global memory. The IDs and predicate are computed in the circuit.

`MATRIX_MAC8 arg` reads two unsigned byte operands from the global bank and
adds their product to the lane's 32-bit accumulator, modulo 2³². Encode
`arg = (K - 1) * 16 + (N - 1) * 4 + k`, with K and N from 1 through 4 and
`0 <= k < K`. Global ID identifies the output row and column. A occupies
word addresses 0 through 31 and B occupies addresses 32 through 47. The
circuit computes `row * K + k` and `k * N + column` and performs both reads.
For a global ID at or above `8 * N`, both matrix operands read as zero,
so this instruction preserves the accumulator. This defines raw uploaded
instructions outside the eight-row operand extent; public kernels also
mask their requested output extent explicitly.
The accumulator's existing indexed instructions load and store C.

Both operations use previously reserved BARRIER immediates in the sixteen-slot
ROM encoding. They decode only in the compute layout. Earlier layouts and
their generated graphs retain their previous semantics.

The public runner compiles three request types to these bounded programs:

| Request | Supported extent | Semantics |
| --- | --- | --- |
| `vector_add` | 0 through 32 values per operand | Elementwise unsigned addition modulo 2³² |
| `reduce` | 0 through 64 values | Sum modulo 2³²; two launches in one cache |
| `matmul` | A is M×K and B is K×N; M ≤ 8, K ≤ 4, N ≤ 4 | Unsigned byte inputs; optional uint32 initial C; uint32 modular accumulation |

A workgroup contains two eight-lane warps. Vector and matrix programs use
one or two sequential workgroups. Reduction uses up to four groups and
first stores each active warp's
partial sum in global memory. Its second uploaded program consumes those
partials without uploading the memory image again. Empty vectors are valid;
an empty reduction returns zero.

The runner packs and uploads inputs, assembles programs, relays model-generated
tokens, and checks results. It does not calculate execution results for the
model. The model chooses warps, advances PCs, changes masks, derives addresses,
performs arithmetic, and emits stores. Scalar code is a validation oracle.

Every saved case checks the complete 128-word global image, including operand
areas and guard words. The frozen manifest precedes model execution. Unit
checks cover vector lengths 0 through 32, reduction lengths 0 through 64,
and 48 matrix shapes; actual model
evidence is recorded separately in VALIDATION.md.

This remains a bounded experimental machine. Global memory is 256 bytes;
the specialized tiled mode has separate fixed matrix banks. Programs have
sixteen slots per warp. Workgroups execute sequentially. Matrix multiplication
uses byte inputs, while ordinary registers, sums, and stored pairs are uint32.
There is no signed multiply, floating point, atomic ISA, simultaneous resident
workgroups, arbitrary matrix geometry, or CUDA compatibility. Tagged writeback
still needs host forwarding. The fixed and planned transports can be compared
on the same checkpoint without changing these semantics.

## General tiled matrix multiply

`matmul_tiled` accepts nonempty byte matrices A and B with
`1 <= M,K,N <= 16` and an optional uint32 C of shape M×N. It returns
`(C + A @ B) mod 2^32`. The launcher tiles M and N into at most 8×4
outputs and splits K into panels of at most four entries. Tail extents use
the same `MATRIX_MAC8` opcode, lane masks, and normal global memory as
the one-tile kernel.

Each output tile starts in its own cache. A subsequent panel sends 192
nibble tags covering exactly the 48 input words and no C word. The model
loads the existing C, performs the new multiply-adds, and writes the result
back into the same cache. The host assembles the output tile grid and
forwards model-generated state and stores as in the single-tile runner.
Each panel's scalar check covers the output and all 256 global words.
