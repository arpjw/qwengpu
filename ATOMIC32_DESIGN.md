# 32-bit global atomic add gate

The current v4 checkpoint has 512 bytes of global memory, stored as 256
unsigned 16-bit words. `STORE_INDEX` and `STORE_INDEX_HI` publish the two
halves of a 32-bit value in separate instructions. They do not provide an
atomic 32-bit read-modify-write. This document defines the next instruction
gate; **no released checkpoint implements it yet**.

## Proposed instruction contract

`ATOMIC_ADD_INDEX32` uses each active lane's eight-bit index to select a
32-bit global word. Address `index & ~1` holds the low 16 bits; the following
address holds the high 16 bits. The lane's 32-bit register is the unsigned
addend. The instruction returns the old 32-bit memory value in that register
and replaces memory with `(old + addend) mod 2**32`.

An issued warp visits active lanes in ascending lane-ID order. Aliased even
and odd indices select the same pair. A later lane addressing that pair sees
the earlier lane's addition and returns its resulting value. Inactive lanes
leave both memory and their registers unchanged. Workgroup scheduling defines
the order across warps: one instruction finishes its memory commits before
the next scheduled warp instruction begins. The operation requires global
memory and is invalid for scoped private or workgroup-shared addresses.

`qwengpu.atomic32.atomic_add32` is an executable reference transition for
this contract. It copies its input memory, applies one instruction, and
returns the completed memory and lane registers. It is not a model result.
`eval/frozen/atomic32_v1/cross_warp.json` fixes a two-warp barrier and
collision case for the future model run.

With `QWENPUTER_DIR` set to the pinned Qwenputer `experimental/riscv`
directory, run the reference checks from this checkout:

```bash
PYTHONPATH="$QWENPUTER_DIR:$PWD" "$QWENPUTER_DIR/.venv/bin/python" \
  -m unittest tests.test_atomic32 tests.test_compiled_atomic32 -v
```

The [reference test log](results/atomic32/reference-tests.txt) includes the
atomic checks and the existing full-integer, assembler, and 512-byte-memory
tests. It records interpreter and encoding behavior, not model inference.
The [isolated circuit log](results/atomic32/isolated-circuit-tests.txt)
records exact Qwenputer graph evaluation for three and eight lanes.
The [commit protocol log](results/atomic32/commit-protocol-tests.txt)
covers both commit widths and the existing resident-generation tests.

## Circuit and qualification work

1. Allocate an unused ROM encoding in a new layout and add the operation to
   parsing, assembly, and exact token decode without changing v4 encodings.
   The reference-only `--atomic32` assembly flag now uses reserved HALT
   immediate 9. It does not enable the opcode in a compiled model.
2. Gather both limbs for each lane from the same pre-instruction snapshot.
   Compute each lane's old value with the sum of earlier active addends to
   the same pair; wrap every sum at 32 bits. The isolated
   `qwengpu.compiled_atomic32.atomic_add32_results` graph now computes
   these prefix values and the last active writer per pair. Its exact graph
   evaluator passes overflow, masks, distinct pairs, and an eight-lane
   collision. It is not connected to model memory or generation yet.
3. Commit each distinct pair's final low and high limbs before issuing the
   next instruction. Reuse the model-owned memory writeback protocol with
   enough commit slots for both limbs. Keep the ordering within one model
   generation cache and account for any extra generated tokens. An optional
   sixteen-slot generated-writeback selector now emits separate high-limb
   commit tokens after the existing eight slots. Its isolated protocol
   test checks both the old eight-slot and new sixteen-slot sequences.
   The RAM journal is not wired to those high-limb tokens yet.
4. Freeze boundary, carry, overflow, same-pair collision, masked lane,
   cross-warp, interleaved-workgroup, barrier, and native-text handoff cases.
   Compare the complete generated trace and final recurrent memory against
   the independent reference on standalone MPS and merged 9B CUDA.
5. Re-run the v4 inherited programs, 31 common compute requests, memory
   probes, non-square matrix shapes, 16×16 matrix, and native-text gate on
   the new layout. Record compile allocation, model revision, device,
   latency, and provider cost before calling the new checkpoint qualified.

The model may physically emit two limb commits per pair. Atomicity is the
instruction-level visibility rule: the scheduler must not issue another
instruction or text handoff between those commits.
