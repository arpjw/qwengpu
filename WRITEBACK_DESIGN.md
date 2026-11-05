# Bounded internal writeback

`qwengpu/compiled_writeback.py` executes the same 16-slot straight-line ROM
instruction set with no lane READ or result-feedback tokens. After the
25-token initialization prefix, each instruction uses TICK and feedback of
the model-generated PC/halt token. The host never supplies computed lane or
scratch values. The interpreter is used only to verify model execution.

## Why this uses a pipeline

The earlier builder reads lane registers in an early layer and computes
results later. A later layer cannot update the input of an earlier layer on
the next token. Removing the eight output-token writes therefore requires a
different state layout in this compiler.

This builder unrolls 16 execution stages in model depth. Stage 0 reads the
uploaded registers and zero scratch. Every stage reads its predecessor's
retained snapshot, executes the fetched instruction, then writes a new
snapshot. Three Gated DeltaNet heads hold each 320-bit snapshot: eight lane
bytes and 32 scratch bytes. Only the stage matching the model-held PC writes
on a TICK. Earlier snapshots remain available to later stages. A fresh cache
is required for each uploaded program.

This implements internal writeback for bounded straight-line programs. It
is not a cyclic register file: revisiting an earlier PC would not move a later
stage's result back up the model. Branches and arbitrary-length loops are not
supported. Future bounded control flow should distinguish execution stage
from program address and carry masks/stack state forward between stages.
Do not add branches by simply changing the existing PC update.

## Arithmetic and cost

A parallel carry adder computes each carry from the integer sum of its lower
input bits. Every such sum is at most 254, preserving the compiler's bounded
BF16 arithmetic constraints. All 65,536 byte pairs passed the compiler's
exact evaluator. This reduces the full pipeline from an initial 147-layer
layout to 51 circuit layers. The circuit has 58 heads, 15,325 logical
dimensions, and 21,244 FFN units. Reusing residual slots keeps the standalone
model practical; the 9B merger also allocates shared residual slots.

The protocol uses two single-token calls per instruction, compared with 18
for each non-halt instruction in the earlier ROM builder. This is a token
count reduction, not a measured acceleration claim. The deeper circuit adds
weights, computation, and recurrent cache state. Timing must include both
changes before claiming improved speed.

## Acceptance evidence

Seven inputs were frozen in `eval/frozen/writeback_gate_v1` before model
execution. Four are byte-for-byte copies of the previous ROM gate. Additional
cases cover immediate HALT, loads from unwritten scratch, and repeated scratch
overwrites across all four banks. The evaluator inspects the physical snapshot
after every instruction, including HALT, and compares all 40 bytes against
the independent interpreter. It also checks generated PC/halt tokens, the
retained PC state, and final ROM rows. All programs reuse one model object
with fresh caches. Fixture SHA-256 hashes are included in result reports.

The compiler, stock MPS model, and optimized CUDA model passed all seven.
The pretrained 9B merge also passed all seven on CUDA, with identical traces
to the standalone runs. It uses 52 chip layers including the exit marker,
hidden size 5,120, and 52,118,362,112 bytes of weights. Full result reports
and the text comparison are in VALIDATION.md. Existing ROM and divergent builders
remain available as separate variants with their own evidence.


The later [control variant](CONTROL_DESIGN.md) implements the bounded
control-flow construction above with separate step and PC fields. Its
WHILE loops and two-level branch/loop stacks are separate from this original
straight-line builder and have their own result reports.
