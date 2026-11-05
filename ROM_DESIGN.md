# Reusable kernel ROM

`qwengpu/compiled_rom.py` integrates a 16-slot instruction ROM, a model-held
program counter, an eight-lane byte ALU, and four scratch bytes per lane in
stock Qwen3.5 layers. It is a separate builder from the earlier masked,
host-dispatched circuit. The straight-line ROM variant supports `ADD_IMM`,
`XOR_IMM`, `STORE`, `LOAD`, and `HALT`. It does not yet include divergence.

## Instruction encoding and launch

Each ROM slot stores an 11-bit payload: a three-bit opcode and an eight-bit
argument. One upload token encodes address and payload. The loader validates
at most 16 instructions, requires a terminal HALT, rejects unsupported
operations and invalid scratch addresses, and fills unused slots with HALT.
The full chip vocabulary has 65,536 rows. Each launch uses a fresh cache.

The caller uploads initial lane values, all 16 ROM slots, and STATE(0).
It then sends TICK. The circuit reads the PC, fetches the instruction from
recurrent ROM state, decodes it, executes all eight lanes, and generates the
next PC/halt token. The caller feeds that generated token back. After every
non-halt TICK, it sends eight fixed lane READ requests and feeds the generated
lane tokens back to update the lane registers. This schedule is independent
of the fetched opcode, including STORE. The host observes the generated
halt token to stop; it does not dispatch instructions after upload.

One non-halt instruction therefore uses 18 single-token forward calls:
TICK, state feedback, and eight READ/result-feedback pairs. Initialization
uses a 25-token prefix. The next gate removes the lane read/feedback pairs
by writing results directly into register state.

## Circuit and evidence

The standalone circuit has 13 layers, 19 heads, 726 logical dimensions, and
709 FFN units. Target head ratios can change the schedule in the 9B merge.
Four JSON programs were frozen before integration under
`eval/frozen/rom_gate_v1`: arithmetic, scratch round trip, byte wraparound,
and HALT at the final ROM slot. The compiler evaluator and one stock model
object with fresh caches for each program passed all four on MPS and CUDA.
The reports compare every generated state/lane token and actual final ROM,
PC, lane, and scratch recurrent state against the independent interpreter.
The same fixtures also passed in the pinned pretrained 9B merge on CUDA.
That checkpoint uses 14 chip layers including an exit marker, hidden size
4,608, and occupies about 27 GiB.
Fixture hashes are included in each report. CUDA uses optimized DeltaNet
with reference causal convolution.

The original isolated fetch experiment remains in `qwengpu/rom_spike.py`.
Its 100-sequence exploratory evidence is separate from the integrated gate.
The integrated four-program gate establishes bounded reuse of one circuit;
it does not establish arbitrary-program reliability, internal writeback,
branching, or acceleration. See VALIDATION.md for current 9B evidence and
REPRODUCE.md for commands.
