# First reusable kernel ROM design

This is the proposed integration design. The current SIMT checkpoint uses
host-supplied instruction tokens. A separate [16-slot ROM spike](qwengpu/rom_spike.py)
already demonstrated uploaded instruction payloads, a model-held PC, and
model-generated instruction and next-PC tokens on stock Qwen3.5 with MPS.
Its three payload sequences and recurrent ROM snapshots passed in
[`results/rom_spike_mps.json`](results/rom_spike_mps.json). It does not yet
drive the SIMT ALU or run on CUDA. The integration below would let one set
of model weights execute different uploaded programs in separate caches.

## Bounded instruction format

Start with 16 instruction slots. Each slot stores a 3-bit opcode and an 8-bit
argument in a Gated DeltaNet head keyed by the 4-bit slot address. The first
five opcodes are `ADD_IMM`, `XOR_IMM`, `STORE`, `LOAD`, and `HALT`.
`STORE` and `LOAD` use slot indices 0 through 3. Arithmetic and scratch
semantics match the current reference interpreter. The loader rejects any
unsupported opcode, invalid memory slot, missing `HALT`, or program longer
than 16 instructions before the model is called.

Program upload uses one raw token per instruction. The token encodes address,
opcode, and argument. A fresh inference cache is required for each launch,
because an omitted address must not inherit a byte from a previous program.
The loader should fill unused slots with `HALT` so every address has a defined
value. The current isolated ROM spike uploads all 16 slots for this reason.
The host may upload tokens and launch the kernel; it does not choose or send
individual instructions during execution.

Packing address and an 11-bit instruction into one upload token gives 32,768
possible upload IDs before other trace tokens. This simplifies the first
integration, but widens the merged embedding and output vocabulary. A later
two-token opcode/argument encoding could use fewer IDs at the cost of an
extra ROM read or head. Measure the merged checkpoint size and token latency
before changing the proven packed encoding.

## Execution protocol

1. Upload `ROM_LOAD(address, opcode, argument)` tokens and an initial
   `STATE(pc=0, halted=0)` token.
2. Send `TICK`. A ROM head reads the instruction at the stored PC. Fixed gates
   decode the opcode. Arithmetic reads all eight lane registers and applies
   the immediate in parallel; memory accesses lane-banked scratch.
3. The model emits a new `STATE` token with the next PC and halt bit. Feed it
   back to update the PC head. During the first ROM gate, the caller also
   requests eight lane-result tokens after operations that change lanes and
   feeds them back, as the current circuit does.
4. Repeat until the model emits the halted state token. The caller observes
   that token and stops. It never computes the next PC or instruction.

The first version can use the current output-token writeback to isolate ROM
fetch/decode from register-writeback changes. Gate 2 in [SCOPE.md](SCOPE.md)
then removes those eight read and feedback pairs per lane-changing operation.

## Circuit changes

Add one 16-row instruction ROM head and one PC/state head. Route ROM argument
bits directly to the eight ALU lanes; this removes the need for eight
host-supplied `LOAD_B` tokens per immediate instruction. The ROM opcode gates
select ADD, XOR, STORE, LOAD, or HALT. The current lane, mask, result, and
scratch heads can remain the starting point. The circuit must emit both lane
results and the next state token without code collisions. Only one active
mask is needed for straight-line Gate 1; divergent control comes later.

The integration should use this order of dependencies within a `TICK`:
PC head read, 16-way address selection, ROM head read, opcode decode, vector
ALU or scratch operation, result-head write, and next-PC/halt output. The
generated state token is fed back before any lane `READ` tokens. These reads
then use the existing result head and output-token writeback. A `STORE` needs
no lane reads; a `LOAD` does. The raw-token ranges for ROM upload, state,
`TICK`, lane reads, and lane outputs must be disjoint, and the output code
must still represent old mask and lane tokens exactly.

Implement the integrated circuit as a separate builder first. Keep the
current validated checkpoint and its evidence intact until the integrated
builder passes. Compare a single loaded program in the circuit IR, then
multiple programs with one stock model object and fresh caches, then the
pretrained 9B merge. Inspect actual ROM, PC, lane, and scratch recurrent
rows rather than relying only on generated tokens.

QwenCPU's [dynamic ROM implementation](https://github.com/arpjw/qwencpu/blob/main/gdn_cpu/qwen8_dynamic.py)
is a useful construction reference. The GPU circuit will need its own 11-bit
instruction value, vector ALU routing, and output protocol. Copying the CPU
ROM alone would not make the GPU instruction stream autonomous.

## Exact acceptance test

Freeze at least four programs before a model run: arithmetic only, scratch
round trip, mixed ADD/XOR with wraparound, and a program that halts from the
last slot. Run all four with one standalone model object and fresh caches,
then repeat on optimized CUDA. Compare every generated PC/halt token, each
lane output, and actual ROM, register, and scratch cache state to the
independent interpreter. After this passes, rebuild the pinned 9B merge and
repeat. Record circuit dimensions and trace latency before considering a
larger ROM or control flow.

The four JSON inputs are already frozen in
[`eval/frozen/rom_gate_v1`](eval/frozen/rom_gate_v1). Their SHA-256 hashes and
host-dispatched baseline results are in
[`results/rom_gate_host_mps.json`](results/rom_gate_host_mps.json). All four
passed on standalone MPS. This baseline establishes expected lane and scratch
behavior; it does not exercise ROM-driven execution. The future integrated
builder should use these unchanged inputs first.
