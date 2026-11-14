# Recurrent writeback without host data tokens

The qualified scheduler reads registers and memory before the ALU.
An ALU result cannot directly change the recurrent state of an earlier
layer. The runner feeds back emitted register tokens to bridge that
boundary. Indexed stores have a shorter dependency chain: their address,
value, and collision decision can be resolved before the memory heads.

The implementation targets indexed global, workgroup-shared, and lane-private stores. A store already
has its value in a register. TICK reads that value and the saved index,
applies the mask and collision rule, and journals the granted lanes before
the main RAM layers. A fixed lane COMMIT reads the journal and writes the
same recurrent RAM rows that future loads query. The command has no address
or value payload. It replaces four address/value nibble tokens with one
lane token for each 16-bit word and removes the four store read requests.
Trace inspection decodes the existing model snapshot without sending
those values to the model. The model still receives generated register
tags.

Three standalone CUDA programs checked 308 stores (180 global and 128
scoped) and every final RAM word. No store address/value tags went in or
came out. The address walk included high-limb stores, masks and wrapped
offsets; the interleaved and isolation programs included shared and
private traffic. See `results/model_stores_cuda/compact_standalone.json`.

A more ambitious replacement would target the four resident warp banks. Keep the
existing warp state heads as the base and add a late recurrent journal.
Each journal row stores the entire state of one physical warp, plus a
valid bit. On TICK, the ALU reads that warp's state and produces the same
snapshot as today. A following fixed COMMIT token reads the late snapshot
and saves it in the journal. On the next TICK the ALU must see that
journal before it computes, which creates a layer-order cycle.

An alternative avoids the cycle by unrolling **two execution phases** in
depth and feeding a compact activation projection through the existing
input path. That remains a host relay; calling it recurrent writeback
would be inaccurate. Another avoids the cycle through same-layer
convolution taps for functions that fit the head's linear projection
(such as copy and masked constant writes). It does not implement the
general arithmetic or arbitrary scatter arbitration in this ISA.

A promising bounded extension is to make the early read heads query
state that a host-independent late path produced within a previously
unrolled execution stage, as in `compiled_control`. This duplicates the
large stateful engine for each stage and currently cannot sustain the
32-slot, 256-step execution envelope. A smaller practical step is
internal commit to a late **output mirror** for verification and final
readout, while keeping the early read source. It does not eliminate
cyclic forwarding and is not a completion criterion on its own.

The useful contract is measurable: report host state and store feedback
tokens, model calls, and equivalent trace bodies. The fused dispatch
transport reduces calls and suppresses unchanged control relays while
keeping model-produced register and memory writes on the wire. Closing
the cyclic writeback gate requires an architectural change beyond that
transport optimization.

### Reproduce the model-owned store checks

Build from the pinned base and compiler recorded in the merged manifest.
Run the MPS command only when an Apple machine is available for validation.

```bash
python scripts/eval_model_stores.py --device mps --out stores-mps.json
python -m qwengpu.merge9b "$base" --internal-stores --out "$model"
python scripts/eval_model_stores.py --model "$model" --device cuda --out stores-cuda.json
python scripts/check_model_stores.py stores-mps.json stores-cuda.json
bash scripts/qualify_model_stores_cuda.sh "$base" "$new_model" "$reports" "$(git rev-parse HEAD)"
python scripts/check_model_store_suite.py --reports "$reports"
```

The global and scoped launch inputs are uploaded as before. Normal stores
use the model's grant mask and fixed commands. This does not change the
legacy tile-mode store path, which retains its existing transport.

A second option for *register* feedback is a runtime recurrent edge. The
early state heads read the previous late snapshot through a cache
adapter that copies a complete model-owned tensor between forwards.
It would remove token encoding and data-dependent host decisions, but
it would no longer be a stock weights-only Qwen model. Keeping that
distinction explicit matters: an edge implemented in the inference
runtime is different from a store committed by the checkpoint itself.


The optional `--transport cache-register-edge` prototype uses a static
mapping from each snapshot chunk to the four recurrent bit columns of
each register nibble. The only runtime selector is the model's issued
physical warp. It copies complete bit columns on the accelerator and
leaves recurrent ballast in place. Neither the reference interpreter
nor a CPU-computed value is used for the copy. Trace decoding remains
independent of the update. This protocol is an experiment; stock
inference still uses the original register tag relay.

In the standalone CUDA publication trace, the adapter retained every
observable result while moving from 4,050 to 3,610 input tokens and 125
to 97 forward calls. It issued no register read or register writeback
tokens. The fused version combines this adapter with the independent
model-store commit path.

The follow-up also removes register-plan requests. The runtime reads the
already-materialized model snapshot to audit changes and copies the
snapshot columns directly. In the same standalone publication case this
uses 3,082 input tokens and 97 forward calls, with zero register-plan,
read, or feedback tokens. See
`results/cache_register_edge_cuda/standalone_planless.json`.

On merged Qwen3.5-9B with model-owned stores, the fused adapter matched
all three frozen resident scheduling traces. Publication used 2,462
input tokens and 89 model calls versus 3,602 and 117 with stock
model-store transport. The barrier and deadlock cases used 1,873 and
1,628 input tokens, down from 2,841 and 2,024. All three issued no
register plan, read, or feedback tokens and no RAM address/value tags.
The adapter copies tensors outside the stock model; it does not solve
weights-only cyclic register writeback.

## Stock generation on the bounded control circuit

The bounded control experiment can use an active PC/stage token as the
clock for the next instruction. The launch prompt carries the lane inputs,
ROM, control links, and first PC/stage. Stock greedy generation then
consumes each predicted PC/stage token as its next input and stops when
the circuit emits a halt or exhaustion token. Stage-local recurrent heads
retain all lane, scratch, mask, and stack state. The caller can audit
those rows after generation without using them to drive execution.

This configuration uses one depth stage per executed instruction and has
the original sixteen-stage limit. It tests a way to remove the per-step
host protocol. The integrated integer GPU still requires a separate
solution for feeding its four late register banks into later instructions
without cloning the full datapath once per step.

```bash
python -m qwengpu.eval_control --generate --device cuda --out control-generate-cuda.json
python -m qwengpu.merge9b "$base" --control-generate --out "$model"
python -m qwengpu.eval_control "$model" --generate --device cuda --out control-generate-9b.json
python scripts/check_control_generation.py results/control_cuda_fla.json \
  control-generate-cuda.json control-generate-9b.json
```

Run `bash scripts/qualify_control_generation_cuda.sh "$base" "$model" "$reports"`
for the compiler checks, both CUDA replays, and the fixed text corpus.
The merged artifact can be rebuilt using the source and compiler pins in
`results/control_generation_cuda/manifest.json`.

## Reusing the integrated register banks through generation

The pinned Qwenputer RV core shows a second stock-model approach. Its exit
emits value chunks and a commit token, and the register heads accept those
tokens on later forwards. The core reuses the same register rows across
instructions. QwenGPU already has a matching bank-tag range and a late
state snapshot. The next experiment can reuse both.

1. Decode a generated resident ticket as the instruction clock. Save the
   snapshot, then emit the new control word.
2. Let the control word request the first changed snapshot nibble. On a
   register tag, the existing head commits that nibble while the late
   readout selects the next changed nibble. The token identifies the
   current bank and nibble, so generation needs no external cursor.
3. After the last register update, issue the fixed store commit tokens
   selected by the same model snapshot. Only then schedule the next
   instruction. Serialize the barrier-ready control updates before the
   next ticket.

Start with a two-warp cross-group dependency whose consumer needs a
previously written high-limb register and a global store. Require the
same physical register rows to be reused at least twice. Compare every
generated record and the final RAM to both the current integrated path
and the interpreter. A generated data token is still a token-level
writeback edge. This test would establish that stock generation drives
that edge without a host protocol or a runtime tensor copy.

The `--resident-generate` layout uses disjoint generated token ranges
for register writes, control updates, barrier releases, and lane commits.
This lets the initial prompt upload the same state and program with
no execution until the final schedule token. The schedule snapshot
preserves the barrier release decision while four control tokens
commit it in order. Register updates follow ascending nibble order.
Store commits follow ascending lane order.

The v2 layout uses the same dispatch path for sequential and interleaved
workgroups. A sequential launch starts with physical banks 0 and 1 active.
After both halt, the circuit emits the next logical group number, clears
each active register bank with a separate token, and resumes scheduling.
The group cursor supplies the IDs and scoped addresses for later groups.
The two uploaded ROMs and global RAM remain resident across transitions.

The v3 protocol also accepts the legacy tile launch descriptor. Its
resident row cursor starts at the requested first row pair. After both
warps halt, a generated row token advances that cursor and clears the
two PCs while preserving the register banks. Tile-nibble and indexed
memory ports see the same model-owned store, including stores to
operand memory. Each launch still enters stock generation once.

The legacy looping stress fixture uses a host instruction budget.
The stopping rule counts only issued instruction tickets and stops
before executing the next ticket. The model still decides when to
emit a ticket, halt, or report deadlock.

```bash
bash scripts/qualify_resident_generation_cuda.sh "$base" "$model" "$reports" \
  "$(git rev-parse HEAD)"
```

The retained-cache workload checker uses the existing kernel compiler
and runs one stock generation call per launch. It feeds the same cache
into the reduction's second pass and into later K panels. The second
pass uploads zero global words. Later panels upload the operand prefix
and keep the output words in recurrent RAM. The checker compares
each launch to the interpreter and each final result to scalar arithmetic.

```bash
python scripts/eval_resident_generation_workloads.py "$model" \
  --out "$reports/workloads.json" eval/frozen/tiled_compute_v1/{3x5x2,9x7x5}.json
```

The native-text handoff check runs the reduction producer, replaces the
pretrained layers' cache entries, and compares four text prompts to
fresh-cache controls on the same weights. It audits all global-memory
mirrors, register nibbles, ROM rows, and control words before and after
each prompt. The consumer then uses the same GPU cache without a RAM
upload and must match a separate cold consumer.

```bash
python scripts/eval_resident_generation_handoff.py "$model" \
  --out "$reports/handoff.json"
python -m qwengpu.eval_resident_generation --model "$model" --out "$reports/single.json" \
  eval/frozen/full_integer_v1/indexed_words.json eval/frozen/memory512_v1/*.json
python scripts/eval_resident_generation_workloads.py "$model" \
  --suite eval/frozen/memory512_compute_v1/cases.json \
  --case vector_0 --case vector_17 --case vector_32 \
  --case reduce_0 --case reduce_17 --case reduce_32 \
  --case matmul_1x1x1 --case matmul_8x4x4 --out "$reports/compute.json"
```

```bash
bash scripts/qualify_cache_register_edge.sh "$model" "$stock_reports" "$edge_reports"
```

The same runtime adapter also matched both panels of the 3×5 by 5×2
matrix multiply on merged CUDA. Each panel committed 12 output words
without register or RAM address/value tokens; the second panel retained
C and uploaded only its A/B operands. See
`results/cache_register_edge_cuda/tiled_3x5x2.json`.
