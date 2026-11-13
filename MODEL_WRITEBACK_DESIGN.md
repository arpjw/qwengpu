# Recurrent writeback without host data tokens

The qualified scheduler reads registers and memory in early model layers.
Its ALU and scatter arbitration run later. Those later layers cannot
directly change the recurrent tensors of earlier layers. The current
runner bridges the boundary by feeding emitted data tokens.

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

```bash
python scripts/eval_model_stores.py --device mps --out stores-mps.json
python -m qwengpu.merge9b "$base" --internal-stores --out "$model"
python scripts/eval_model_stores.py --model "$model" --device cuda --out stores-cuda.json
python scripts/check_model_stores.py stores-mps.json stores-cuda.json
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
