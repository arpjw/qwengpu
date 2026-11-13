# Recurrent writeback without host data tokens

The qualified scheduler reads registers and memory in early model layers.
Its ALU and scatter arbitration run later. Those later layers cannot
directly change the recurrent tensors of earlier layers. The current
runner bridges the boundary by feeding emitted data tokens.

The first replacement targets the four resident warp banks. Keep the
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
