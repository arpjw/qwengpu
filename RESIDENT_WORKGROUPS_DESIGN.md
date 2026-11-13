# Interleaving resident workgroups

The next scheduler gate targets two concurrently resident groups of two
warps each. Preserve the sequential mode for the inherited kernels. A
launch descriptor selects the interleaved mode and the number of groups.
Keep global and scoped memory shared across the same recurrent cache
using the qualified address rules. The physical warp bank is
`2 * workgroup + local_warp`. The two uploaded programs remain common to
all groups and use only the local warp bit for ROM selection.

Bank each physical warp's PC, halt and barrier state, registers, index,
lane mask, branch frames, and loop frames. Update one physical bank on
each tick. Start the next schedule scan after the last issued warp, with
wraparound. A barrier releases only when both warps of that group are
waiting. One halted warp paired with a waiting warp leaves that group
blocked. Other groups continue until no runnable warp remains. The
terminal ticket distinguishes all halted from deadlock. A step limit
continues to apply to the whole launch.

Reuse each state head's row selection rather than cloning all state
heads. Use bank-tagged control and register packets so the host can
forward them without decoding their values. Extend the control-token
bank bits and reserve a ticket range that cannot overlap sequential
tile, workgroup, or terminal tickets. Keep chip opcodes and the public
per-warp program length unchanged. The launch recorder should expose
both physical warp and logical group while preserving existing trace
records for sequential mode.

Freeze three initial cases. The first must interleave at least one
instruction of each group before either can halt and must retain
divergent masks and unequal PCs. The second must release one group's
barrier while the other group remains blocked. The third publishes
distinct group-local results into global RAM, then consumes the
results in a second launch. Use an independent ready-queue interpreter
for every ticket and state transition. Compare standalone MPS and
merged Qwen3.5-9B before replaying the inherited sequential traces.

The prototype selects the new mode with `"interleaved": true` and
`"workgroups": 2` in the raw launch. Without that descriptor the model
uses the existing sequential scheduler. The host forwards the returned
schedule, control and state tokens. It never selects a runnable warp.
The launch returns four accumulator banks in physical warp order.

```bash
python -m qwengpu.run_resident eval/frozen/resident_interleaved_v1/interleaved_publish.json \
  --resident-interleaved --device mps --check-reference --trace --out publish.json
python scripts/check_interleaved.py publish.json
```

## Fused dispatch transport

An optional `--transport fused-dispatch` combines the requested state and
store reads with the next scheduler decision. The prior instruction's
model-generated control token enters the call before the reads. A
scheduler decision uses control state only, so its output does not depend
on the outstanding data writes. The next TICK receives those exact
writeback tags before it can read any register or memory word. On a
terminal ticket or step limit, any prefetched updates are flushed first.

The standalone CUDA publication trace has identical instructions, control
decisions, masks, write tags, and final RAM. It uses 89 forward calls versus
125 with the state/store plan, at the same 4,050 input tokens. The host
still forwards generated data tags. This change reduces model calls and
does not move the recurrent data writes into the model or establish a
speedup.
