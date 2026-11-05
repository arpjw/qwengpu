"""Bounded 16-stage ROM machine with internal register/scratch writeback.

Each instruction position owns a recurrent snapshot. Data moves forward in
layer depth, so no lane values cross the token interface. This is bounded
unrolling, not a reusable cyclic register file for arbitrary-length loops.
"""

from __future__ import annotations

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target
from chipformer.q35.prims import Circ
from qwengpu.compiled_rom import (
    RomSIMT,
    ROM_SLOTS,
    SCRATCH_SLOTS,
    VOCAB,
    KAPPA,
    STATE_BASE,
    OP_ADD,
    OP_XOR,
    OP_STORE,
    OP_LOAD,
    OP_HALT,
    _increment4,
)
from qwengpu.reference import LANES

SNAPSHOT_BITS = LANES * 8 * (1 + SCRATCH_SLOTS)
CHUNK_BITS = 120


def _add_byte(c: Circ, left: list[Expr], right: list[Expr], name: str) -> list[Expr]:
    """Parallel carries from bounded integer sums of each lower-bit prefix."""
    result = []
    for bit, (a, b) in enumerate(zip(left, right, strict=True)):
        pair = a + b - 2 * c.and_([a, b], f"{name}.pair.{bit}")
        if not bit:
            result.append(pair)
            continue
        lower = sum(((left[j] + right[j]) * (1 << j) for j in range(bit)), Expr())
        carry = c.node(
            c.t_ge(lower, 1 << bit, 0, (1 << (bit + 1)) - 2), f"{name}.carry.{bit}"
        )
        both = c.and_([pair, carry], f"{name}.both.{bit}")
        result.append(pair + carry - 2 * both)
    return result


def build(target: Target | None = None) -> RomSIMT:
    c = RomSIMT(target)
    one = c.one
    for code in c.code:
        c.ffn([(-1, one, code), (1, one, -code)], "code.clear", into=code)
    initial = []
    for lane in range(LANES):
        initial.extend(
            c.slot_head(
                f"initial.{lane}",
                {0: c.is_a[lane]},
                {0: c.is_tick},
                {2 + b: x for b, x in enumerate(c.raw_byte)},
                c.is_a[lane],
            )
        )
    pc = c.slot_head(
        "pc",
        {0: c.is_state},
        {0: c.is_tick},
        {2 + b: x for b, x in enumerate(c.raw_state)},
        c.is_state,
    )
    positions = c.onehot_of_bits(pc[:4], "pc.onehot", ROM_SLOTS)
    queries = {
        i: c.and_([c.is_tick, positions[i]], f"rom.query.{i}") for i in range(ROM_SLOTS)
    }
    payload = c.slot_head(
        "rom",
        dict(enumerate(c.is_rom_address)),
        queries,
        {2 + b: x for b, x in enumerate(c.raw_rom)},
        c.is_rom_load,
    )
    arg = payload[:8]
    flags = {
        op: c.and_(
            [
                c.is_tick,
                *[
                    payload[8 + b] if op >> b & 1 else one - payload[8 + b]
                    for b in range(3)
                ],
            ],
            f"op.{op}",
        )
        for op in (OP_ADD, OP_XOR, OP_STORE, OP_LOAD, OP_HALT)
    }
    slots = c.onehot_of_bits(arg[:2], "scratch.slot", SCRATCH_SLOTS)
    store_slot = [
        c.and_([flags[OP_STORE], slots[s]], f"store.slot.{s}")
        for s in range(SCRATCH_SLOTS)
    ]
    previous = initial + [Expr()] * (SNAPSHOT_BITS - len(initial))
    c.snapshots = []
    for stage in range(ROM_SLOTS):
        active = queries[stage]
        lanes = [previous[lane * 8 : (lane + 1) * 8] for lane in range(LANES)]
        memory = [
            previous[64 + s * 64 : 64 + (s + 1) * 64] for s in range(SCRATCH_SLOTS)
        ]
        next_lanes = []
        for lane in range(LANES):
            summed = _add_byte(c, lanes[lane], arg, f"s{stage}.sum.{lane}")
            xored = [
                lanes[lane][b]
                + arg[b]
                - 2 * c.and_([lanes[lane][b], arg[b]], f"s{stage}.xor.{lane}.{b}")
                for b in range(8)
            ]
            loaded = c.select_bits(
                [
                    ([slots[s]], memory[s][lane * 8 : (lane + 1) * 8])
                    for s in range(SCRATCH_SLOTS)
                ],
                8,
                f"s{stage}.load.{lane}",
            )
            chosen = c.select_bits(
                [
                    ([flags[OP_ADD]], summed),
                    ([flags[OP_XOR]], xored),
                    ([flags[OP_LOAD]], loaded),
                    (
                        [one - flags[OP_ADD] - flags[OP_XOR] - flags[OP_LOAD]],
                        lanes[lane],
                    ),
                ],
                8,
                f"s{stage}.lane.{lane}",
            )
            next_lanes.extend(chosen)
        next_memory = []
        for slot in range(SCRATCH_SLOTS):
            next_memory.extend(
                c.select_bits(
                    [
                        ([store_slot[slot]], previous[:64]),
                        ([one - store_slot[slot]], memory[slot]),
                    ],
                    64,
                    f"s{stage}.mem.{slot}",
                )
            )
        values = next_lanes + next_memory
        previous = []
        for chunk, start in enumerate(range(0, SNAPSHOT_BITS, CHUNK_BITS)):
            data = values[start : start + CHUNK_BITS]
            # Write only this stage's instruction. Always read the retained
            # snapshot on TICK, allowing later stages to consume it next tick.
            previous.extend(
                c.slot_head(
                    f"stage.{stage}.{chunk}",
                    {0: active},
                    {0: c.is_tick},
                    {2 + b: x for b, x in enumerate(data)},
                    active,
                )
            )
        c.snapshots.append(previous)
    halt = pc[4] + flags[OP_HALT] - c.and_([pc[4], flags[OP_HALT]], "halt.or")
    inc = _increment4(c, pc[:4])
    next_pc = c.select_bits([([halt], pc[:4]), ([one - halt], inc)], 4, "next.pc")
    floor = max(node.level for node in c.nodes)
    for b, code in enumerate(c.code):
        value = (
            next_pc[b]
            if b < 4
            else halt
            if b == 4
            else one
            if STATE_BASE >> b & 1
            else Expr()
        )
        c.ffn(
            c.t_const(-KAPPA) + c.t_and([c.is_tick, value], 2 * KAPPA),
            f"exit.code{b}",
            into=code,
            min_level=floor,
        )
    return c


def build_standalone(*, dtype=None):
    import torch
    from chipformer.q35.tiny import build as build_qwen

    circuit = build()
    model = build_qwen(
        circuit,
        list(range(VOCAB)),
        [next(iter(e.terms)) for e in circuit.code],
        dtype=torch.bfloat16 if dtype is None else dtype,
        share=True,
    )
    return model, circuit
