"""Bounded loops and nested divergence in a model-state ROM pipeline."""

from __future__ import annotations
from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target
from chipformer.q35.prims import Circ, from_bits
from qwengpu.compiled_rom import KAPPA, CODE_AMP, _increment4
from qwengpu.compiled_writeback import _add_byte
from qwengpu.control_reference import ISA_OPS, OPS, WORKGROUP_OPS, MAX_STEPS, validate
from qwengpu.reference import Instruction

VOCAB = 65536
STATE_BASE = 4096
STATE_COUNT = 1024
TICK = 5120
LINK_BASE = 6144
ROM_BASE = 8192
STRIDE = len(OPS) * 256
ROM_END = ROM_BASE + 16 * STRIDE
SNAPSHOT_BITS = 380
CHUNK_BITS = 120
OP = {name: index for index, name in enumerate(OPS + ISA_OPS + WORKGROUP_OPS)}
# Extra instructions use formerly invalid HALT arguments on the upload wire.
ISA_UPLOAD = {
    name: OP["HALT"] * 256 + index
    for index, name in enumerate(ISA_OPS + WORKGROUP_OPS, 1)
}
ISA_DECODE = {payload: OP[name] * 256 for name, payload in ISA_UPLOAD.items()}
ISA_SNAPSHOT_BITS = SNAPSHOT_BITS + 64

CHECKPOINT_BYTES = (SNAPSHOT_BITS + 7) // 8
CHECKPOINT_BASE = ROM_END
EXPORT_BASE = TICK + 1
RESUME = EXPORT_BASE + CHECKPOINT_BYTES


def state_token(
    pc: int, stage: int, halted: bool = False, exhausted: bool = False
) -> int:
    if not 0 <= pc < 16 or not 0 <= stage < MAX_STEPS or (halted and exhausted):
        raise ValueError("invalid control state")
    return STATE_BASE + pc + (stage << 4) + (int(halted) << 8) + (int(exhausted) << 9)


def program_tokens(
    program: tuple[Instruction, ...], *, extended: bool = False, workgroup: bool = False
) -> list[int]:
    links = validate(program, extended=extended, workgroup=workgroup)
    instructions = list(program) + [Instruction("HALT")] * (16 - len(program))
    rom = []
    for address, ins in enumerate(instructions):
        arg = ins.arg // 8 if ins.op in ("STORE", "LOAD") else ins.arg
        payload = ISA_UPLOAD.get(ins.op, OP[ins.op] * 256 + arg)
        rom.append(ROM_BASE + address * STRIDE + payload)
    return rom + [
        LINK_BASE + address * 16 + links.get(address, 0) for address in range(16)
    ]


class ControlSIMT(Circ):
    def __init__(self, target: Target | None = None, *, extended: bool = False):
        super().__init__(target or Target())
        f = self.token_feature
        self.is_tick = f("is_tick", lambda t: int(t == TICK), full=True)
        self.is_state = f(
            "is_state",
            lambda t: int(STATE_BASE <= t < STATE_BASE + STATE_COUNT),
            full=True,
        )
        self.raw_state = [
            f(
                f"state.{b}",
                lambda t, b=b: (
                    ((t - STATE_BASE) >> b) & 1
                    if STATE_BASE <= t < STATE_BASE + STATE_COUNT
                    else 0
                ),
                full=True,
            )
            for b in range(10)
        ]
        self.raw_byte = [
            f(f"byte.{b}", lambda t, b=b: (t >> b) & 1, full=True) for b in range(8)
        ]
        self.is_a = [
            f(
                f"load.{lane}",
                lambda t, lane=lane: int(lane * 256 <= t < (lane + 1) * 256),
                full=True,
            )
            for lane in range(8)
        ]
        self.rom_keys = [
            f(
                f"rom.load.{i}",
                lambda t, i=i: int(
                    ROM_BASE + i * STRIDE <= t < ROM_BASE + (i + 1) * STRIDE
                ),
                full=True,
            )
            for i in range(16)
        ]
        self.is_rom = f("is_rom", lambda t: int(ROM_BASE <= t < ROM_END), full=True)

        def rom_payload(token: int) -> int:
            if not ROM_BASE <= token < ROM_END:
                return 0
            payload = (token - ROM_BASE) % STRIDE
            return ISA_DECODE.get(payload, payload) if extended else payload

        self.raw_rom = [
            f(
                f"rom.bit.{b}",
                lambda t, b=b: (rom_payload(t) >> b) & 1,
                full=True,
            )
            for b in range(12)
        ]
        self.link_keys = [
            f(
                f"link.load.{i}",
                lambda t, i=i: int(LINK_BASE + i * 16 <= t < LINK_BASE + (i + 1) * 16),
                full=True,
            )
            for i in range(16)
        ]
        self.is_link = f(
            "is_link", lambda t: int(LINK_BASE <= t < LINK_BASE + 256), full=True
        )
        self.raw_link = [
            f(
                f"link.bit.{b}",
                lambda t, b=b: (
                    ((t - LINK_BASE) >> b) & 1
                    if LINK_BASE <= t < LINK_BASE + 256
                    else 0
                ),
                full=True,
            )
            for b in range(4)
        ]
        self.code = [
            f(
                f"code.{b}",
                lambda t, b=b: CODE_AMP * (2 * ((t >> b) & 1) - 1),
                full=True,
            )
            for b in range(16)
        ]
        self.keep = {next(iter(e.terms)) for e in self.code}
        self.snapshots: list[list[Expr]] = []
        self.snapshot_bits = ISA_SNAPSHOT_BITS if extended else SNAPSHOT_BITS


def _less_byte(c: Circ, left: list[Expr], right: list[Expr], name: str) -> Expr:
    return c.node(c.t_lt(from_bits(left) - from_bits(right), 0, -255, 255), name)


def _stack(c, depth, frames, push, pop, new_frame, name):
    """Two-entry stack; callers provide structurally validated instructions."""
    hot = c.onehot_of_bits(depth, f"{name}.depth", 3)
    top = c.select_bits(
        [([hot[i + 1]], frame) for i, frame in enumerate(frames)],
        len(new_frame),
        f"{name}.top",
    )
    push_at = [c.and_([push, hot[i]], f"{name}.push.{i}") for i in range(2)]
    pop_at = [c.and_([pop, hot[i + 1]], f"{name}.pop.{i}") for i in range(2)]
    new_depth = c.select_bits(
        [
            ([push_at[0]], 1),
            ([push_at[1]], 2),
            ([pop_at[0]], 0),
            ([pop_at[1]], 1),
            ([c.one - push - pop], depth),
        ],
        2,
        f"{name}.next_depth",
    )
    result = []
    for i, frame in enumerate(frames):
        result.extend(
            c.select_bits(
                [([push_at[i]], new_frame), ([c.one - push_at[i] - pop_at[i]], frame)],
                len(frame),
                f"{name}.frame.{i}",
            )
        )
    return top, new_depth, result


def _product_byte(
    c: Circ, left: list[Expr], right: list[Expr], name: str
) -> list[Expr]:
    """Modulo-256 product with parallel bit columns and a nibble carry."""
    columns = [
        c.node(
            [
                term
                for i in range(column + 1)
                for term in c.t_and([left[i], right[column - i]])
            ],
            f"{name}.column.{column}",
        )
        for column in range(8)
    ]

    return _byte_from_columns(c, columns, list(range(1, 9)), name)


def _byte_from_columns(
    c: Circ, columns: list[Expr], limits: list[int], name: str
) -> list[Expr]:
    def digit(value: Expr, upper: int, bit: int) -> Expr:
        half = 1 << bit
        terms = []
        for start in range(half, upper + 1, 2 * half):
            terms += c.t_ge(value, start, 0, upper)
            if start + half <= upper:
                terms += c.t_ge(value, start + half, 0, upper, coef=-1)
        return c.node(terms, f"{name}.bit.{bit}")

    low = sum((columns[i] * (1 << i) for i in range(4)), Expr())
    low_limit = sum(limits[i] * (1 << i) for i in range(4))
    carry = c.node(
        [
            term
            for threshold in range(16, low_limit + 1, 16)
            for term in c.t_ge(low, threshold, 0, low_limit)
        ],
        f"{name}.carry",
    )
    result = [
        digit(
            sum((columns[i] * (1 << i) for i in range(bit + 1)), Expr()),
            sum(limits[i] * (1 << i) for i in range(bit + 1)),
            bit,
        )
        for bit in range(4)
    ]
    for bit in range(4, 8):
        high = carry + sum(
            (columns[i] * (1 << (i - 4)) for i in range(4, bit + 1)), Expr()
        )
        upper = low_limit // 16 + sum(
            limits[i] * (1 << (i - 4)) for i in range(4, bit + 1)
        )
        # The threshold bit within the high nibble is independent of the low nibble.
        half = 1 << (bit - 4)
        terms = []
        for threshold in range(half, upper + 1, half):
            sign = 1 if (threshold // half) & 1 else -1
            terms += c.t_ge(high, threshold, 0, upper, coef=sign)
        result.append(c.node(terms, f"{name}.bit.{bit}"))
    return result


def build(
    target: Target | None = None,
    *,
    resumable: bool = False,
    extended: bool = False,
    workgroup: bool = False,
) -> ControlSIMT:
    extended = extended or workgroup
    resumable = resumable or extended
    steps = 1 if workgroup else MAX_STEPS
    c = ControlSIMT(target, extended=extended)
    byte_count = (c.snapshot_bits + 7) // 8
    one = c.one
    for code in c.code:
        c.ffn([(-1, one, code), (1, one, -code)], "code.clear", into=code)
    export = []
    resume = Expr()
    if resumable:
        export = [
            c.token_feature(
                f"export.{index}",
                lambda token, index=index: int(token == EXPORT_BASE + index),
                full=True,
            )
            for index in range(byte_count)
        ]
        resume = c.token_feature(
            "resume", lambda token: int(token == EXPORT_BASE + byte_count), full=True
        )
    initial = []
    for lane in range(0 if resumable else 8):
        initial.extend(
            c.slot_head(
                f"initial.{lane}",
                {0: c.is_a[lane]},
                {0: c.is_tick},
                {2 + b: v for b, v in enumerate(c.raw_byte)},
                c.is_a[lane],
            )
        )
    state = c.slot_head(
        "control",
        {0: c.is_state},
        {0: c.is_tick + resume},
        {2 + b: v for b, v in enumerate(c.raw_state)},
        c.is_state,
    )
    pc, stage = state[:4], state[4:8]
    pc_hot = c.onehot_of_bits(pc, "pc", 16)
    stage_hot = c.onehot_of_bits(stage, "stage", 16)
    queries = {i: c.and_([c.is_tick, pc_hot[i]], f"rom.query.{i}") for i in range(16)}
    payload = c.slot_head(
        "rom",
        dict(enumerate(c.rom_keys)),
        queries,
        {2 + b: v for b, v in enumerate(c.raw_rom)},
        c.is_rom,
    )
    link = c.slot_head(
        "link",
        dict(enumerate(c.link_keys)),
        queries,
        {2 + b: v for b, v in enumerate(c.raw_link)},
        c.is_link,
    )
    arg = payload[:8]
    flags = {
        op: c.and_(
            [
                c.is_tick,
                *[
                    payload[8 + b] if index >> b & 1 else one - payload[8 + b]
                    for b in range(4)
                ],
            ],
            f"op.{op}",
        )
        for op, index in OP.items()
        if op in OPS or (extended and op in ISA_OPS) or workgroup
    }
    slots = c.onehot_of_bits(arg[:2], "scratch.slot", 4)
    stores = [c.and_([flags["STORE"], slots[s]], f"store.{s}") for s in range(4)]
    if resumable:
        previous = []
        for index in range(byte_count):
            start = CHECKPOINT_BASE + index * 256
            write = c.token_feature(
                f"restore.{index}",
                lambda token, start=start: int(start <= token < start + 256),
                full=True,
            )
            width = min(8, c.snapshot_bits - index * 8)
            previous.extend(
                c.slot_head(
                    f"restore.{index}",
                    {0: write},
                    {0: c.is_tick},
                    {2 + bit: c.raw_byte[bit] for bit in range(width)},
                    write,
                )
            )
    else:
        previous = (
            initial + [Expr()] * 256 + [one] * 8 + [Expr()] * (SNAPSHOT_BITS - 328)
        )
    exits = []
    for s in range(steps):
        prefix = f"s{s}"
        active = c.and_([c.is_tick, stage_hot[s]], f"{prefix}.write")
        lanes = [previous[i * 8 : (i + 1) * 8] for i in range(8)]
        memory = [previous[64 + i * 64 : 64 + (i + 1) * 64] for i in range(4)]
        mask = previous[320:328]
        less = [_less_byte(c, lanes[i], arg, f"{prefix}.less.{i}") for i in range(8)]
        predicate = [c.and_([mask[i], less[i]], f"{prefix}.pred.{i}") for i in range(8)]
        any_lane = c.node(c.t_ge(sum(predicate, Expr()), 1, 0, 8), f"{prefix}.any")
        enter = c.and_([flags["WHILE_LT"], any_lane], f"{prefix}.enter")
        leave = c.and_([flags["WHILE_LT"], one - any_lane], f"{prefix}.leave")
        exits.append(c.and_([active, leave], f"{prefix}.exit_active"))
        branch_top, branch_depth, branch_frames = _stack(
            c,
            previous[328:330],
            [previous[330:346], previous[346:362]],
            flags["IF_LT"],
            flags["ENDIF"],
            mask + predicate,
            f"{prefix}.branch",
        )
        loop_top, loop_depth, loop_frames = _stack(
            c,
            previous[362:364],
            [previous[364:372], previous[372:380]],
            enter,
            flags["ENDWHILE"],
            mask,
            f"{prefix}.loop",
        )
        else_mask = [
            c.and_([branch_top[i], one - branch_top[8 + i]], f"{prefix}.else.{i}")
            for i in range(8)
        ]
        next_mask = c.select_bits(
            [
                ([flags["IF_LT"] + enter], predicate),
                ([flags["ELSE"]], else_mask),
                ([flags["ENDIF"]], branch_top[:8]),
                ([flags["ENDWHILE"]], loop_top),
                (
                    [
                        one
                        - flags["IF_LT"]
                        - enter
                        - flags["ELSE"]
                        - flags["ENDIF"]
                        - flags["ENDWHILE"]
                    ],
                    mask,
                ),
            ],
            8,
            f"{prefix}.mask",
        )
        if workgroup:
            sum_columns = [
                c.node(
                    [
                        term
                        for lane in range(8)
                        for term in c.t_and([mask[lane], lanes[lane][bit]])
                    ],
                    f"{prefix}.reduce.column.{bit}",
                )
                for bit in range(8)
            ]
            reduction = _byte_from_columns(c, sum_columns, [8] * 8, f"{prefix}.reduce")
        next_lanes = []
        next_r1 = []
        scatter_addresses = []
        for i in range(8):
            added = _add_byte(c, lanes[i], arg, f"{prefix}.add.{i}")
            xored = [
                lanes[i][b]
                + arg[b]
                - 2 * c.and_([lanes[i][b], arg[b]], f"{prefix}.xor.{i}.{b}")
                for b in range(8)
            ]
            loaded = c.select_bits(
                [([slots[j]], memory[j][i * 8 : (i + 1) * 8]) for j in range(4)],
                8,
                f"{prefix}.load.{i}",
            )
            enables = {
                op: c.and_([mask[i], flags[op]], f"{prefix}.{op}.{i}")
                for op in ("ADDI", "XORI", "LOAD")
            }
            choices = [
                ([enables["ADDI"]], added),
                ([enables["XORI"]], xored),
                ([enables["LOAD"]], loaded),
            ]
            if extended:
                r1 = previous[SNAPSHOT_BITS + i * 8 : SNAPSHOT_BITS + (i + 1) * 8]
                address_matches = [
                    [r1[bit] if byte >> bit & 1 else one - r1[bit] for bit in range(5)]
                    for byte in range(32)
                ]
                scatter_addresses.append(address_matches)
                # Stagger the gather lanes so each FFN has room for slot reuse.
                gather_floor = c.level(*r1, *previous[64:320]) + int(i >= 4)
                gather = [
                    c.ffn(
                        [
                            term
                            for byte, address in enumerate(address_matches)
                            for term in c.t_and(
                                [*address, previous[64 + 8 * byte + bit]]
                            )
                        ],
                        f"{prefix}.gather.{i}.{bit}",
                        min_level=gather_floor,
                    )
                    for bit in range(8)
                ]
                product = _product_byte(c, lanes[i], r1, f"{prefix}.mul.{i}")
                for op, value in (("GATHER_R1", gather), ("MUL_R1", product)):
                    enables[op] = c.and_([mask[i], flags[op]], f"{prefix}.{op}.{i}")
                    choices.append(([enables[op]], value))
                move = c.and_([mask[i], flags["MOV_R1"]], f"{prefix}.move.{i}")
                next_r1.extend(
                    c.select_bits(
                        [([move], lanes[i]), ([one - move], r1)], 8, f"{prefix}.r1.{i}"
                    )
                )
            if workgroup:
                enables["REDUCE_SUM"] = c.and_(
                    [mask[i], flags["REDUCE_SUM"]], f"{prefix}.reduce.active.{i}"
                )
                choices.append(([enables["REDUCE_SUM"]], reduction))
            choices.append(([one - sum(enables.values(), Expr())], lanes[i]))
            next_lanes.extend(c.select_bits(choices, 8, f"{prefix}.lane.{i}"))
        next_memory = []
        for slot in range(4):
            for lane in range(8):
                write = c.and_(
                    [stores[slot], mask[lane]], f"{prefix}.store.{slot}.{lane}"
                )
                choices = [([write], lanes[lane])]
                keep = one - write
                if workgroup:
                    address = slot * 8 + lane
                    candidates = [
                        c.and_(
                            [
                                flags["SCATTER_R1"],
                                mask[source],
                                *scatter_addresses[source][address],
                            ],
                            f"{prefix}.scatter.{address}.{source}",
                        )
                        for source in range(8)
                    ]
                    choices.extend(
                        (
                            [hit, *[one - earlier for earlier in candidates[:source]]],
                            lanes[source],
                        )
                        for source, hit in enumerate(candidates)
                    )
                    keep = (
                        one
                        - write
                        - c.node(
                            c.t_ge(sum(candidates, Expr()), 1, 0, 8),
                            f"{prefix}.scatter.any.{address}",
                        )
                    )
                choices.append(([keep], memory[slot][lane * 8 : (lane + 1) * 8]))
                next_memory.extend(
                    c.select_bits(choices, 8, f"{prefix}.memory.{slot}.{lane}")
                )
        values = (
            next_lanes
            + next_memory
            + next_mask
            + branch_depth
            + branch_frames
            + loop_depth
            + loop_frames
            + next_r1
        )
        assert len(values) == c.snapshot_bits
        previous = []
        for chunk, start in enumerate(range(0, c.snapshot_bits, CHUNK_BITS)):
            previous.extend(
                c.slot_head(
                    f"stage.{s}.{chunk}",
                    {0: active},
                    {
                        0: c.is_tick
                        + (sum(export, Expr()) if s == steps - 1 else Expr())
                    },
                    {
                        2 + b: v
                        for b, v in enumerate(values[start : start + CHUNK_BITS])
                    },
                    active,
                )
            )
        c.snapshots.append(previous)
    leave = sum(exits, Expr())
    halt = flags["HALT"]
    next_pc = c.select_bits(
        [
            ([halt], pc),
            ([flags["ENDWHILE"] + leave], link),
            ([one - halt - flags["ENDWHILE"] - leave], _increment4(c, pc)),
        ],
        4,
        "next.pc",
    )
    if workgroup:
        barrier = flags["BARRIER"]
        result = next_pc + [Expr()] * 4 + [halt + barrier, barrier]
    else:
        exhausted = c.and_([stage_hot[15], c.is_tick, one - halt], "exhausted")
        next_stage = c.select_bits(
            [
                ([halt + exhausted], stage),
                ([one - halt - exhausted], _increment4(c, stage)),
            ],
            4,
            "next.stage",
        )
        result = next_pc + next_stage + [halt, exhausted]
    floor = max(node.level for node in c.nodes)
    if extended:
        # Reserve an FFN sublayer for output and residual cleanup.
        floor += 1
    for b, code in enumerate(c.code):
        value = result[b] if b < 10 else one if STATE_BASE >> b & 1 else Expr()
        terms = c.t_const(-KAPPA) + c.t_and([c.is_tick, value], 2 * KAPPA)
        if resumable:
            resume_bit = pc[b] if b < 4 else (STATE_BASE >> b) & 1
            terms += c.t_and([resume, resume_bit], 2 * KAPPA)
            for index, request in enumerate(export):
                if b < 8:
                    position = index * 8 + b
                    if position < c.snapshot_bits:
                        terms += c.t_and([request, previous[position]], 2 * KAPPA)
                elif (CHECKPOINT_BASE + index * 256) >> b & 1:
                    terms += c.t_and([request], 2 * KAPPA)
        c.ffn(
            terms,
            f"exit.code.{b}",
            into=code,
            min_level=floor,
        )
    return c


def build_standalone(
    *,
    dtype=None,
    resumable: bool = False,
    extended: bool = False,
    workgroup: bool = False,
):
    import torch
    from chipformer.q35.tiny import build as build_qwen

    c = build(resumable=resumable, extended=extended, workgroup=workgroup)
    model = build_qwen(
        c,
        list(range(VOCAB)),
        [next(iter(e.terms)) for e in c.code],
        dtype=torch.bfloat16 if dtype is None else dtype,
        share=True,
    )
    return model, c
