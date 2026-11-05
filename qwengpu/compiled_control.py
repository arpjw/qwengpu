"""Bounded loops and nested divergence in a model-state ROM pipeline."""

from __future__ import annotations
from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target
from chipformer.q35.prims import Circ, from_bits
from qwengpu.compiled_rom import KAPPA, CODE_AMP, _increment4
from qwengpu.compiled_writeback import _add_byte
from qwengpu.control_reference import OPS, MAX_STEPS, validate
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
OP = dict(zip(OPS, range(len(OPS))))

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


def program_tokens(program: tuple[Instruction, ...]) -> list[int]:
    links = validate(program)
    instructions = list(program) + [Instruction("HALT")] * (16 - len(program))
    rom = []
    for address, ins in enumerate(instructions):
        arg = ins.arg // 8 if ins.op in ("STORE", "LOAD") else ins.arg
        rom.append(ROM_BASE + address * STRIDE + OP[ins.op] * 256 + arg)
    return rom + [
        LINK_BASE + address * 16 + links.get(address, 0) for address in range(16)
    ]


class ControlSIMT(Circ):
    def __init__(self, target: Target | None = None):
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
        self.raw_rom = [
            f(
                f"rom.bit.{b}",
                lambda t, b=b: (
                    (((t - ROM_BASE) % STRIDE) >> b) & 1
                    if ROM_BASE <= t < ROM_END
                    else 0
                ),
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


def build(target: Target | None = None, *, resumable: bool = False) -> ControlSIMT:
    c = ControlSIMT(target)
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
            for index in range(CHECKPOINT_BYTES)
        ]
        resume = c.token_feature(
            "resume", lambda token: int(token == RESUME), full=True
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
    }
    slots = c.onehot_of_bits(arg[:2], "scratch.slot", 4)
    stores = [c.and_([flags["STORE"], slots[s]], f"store.{s}") for s in range(4)]
    if resumable:
        previous = []
        for index in range(CHECKPOINT_BYTES):
            start = CHECKPOINT_BASE + index * 256
            write = c.token_feature(
                f"restore.{index}",
                lambda token, start=start: int(start <= token < start + 256),
                full=True,
            )
            width = min(8, SNAPSHOT_BITS - index * 8)
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
    for s in range(MAX_STEPS):
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
        next_lanes = []
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
            next_lanes.extend(
                c.select_bits(
                    [
                        ([enables["ADDI"]], added),
                        ([enables["XORI"]], xored),
                        ([enables["LOAD"]], loaded),
                        ([one - sum(enables.values(), Expr())], lanes[i]),
                    ],
                    8,
                    f"{prefix}.lane.{i}",
                )
            )
        next_memory = []
        for slot in range(4):
            for lane in range(8):
                write = c.and_(
                    [stores[slot], mask[lane]], f"{prefix}.store.{slot}.{lane}"
                )
                next_memory.extend(
                    c.select_bits(
                        [
                            ([write], lanes[lane]),
                            ([one - write], memory[slot][lane * 8 : (lane + 1) * 8]),
                        ],
                        8,
                        f"{prefix}.memory.{slot}.{lane}",
                    )
                )
        values = (
            next_lanes
            + next_memory
            + next_mask
            + branch_depth
            + branch_frames
            + loop_depth
            + loop_frames
        )
        assert len(values) == SNAPSHOT_BITS
        previous = []
        for chunk, start in enumerate(range(0, SNAPSHOT_BITS, CHUNK_BITS)):
            previous.extend(
                c.slot_head(
                    f"stage.{s}.{chunk}",
                    {0: active},
                    {
                        0: c.is_tick
                        + (sum(export, Expr()) if s == MAX_STEPS - 1 else Expr())
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
    exhausted = c.and_([stage_hot[15], c.is_tick, one - halt], "exhausted")
    next_pc = c.select_bits(
        [
            ([halt], pc),
            ([flags["ENDWHILE"] + leave], link),
            ([one - halt - flags["ENDWHILE"] - leave], _increment4(c, pc)),
        ],
        4,
        "next.pc",
    )
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
    for b, code in enumerate(c.code):
        value = result[b] if b < 10 else one if STATE_BASE >> b & 1 else Expr()
        terms = c.t_const(-KAPPA) + c.t_and([c.is_tick, value], 2 * KAPPA)
        if resumable:
            resume_bit = pc[b] if b < 4 else (STATE_BASE >> b) & 1
            terms += c.t_and([resume, resume_bit], 2 * KAPPA)
            for index, request in enumerate(export):
                if b < 8:
                    position = index * 8 + b
                    if position < SNAPSHOT_BITS:
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


def build_standalone(*, dtype=None, resumable: bool = False):
    import torch
    from chipformer.q35.tiny import build as build_qwen

    c = build(resumable=resumable)
    model = build_qwen(
        c,
        list(range(VOCAB)),
        [next(iter(e.terms)) for e in c.code],
        dtype=torch.bfloat16 if dtype is None else dtype,
        share=True,
    )
    return model, c
