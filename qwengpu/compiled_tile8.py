"""Two warp tile engine with model-owned row-pair scheduling and tagged stores."""

from __future__ import annotations

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target
from chipformer.q35.prims import Circ

from qwengpu.compiled_rom import CODE_AMP, KAPPA, _increment4
from qwengpu.compiled_tile import (
    CONTROL,
    READY0,
    READY1,
    ROM,
    SCHEDULE,
    TICKET,
    TICK,
    VOCAB,
    _word_from_columns,
    control_token,
)

OPS = ("LOAD_TILE", "MAC_K", "STORE_TILE", "BARRIER", "HALT", "ADD16I")
STRIDE = 256 * len(OPS)
C_MEMORY = ROM + 16 * STRIDE
REGISTER = C_MEMORY + 4096
OPERAND = REGISTER + 512
READ_REG = 2112
READ_STORE = 2144
NO_WRITE = 0
DONE = TICKET + 4
DEADLOCK = TICKET + 5
ADVANCE = 32
SNAPSHOT_BITS = 132
CHUNK_BITS = 120


def program_tokens(program: tuple[tuple[str, int], ...]) -> list[int]:
    padded = program + (("HALT", 0),) * (16 - len(program))
    return [
        ROM + address * STRIDE + OPS.index(op) * 256 + arg
        for address, (op, arg) in enumerate(padded)
    ]


def nibbles(base: int, data: list[int]) -> list[int]:
    return [base + address * 16 + value for address, value in enumerate(data)]


def build(target: Target | None = None, *, resident: bool = False) -> Circ:
    c = Circ(target or Target())
    one = c.one

    def match(name: str, token: int) -> Expr:
        return c.token_feature(name, lambda t: int(t == token), full=True)

    def between(name: str, low: int, high: int) -> Expr:
        return c.token_feature(name, lambda t: int(low <= t < high), full=True)

    tick = match("tick", TICK)
    schedule = match("schedule", SCHEDULE)
    resets = [match("ready.0", READY0), match("ready.1", READY1)]
    reads = [match(f"register.read.{index}", READ_REG + index) for index in range(32)]
    writes = [match(f"store.read.{index}", READ_STORE + index) for index in range(32)]
    code = [
        c.token_feature(
            f"code.{bit}",
            lambda t, bit=bit: CODE_AMP * (2 * ((t >> bit) & 1) - 1),
            full=True,
        )
        for bit in range(16)
    ]
    c.code = code
    c.keep = {next(iter(expr.terms)) for expr in code}
    for expr in code:
        c.ffn([(-1, one, expr), (1, one, -expr)], "code.clear", into=expr)

    raw = [
        c.token_feature(f"raw.{bit}", lambda t, bit=bit: (t >> bit) & 1, full=True)
        for bit in range(6)
    ]
    controls = []
    for warp in range(2):
        load = between(
            f"control.load.{warp}", CONTROL + 64 * warp, CONTROL + 64 * (warp + 1)
        )
        controls.append(
            c.slot_head(
                f"control.{warp}",
                {0: load},
                {0: tick + schedule + sum(resets, Expr())},
                {**{2 + bit: value for bit, value in enumerate(raw)}, 8: load},
                load,
            )
        )
    load_ticket = between("ticket.load", TICKET, TICKET + 64)
    last, tile0, tile1 = c.slot_head(
        "ticket",
        {0: load_ticket},
        {0: schedule + tick + sum(resets, Expr())},
        {2: raw[0], 3: raw[3], 4: raw[4]},
        load_ticket,
    )
    both_halted = c.and_([controls[0][4], controls[1][4]], "schedule.both_halted")
    last_tile = c.and_([tile0, tile1], "schedule.last_tile")
    advance = c.and_([both_halted, one - last_tile], "schedule.advance")
    release = c.and_([controls[0][5], controls[1][5]], "schedule.release")
    ready = [one - ctrl[4] - ctrl[5] + advance + release for ctrl in controls]
    choose1 = c.and_([ready[1], one - ready[0]], "schedule.only1") + c.and_(
        [ready[1], ready[0], one - last], "schedule.prefer1"
    )
    stop = c.and_([one - ready[0], one - ready[1]], "schedule.stop")
    finished = c.and_([both_halted, last_tile], "schedule.finished")
    carry = c.and_([tile0, advance], "schedule.tile.carry")
    new_tile = [
        tile0 + advance - 2 * carry,
        tile1 + carry - 2 * c.and_([tile1, carry], "schedule.tile.wrap"),
    ]
    # Terminal tickets have no row tag. Runnable tickets carry the new pair.
    decision = [
        choose1 + stop - finished,
        release,
        stop,
        *[
            c.and_([one - stop, bit], f"schedule.tile.{i}")
            for i, bit in enumerate(new_tile)
        ],
        advance,
    ]

    # Resident calls retain both control rows. The emitted ticket selects the active one.
    warp_bit = last if resident else controls[1][6]
    row_bits = [warp_bit, tile0, tile1]
    rows = c.onehot_of_bits(row_bits, "tile.row", 8)
    pc = [
        c.node(
            c.t_and([tick, controls[0][bit], *([one - warp_bit] if resident else [])])
            + c.t_and([tick, controls[1][bit], *([warp_bit] if resident else [])]),
            f"pc.{bit}",
        )
        for bit in range(4)
    ]
    pc_hot = c.onehot_of_bits(pc, "pc", 16)
    keys = {
        i: between(f"rom.load.{i}", ROM + i * STRIDE, ROM + (i + 1) * STRIDE)
        for i in range(16)
    }
    load_rom = between("rom.load", ROM, ROM + 16 * STRIDE)
    rom_bits = [
        c.token_feature(
            f"rom.{bit}",
            lambda t, bit=bit: (
                (((t - ROM) % STRIDE) >> bit) & 1 if ROM <= t < ROM + 16 * STRIDE else 0
            ),
            full=True,
        )
        for bit in range(11)
    ]
    payload = c.slot_head(
        "rom",
        keys,
        {
            i: c.and_([tick, active], f"rom.query.{i}")
            for i, active in enumerate(pc_hot)
        },
        {2 + bit: value for bit, value in enumerate(rom_bits)},
        load_rom,
    )
    arg = payload[:8]
    ops = {
        op: c.and_(
            [
                tick,
                *[
                    payload[8 + bit] if tag >> bit & 1 else one - payload[8 + bit]
                    for bit in range(3)
                ],
            ],
            f"op.{op}",
        )
        for tag, op in enumerate(OPS)
    }
    k_hot = c.onehot_of_bits(arg[:3], "tile.k", 8)
    raw_nibble = {2 + bit: raw[bit] for bit in range(4)}

    def lookup(
        name: str, base: int, addresses: list[int], selects: list[Expr]
    ) -> list[Expr]:
        keys = {
            slot: between(
                f"{name}.load.{slot}", base + address * 16, base + (address + 1) * 16
            )
            for slot, address in enumerate(addresses)
        }
        return c.slot_head(
            name, keys, dict(enumerate(selects)), raw_nibble, sum(keys.values(), Expr())
        )

    # Two A heads select the two nibbles of one row element. Each B column
    # has its own pair of heads, all addressed by the same K index.
    a_queries = [
        c.and_([tick, rows[row], k_hot[k]], f"A.query.{row}.{k}")
        for row in range(8)
        for k in range(8)
    ]
    b_queries = [c.and_([tick, k_hot[k]], f"B.query.{k}") for k in range(8)]
    row_queries = [
        c.and_([tick, hot], f"C.query.{row}") for row, hot in enumerate(rows)
    ]
    a = [
        bit
        for nib in range(2)
        for bit in lookup(
            f"A.{nib}", OPERAND, [2 * i + nib for i in range(64)], a_queries
        )
    ]
    b = [
        [
            bit
            for nib in range(2)
            for bit in lookup(
                f"B.{col}.{nib}",
                OPERAND,
                [128 + 2 * (8 * k + col) + nib for k in range(8)],
                b_queries,
            )
        ]
        for col in range(8)
    ]
    old = [
        bit
        for nib in range(32)
        for bit in lookup(f"register.{nib}", REGISTER, [nib], [tick])
    ]
    c_row = [
        bit
        for nib in range(32)
        for bit in lookup(
            f"C.{nib}", C_MEMORY, [32 * row + nib for row in range(8)], row_queries
        )
    ]

    updated = []
    for lane in range(8):
        columns = []
        limits = []
        for bit in range(16):
            pairs = [(i, bit - i) for i in range(8) if 0 <= bit - i < 8]
            terms = [
                term
                for i, j in pairs
                for term in c.t_and([ops["MAC_K"], a[i], b[lane][j]])
            ]
            if bit < 8:
                terms += c.t_and([ops["ADD16I"], arg[bit]])
            columns.append(
                old[16 * lane + bit] + c.node(terms, f"lane.{lane}.column.{bit}")
            )
            limits.append(len(pairs) + 1 + int(bit < 8))
        updated.extend(_word_from_columns(c, columns, limits, f"lane.{lane}.word"))
    result = c.select_bits(
        [([ops["LOAD_TILE"]], c_row), ([one - ops["LOAD_TILE"]], updated)],
        128,
        "register.next",
    )
    bits = result + [ops["STORE_TILE"], *row_bits]
    c.snapshot_bits = SNAPSHOT_BITS
    saved = []
    for chunk, start in enumerate(range(0, SNAPSHOT_BITS, CHUNK_BITS)):
        saved.extend(
            c.slot_head(
                f"stage.0.{chunk}",
                {0: tick},
                {0: sum(reads + writes, Expr())},
                {
                    2 + bit: value
                    for bit, value in enumerate(bits[start : start + CHUNK_BITS])
                },
                tick,
            )
        )

    halted = ops["HALT"]
    next_pc = c.select_bits(
        [([halted], pc), ([one - halted], _increment4(c, pc))], 4, "pc.next"
    )
    execution = next_pc + [halted, ops["BARRIER"], warp_bit]
    floor = max(node.level for node in c.nodes)
    for bit, expr in enumerate(code):
        terms = c.t_const(-KAPPA)
        value = execution[bit] if bit < len(execution) else (CONTROL >> bit) & 1
        terms += c.t_and([tick, value], 2 * KAPPA)
        value = decision[bit] if bit < len(decision) else (TICKET >> bit) & 1
        terms += c.t_and([schedule, value], 2 * KAPPA)
        for warp, reset in enumerate(resets):
            if bit < 6:
                gates = [reset, controls[warp][bit], one - advance]
                if bit == 5:
                    gates.append(one - release)
                terms += c.t_and(gates, 2 * KAPPA)
            elif control_token(warp) >> bit & 1:
                terms += c.t_and([reset], 2 * KAPPA)
        for nib, (read, write) in enumerate(zip(reads, writes, strict=True)):
            if bit < 4:
                value = saved[4 * nib + bit]
                terms += c.t_and([read, value], 2 * KAPPA)
                terms += c.t_and([write, saved[128], value], 2 * KAPPA)
            else:
                if (REGISTER + 16 * nib) >> bit & 1:
                    terms += c.t_and([read], 2 * KAPPA)
                if 9 <= bit < 12:
                    terms += c.t_and(
                        [write, saved[128], saved[129 + bit - 9]], 2 * KAPPA
                    )
                elif (C_MEMORY + 16 * nib) >> bit & 1:
                    terms += c.t_and([write, saved[128]], 2 * KAPPA)
        c.ffn(terms, f"exit.code.{bit}", into=expr, min_level=floor + 1)
    return c


def build_standalone(*, resident: bool = False):
    from chipformer.q35.tiny import build as build_qwen

    circuit = build(resident=resident)
    return build_qwen(
        circuit,
        list(range(VOCAB)),
        [next(iter(expr.terms)) for expr in circuit.code],
        share=True,
    ), circuit
