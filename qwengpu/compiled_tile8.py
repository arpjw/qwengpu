"""Two warp tile engine with model-owned row-pair scheduling and tagged stores."""

from __future__ import annotations

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target
from chipformer.q35.prims import Circ, from_bits

from qwengpu.compiled_control import _stack
from qwengpu.compiled_rom import CODE_AMP, KAPPA, _increment4
from qwengpu.compiled_tile import (
    CONTROL,
    READY0,
    READY1,
    ROM,
    SCHEDULE,
    TICK,
    TICKET,
    VOCAB,
    _word_from_columns,
    control_token,
)

OPS = ("LOAD_TILE", "MAC_K", "STORE_TILE", "BARRIER", "HALT", "ADD16I")
INDEX_OPS = ("SAVE_INDEX", "LOAD_INDEX", "STORE_INDEX")
MASK_OPS = ("MASK_ALL", "MASK_BIT")
BRANCH_OPS = ("IF_BIT", "ELSE", "ENDIF")
LOOP_OPS = ("WHILE_BIT", "ENDWHILE")
STRIDE = 256 * len(OPS)
C_MEMORY = ROM + 16 * STRIDE
REGISTER = C_MEMORY + 4096
OPERAND = REGISTER + 512
READ_REG = 2112
READ_STORE = 2144
READ_INDEX = 2176
INDEX_REGISTER = 0x9000
INDEX_OPERAND = 0xA000
INDEX_WORDS = 128
READ_MASK = READ_INDEX + 16
MASK_REGISTER = 0x9100
MASK_OFFSET = 196
BRANCH_OFFSET = MASK_OFFSET + 8
BRANCH_REGISTER = 0x9200
READ_BRANCH = READ_MASK + 2
BRANCH_NIBBLES = 9
LOOP_OFFSET = BRANCH_OFFSET + 4 * BRANCH_NIBBLES
LOOP_REGISTER = 0x9300
READ_LOOP = READ_BRANCH + BRANCH_NIBBLES
LOOP_NIBBLES = 5
WARP_DIRTY_OFFSET = LOOP_OFFSET + 4 * LOOP_NIBBLES
ROM_BANK = READ_LOOP + LOOP_NIBBLES
ROM_PAYLOAD_BITS = 12
READ_PLAN = ROM_BANK + 2
PLAN_TAG = 0xD000
GRID = 0xDE00
ROW_PAIRS = 4
PLAN_BYTES = 8
WARP_BANKS = (0xB000, 0xC000)
WARP_TAGS = tuple(
    base + 16 * nib
    for base, count in (
        (REGISTER, 32),
        (INDEX_REGISTER, 16),
        (MASK_REGISTER, 2),
        (BRANCH_REGISTER, BRANCH_NIBBLES),
        (LOOP_REGISTER, LOOP_NIBBLES),
    )
    for nib in range(count)
)
NO_WRITE = 0
DONE = TICKET + 4
DEADLOCK = TICKET + 5
ADVANCE = 32
SNAPSHOT_BITS = 132
CHUNK_BITS = 120


def program_tokens(program: tuple[tuple[str, int], ...]) -> list[int]:
    padded = program + (("HALT", 0),) * (16 - len(program))

    links = {}
    opens = []
    for address, (op, _) in enumerate(program):
        if op == "WHILE_BIT":
            opens.append(address)
        elif op == "ENDWHILE":
            start = opens.pop()
            links[start] = address + 1
            links[address] = start

    def wire(address: int, op: str, arg: int) -> int:
        # The reserved HALT immediates extend the ROM without changing its stride.
        halt = OPS.index("HALT") * 256
        if op in INDEX_OPS:
            return halt + 1 + INDEX_OPS.index(op)
        if op == "MASK_ALL":
            return halt + 4
        if op == "MASK_BIT":
            return halt + 16 + arg
        if op == "IF_BIT":
            return halt + 32 + arg
        if op == "ELSE":
            return halt + 5
        if op == "ENDIF":
            return halt + 6
        # A validated loop exits past its ENDWHILE, so its packed argument is nonzero.
        if op == "WHILE_BIT":
            return OPS.index("LOAD_TILE") * 256 + 16 * links[address] + arg
        if op == "ENDWHILE":
            return OPS.index("BARRIER") * 256 + 1 + links[address]
        return OPS.index(op) * 256 + arg

    return [
        ROM + address * STRIDE + wire(address, op, arg)
        for address, (op, arg) in enumerate(padded)
    ]


def rom_payload(
    token: int, *, indexed: bool, masked: bool, divergent: bool, looping: bool
) -> int:
    if not ROM <= token < ROM + 16 * STRIDE:
        return 0
    wire = (token - ROM) % STRIDE
    tag, arg = divmod(wire, 256)
    if indexed and OPS[tag] == "HALT" and 1 <= arg <= len(INDEX_OPS):
        return (len(OPS) + arg - 1) * 256
    if masked and OPS[tag] == "HALT":
        if arg == 4:
            return (len(OPS) + len(INDEX_OPS)) * 256
        if 16 <= arg < 32:
            return (len(OPS) + len(INDEX_OPS) + 1) * 256 + arg - 16
    if divergent and OPS[tag] == "HALT":
        branch_tag = len(OPS) + len(INDEX_OPS) + len(MASK_OPS)
        if 32 <= arg < 48:
            return branch_tag * 256 + arg - 32
        if arg in (5, 6):
            return (branch_tag + arg - 4) * 256
    if looping:
        loop_tag = len(OPS) + len(INDEX_OPS) + len(MASK_OPS) + len(BRANCH_OPS)
        if OPS[tag] == "LOAD_TILE" and arg >= 32:
            return loop_tag * 256 + arg
        if OPS[tag] == "BARRIER" and 1 <= arg <= 16:
            return (loop_tag + 1) * 256 + arg - 1
    return wire


def nibbles(base: int, data: list[int]) -> list[int]:
    return [base + address * 16 + value for address, value in enumerate(data)]


def build(
    target: Target | None = None,
    *,
    resident: bool = False,
    indexed: bool = False,
    masked: bool = False,
    divergent: bool = False,
    looping: bool = False,
    resident_rom: bool = False,
    resident_state: bool = False,
    resident_delta: bool = False,
    resident_plan: bool = False,
    resident_io: bool = False,
    resident_grid: bool = False,
) -> Circ:
    resident_io = resident_io or resident_grid
    resident_plan = resident_plan or resident_io
    resident_delta = resident_delta or resident_plan
    resident_state = resident_state or resident_delta
    resident_rom = resident_rom or resident_state
    looping = looping or resident_rom
    divergent = divergent or looping
    masked = masked or divergent
    indexed = indexed or masked
    resident = resident or indexed
    operand_base = INDEX_OPERAND if indexed else OPERAND
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
    index_reads = (
        [match(f"index.read.{nib}", READ_INDEX + nib) for nib in range(16)]
        if indexed
        else []
    )
    mask_reads = (
        [match(f"mask.read.{nib}", READ_MASK + nib) for nib in range(2)]
        if masked
        else []
    )
    branch_reads = (
        [
            match(f"branch.read.{nib}", READ_BRANCH + nib)
            for nib in range(BRANCH_NIBBLES)
        ]
        if divergent
        else []
    )
    loop_reads = (
        [match(f"loop.read.{nib}", READ_LOOP + nib) for nib in range(LOOP_NIBBLES)]
        if looping
        else []
    )
    plan_reads = (
        [
            match(f"state.plan.{byte}", READ_PLAN + byte)
            for byte in range(PLAN_BYTES + int(resident_io))
        ]
        if resident_plan
        else []
    )
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
    if resident_grid:
        load_grid = between("grid.load", GRID, GRID + ROW_PAIRS)
        limit = c.slot_head(
            "grid",
            {0: load_grid},
            {0: schedule + sum(resets, Expr())},
            {2: raw[0], 3: raw[1]},
            load_grid,
        )
        difference = from_bits([tile0, tile1]) - from_bits(limit)
        last_tile = c.node(c.t_eq(difference, 0, -3, 3), "schedule.last_tile")
    else:
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

    op_names = (
        OPS
        + (INDEX_OPS if indexed else ())
        + (MASK_OPS if masked else ())
        + (BRANCH_OPS if divergent else ())
        + (LOOP_OPS if looping else ())
    )
    op_width = (len(op_names) - 1).bit_length()
    rom_bits = [
        c.token_feature(
            f"rom.{bit}",
            lambda t, bit=bit: (
                (
                    rom_payload(
                        t,
                        indexed=indexed,
                        masked=masked,
                        divergent=divergent,
                        looping=looping,
                    )
                    >> bit
                )
                & 1
            ),
            full=True,
        )
        for bit in range(8 + op_width)
    ]
    if resident_rom:
        bank0 = match("rom.bank.0", ROM_BANK)
        bank1 = match("rom.bank.1", ROM_BANK + 1)
        bank_load = bank0 + bank1
        (upload_bank,) = c.slot_head(
            "rom.bank", {0: bank_load}, {0: load_rom}, {2: bank1}, bank_load
        )
        keys = {
            16 * warp + address: c.and_([key, selected], f"rom.write.{warp}.{address}")
            for warp, selected in enumerate((one - upload_bank, upload_bank))
            for address, key in keys.items()
        }
        queries = {
            16 * warp + address: c.and_(
                [tick, active, selected], f"rom.query.{warp}.{address}"
            )
            for warp, selected in enumerate((one - warp_bit, warp_bit))
            for address, active in enumerate(pc_hot)
        }
    else:
        queries = {
            i: c.and_([tick, active], f"rom.query.{i}")
            for i, active in enumerate(pc_hot)
        }
    payload = c.slot_head(
        "rom",
        keys,
        queries,
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
                    for bit in range(op_width)
                ],
            ],
            f"op.{op}",
        )
        for tag, op in enumerate(op_names)
    }
    k_hot = c.onehot_of_bits(arg[:3], "tile.k", 8)
    raw_nibble = {2 + bit: raw[bit] for bit in range(4)}

    input_keys: dict[int, Expr] = {}

    def lookup(
        name: str, base: int, addresses: list[int], selects: list[Expr]
    ) -> list[Expr]:
        keys = {}
        for slot, address in enumerate(addresses):
            start = base + address * 16
            if start not in input_keys:
                input_keys[start] = between(f"{name}.load.{slot}", start, start + 16)
            keys[slot] = input_keys[start]
        return c.slot_head(
            name, keys, dict(enumerate(selects)), raw_nibble, sum(keys.values(), Expr())
        )

    state_queries = (
        [
            c.and_([tick, one - warp_bit], "state.query.0"),
            c.and_([tick, warp_bit], "state.query.1"),
        ]
        if resident_state
        else [tick]
    )

    def state_lookup(name: str, base: int, nib: int) -> list[Expr]:
        if resident_state:
            offset = WARP_TAGS.index(base) + nib
            return lookup(
                name, 0, [bank // 16 + offset for bank in WARP_BANKS], state_queries
            )
        return lookup(name, base, [nib], state_queries)

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
            f"A.{nib}", operand_base, [2 * i + nib for i in range(64)], a_queries
        )
    ]
    b = [
        [
            bit
            for nib in range(2)
            for bit in lookup(
                f"B.{col}.{nib}",
                operand_base,
                [128 + 2 * (8 * k + col) + nib for k in range(8)],
                b_queries,
            )
        ]
        for col in range(8)
    ]
    old = [
        bit
        for nib in range(32)
        for bit in state_lookup(f"register.{nib}", REGISTER, nib)
    ]
    c_row = [
        bit
        for nib in range(32)
        for bit in lookup(
            f"C.{nib}", C_MEMORY, [32 * row + nib for row in range(8)], row_queries
        )
    ]

    active = [one] * 8
    next_mask = []
    next_branch = []
    next_loop = []
    if masked:
        active = [
            bit
            for nib in range(2)
            for bit in state_lookup(f"mask.{nib}", MASK_REGISTER, nib)
        ]
        selectors = c.onehot_of_bits(arg[:4], "mask.bit", 16)

        def selected_bit(lane: int, gate: Expr) -> Expr:
            return c.node(
                [
                    term
                    for selector, value in zip(
                        selectors, old[16 * lane : 16 * (lane + 1)], strict=True
                    )
                    for term in c.t_and([selector, value, gate])
                ],
                f"mask.selected.{lane}",
            )

        if divergent:
            predicate_op = (
                ops["MASK_BIT"] + ops["IF_BIT"] + (ops["WHILE_BIT"] if looping else 0)
            )
            predicate = [selected_bit(lane, predicate_op) for lane in range(8)]
            stack = [
                bit
                for nib in range(BRANCH_NIBBLES)
                for bit in state_lookup(f"branch.{nib}", BRANCH_REGISTER, nib)
            ]
            then_mask = [
                c.and_([enabled, test], f"branch.then.{lane}")
                for lane, (enabled, test) in enumerate(
                    zip(active, predicate, strict=True)
                )
            ]
            top, depth, frames = _stack(
                c,
                stack[:2],
                [stack[4:20], stack[20:36]],
                ops["IF_BIT"],
                ops["ENDIF"],
                active + then_mask,
                "branch",
            )
            next_branch = depth + [Expr(), Expr()] + frames
            else_mask = [
                c.and_([parent, one - chosen], f"branch.else.{lane}")
                for lane, (parent, chosen) in enumerate(
                    zip(top[:8], top[8:], strict=True)
                )
            ]
            mask_sources = [
                ([ops["IF_BIT"]], then_mask),
                ([ops["ELSE"]], else_mask),
                ([ops["ENDIF"]], top[:8]),
                ([ops["MASK_BIT"]], predicate),
                ([ops["MASK_ALL"]], 255),
            ]
            keep_mask = (
                one
                - ops["IF_BIT"]
                - ops["ELSE"]
                - ops["ENDIF"]
                - ops["MASK_BIT"]
                - ops["MASK_ALL"]
            )
            if looping:
                loop = [
                    bit
                    for nib in range(LOOP_NIBBLES)
                    for bit in state_lookup(f"loop.{nib}", LOOP_REGISTER, nib)
                ]
                any_lane = c.node(c.t_ge(sum(then_mask, Expr()), 1, 0, 8), "loop.any")
                enter = c.and_([ops["WHILE_BIT"], any_lane], "loop.enter")
                leave = c.and_([ops["WHILE_BIT"], one - any_lane], "loop.leave")
                loop_top, loop_depth, loop_frames = _stack(
                    c,
                    loop[:2],
                    [loop[4:12], loop[12:20]],
                    enter,
                    ops["ENDWHILE"],
                    active,
                    "loop",
                )
                next_loop = loop_depth + [Expr(), Expr()] + loop_frames
                mask_sources.extend(
                    [([enter], then_mask), ([ops["ENDWHILE"]], loop_top)]
                )
                keep_mask = keep_mask - enter - ops["ENDWHILE"]
            mask_sources.append(([keep_mask], active))
            next_mask = c.select_bits(mask_sources, 8, "branch.mask.next")
        else:
            next_mask = [
                ops["MASK_ALL"]
                + selected_bit(lane, ops["MASK_BIT"])
                + c.and_(
                    [prior, one - ops["MASK_BIT"] - ops["MASK_ALL"]],
                    f"mask.keep.{lane}",
                )
                for lane, prior in enumerate(active)
            ]

    next_index = []
    old_indices = []
    grants = []
    indexed_values = []
    if indexed:
        prior_addresses = []
        for lane in range(8):
            address = [
                bit
                for nib, width in enumerate((4, 3))
                for bit in state_lookup(
                    f"index.{lane}.{nib}", INDEX_REGISTER, 2 * lane + nib
                )[:width]
            ]
            old_indices.extend(address)
            queries = [
                c.and_(
                    [
                        ops["LOAD_INDEX"],
                        *[
                            value if word >> bit & 1 else one - value
                            for bit, value in enumerate(address)
                        ],
                    ],
                    f"gather.{lane}.{word}",
                )
                for word in range(INDEX_WORDS)
            ]
            for nib in range(4):
                slots = [
                    base // 16 + 4 * word + nib
                    for base in (INDEX_OPERAND, C_MEMORY)
                    for word in range(INDEX_WORDS // 2)
                ]
                indexed_values.extend(lookup(f"shared.{lane}.{nib}", 0, slots, queries))
            save_index = (
                c.and_([ops["SAVE_INDEX"], active[lane]], f"index.save.{lane}")
                if masked
                else ops["SAVE_INDEX"]
            )
            next_index.extend(
                c.select_bits(
                    [
                        ([save_index], old[16 * lane : 16 * lane + 7]),
                        ([one - save_index], address),
                    ],
                    7,
                    f"index.next.{lane}",
                )
            )
            # STORE_INDEX retains the old address, so it can use the same snapshot.
            addr = from_bits(address)
            earlier = [
                c.node(
                    c.t_eq(
                        addr - prior, 0, -127, 127, lits=[active[i]] if masked else ()
                    ),
                    f"scatter.same.{lane}.{i}",
                )
                for i, prior in enumerate(prior_addresses)
            ]
            grants.append(
                c.and_(
                    [
                        ops["STORE_INDEX"],
                        *([active[lane]] if masked else []),
                        *[one - equal for equal in earlier],
                    ],
                    f"scatter.grant.{lane}",
                )
            )
            prior_addresses.append(addr)

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
    sources = [([ops["LOAD_TILE"]], c_row)]
    if indexed:
        sources.append(([ops["LOAD_INDEX"]], indexed_values))
    sources.append(([one - sum((gate[0] for gate, _ in sources), Expr())], updated))
    result = c.select_bits(sources, 128, "register.next")
    if masked:
        result = [
            bit
            for lane, enabled in enumerate(active)
            for bit in c.select_bits(
                [
                    ([enabled], result[16 * lane : 16 * (lane + 1)]),
                    ([one - enabled], old[16 * lane : 16 * (lane + 1)]),
                ],
                16,
                f"masked.result.{lane}",
            )
        ]
    bits = result + [
        ops["STORE_TILE"],
        *row_bits,
        *next_index,
        *grants,
        *next_mask,
        *next_branch,
        *next_loop,
    ]
    if resident_delta:

        def padded_indices(indices: list[Expr]) -> list[Expr]:
            return [
                bit
                for start in range(0, len(indices), 7)
                for bit in (*indices[start : start + 7], Expr())
            ]

        prior_state = old + padded_indices(old_indices) + active + stack + loop
        updated_state = (
            result + padded_indices(next_index) + next_mask + next_branch + next_loop
        )
        for nib in range(len(WARP_TAGS)):
            start = 4 * nib
            difference = from_bits(updated_state[start : start + 4]) - from_bits(
                prior_state[start : start + 4]
            )
            equal = c.node(c.t_eq(difference, 0, -15, 15), f"state.equal.{nib}")
            bits.append(one - equal)
    c.snapshot_bits = len(bits)
    saved = []
    for chunk, start in enumerate(range(0, len(bits), CHUNK_BITS)):
        saved.extend(
            c.slot_head(
                f"stage.0.{chunk}",
                {0: tick},
                {
                    0: sum(
                        reads
                        + writes
                        + index_reads
                        + mask_reads
                        + branch_reads
                        + loop_reads
                        + plan_reads,
                        Expr(),
                    )
                },
                {
                    2 + bit: value
                    for bit, value in enumerate(bits[start : start + CHUNK_BITS])
                },
                tick,
            )
        )

    halted = ops["HALT"]
    pc_sources = [([halted], pc)]
    increment = one - halted
    if looping:
        pc_sources.extend([([leave], arg[4:8]), ([ops["ENDWHILE"]], arg[:4])])
        increment = increment - leave - ops["ENDWHILE"]
    pc_sources.append(([increment], _increment4(c, pc)))
    next_pc = c.select_bits(pc_sources, 4, "pc.next")
    execution = next_pc + [halted, ops["BARRIER"], warp_bit]

    if resident_delta:
        # State output requests are in packet order. Only changed fields
        # participate in the output code. The default code is token zero.
        dirty_reads = iter(saved[WARP_DIRTY_OFFSET:])

        def only_dirty(requests: list[Expr], name: str) -> list[Expr]:
            return [
                c.and_([read, next(dirty_reads)], f"{name}.dirty.{nib}")
                for nib, read in enumerate(requests)
            ]

        reads = only_dirty(reads, "register")
        index_reads = only_dirty(index_reads, "index")
        mask_reads = only_dirty(mask_reads, "mask")
        branch_reads = only_dirty(branch_reads, "branch")
        loop_reads = only_dirty(loop_reads, "loop")

    def state_tag(base: int, nib: int) -> int:
        if resident_state:
            return 16 * (WARP_TAGS.index(base) + nib)
        return base + 16 * nib

    state_reads = sum(
        reads + index_reads + mask_reads + branch_reads + loop_reads, Expr()
    )
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
            tile_write = [write, saved[128]]
            if masked:
                tile_write.append(saved[MASK_OFFSET + nib // 4])
            if bit < 4:
                value = saved[4 * nib + bit]
                terms += c.t_and([read, value], 2 * KAPPA)
                terms += c.t_and([*tile_write, value], 2 * KAPPA)
            else:
                if state_tag(REGISTER, nib) >> bit & 1:
                    terms += c.t_and([read], 2 * KAPPA)
                if 9 <= bit < 12:
                    terms += c.t_and([*tile_write, saved[129 + bit - 9]], 2 * KAPPA)
                elif (C_MEMORY + 16 * nib) >> bit & 1:
                    terms += c.t_and(tile_write, 2 * KAPPA)
        if indexed:
            for nib, read in enumerate(index_reads):
                lane, half = divmod(nib, 2)
                if bit < (4 if half == 0 else 3):
                    index_bit = saved[SNAPSHOT_BITS + 7 * lane + 4 * half + bit]
                    terms += c.t_and([read, index_bit], 2 * KAPPA)
                elif state_tag(INDEX_REGISTER, nib) >> bit & 1:
                    terms += c.t_and([read], 2 * KAPPA)
            for nib, write in enumerate(writes):
                lane, digit = divmod(nib, 4)
                index_start = SNAPSHOT_BITS + 7 * lane
                address = saved[index_start : index_start + 7]
                gate = [write, saved[SNAPSHOT_BITS + 56 + lane]]
                if bit < 4:
                    terms += c.t_and([*gate, saved[4 * nib + bit]], 2 * KAPPA)
                elif bit < 6:
                    if digit >> (bit - 4) & 1:
                        terms += c.t_and(gate, 2 * KAPPA)
                elif bit < 12:
                    terms += c.t_and([*gate, address[bit - 6]], 2 * KAPPA)
                else:
                    if INDEX_OPERAND >> bit & 1:
                        terms += c.t_and([*gate, one - address[6]], 2 * KAPPA)
                    if C_MEMORY >> bit & 1:
                        terms += c.t_and([*gate, address[6]], 2 * KAPPA)
        for nib, read in enumerate(mask_reads):
            if bit < 4:
                terms += c.t_and([read, saved[MASK_OFFSET + 4 * nib + bit]], 2 * KAPPA)
            elif state_tag(MASK_REGISTER, nib) >> bit & 1:
                terms += c.t_and([read], 2 * KAPPA)
        if divergent:
            # Split the extra packet across FFNs to fit the standalone model's
            # intermediate width, including compiler cleanup units.
            branch_terms = []
            for nib, read in enumerate(branch_reads):
                if bit < 4:
                    branch_terms += c.t_and(
                        [read, saved[BRANCH_OFFSET + 4 * nib + bit]], 2 * KAPPA
                    )
                elif state_tag(BRANCH_REGISTER, nib) >> bit & 1:
                    branch_terms += c.t_and([read], 2 * KAPPA)
            for nib, read in enumerate(loop_reads):
                if bit < 4:
                    branch_terms += c.t_and(
                        [read, saved[LOOP_OFFSET + 4 * nib + bit]], 2 * KAPPA
                    )
                elif state_tag(LOOP_REGISTER, nib) >> bit & 1:
                    branch_terms += c.t_and([read], 2 * KAPPA)
            if branch_terms:
                c.ffn(
                    branch_terms, f"exit.branch.{bit}", into=expr, min_level=floor + 1
                )
        for byte, read in enumerate(plan_reads):
            if bit < 8:
                if byte == PLAN_BYTES:
                    # Tile stores use the active mask. Indexed stores use
                    # the collision arbiter's grants.
                    terms += c.t_and(
                        [read, saved[128], saved[MASK_OFFSET + bit]], 2 * KAPPA
                    )
                    terms += c.t_and([read, saved[SNAPSHOT_BITS + 56 + bit]], 2 * KAPPA)
                else:
                    terms += c.t_and(
                        [read, saved[WARP_DIRTY_OFFSET + 8 * byte + bit]], 2 * KAPPA
                    )
            elif (PLAN_TAG + 256 * byte) >> bit & 1:
                terms += c.t_and([read], 2 * KAPPA)
        if resident_state:
            if WARP_BANKS[0] >> bit & 1:
                terms += c.t_and([state_reads, one - saved[129]], 2 * KAPPA)
            if WARP_BANKS[1] >> bit & 1:
                terms += c.t_and([state_reads, saved[129]], 2 * KAPPA)
        c.ffn(
            terms,
            f"exit.code.{bit}",
            into=expr,
            min_level=floor + (3 if divergent else 1),
        )
    return c


def build_standalone(
    *,
    resident: bool = False,
    indexed: bool = False,
    masked: bool = False,
    divergent: bool = False,
    looping: bool = False,
    resident_rom: bool = False,
    resident_state: bool = False,
    resident_delta: bool = False,
    resident_plan: bool = False,
    resident_io: bool = False,
    resident_grid: bool = False,
):
    from chipformer.q35.tiny import build as build_qwen

    circuit = build(
        resident=resident,
        indexed=indexed,
        masked=masked,
        divergent=divergent,
        looping=looping,
        resident_rom=resident_rom,
        resident_state=resident_state,
        resident_delta=resident_delta,
        resident_plan=resident_plan,
        resident_io=resident_io,
        resident_grid=resident_grid,
    )
    return build_qwen(
        circuit,
        list(range(VOCAB)),
        [next(iter(expr.terms)) for expr in circuit.code],
        share=True,
    ), circuit
