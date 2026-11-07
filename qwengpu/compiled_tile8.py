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
from qwengpu.memory_geometry import LEGACY_MEMORY, PAIR_DECODE, PAIR_WIRES, WIDE_MEMORY

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
ALT_OPERAND = 0xE000
SECOND_OUTPUT = 0xF000
RIGHT_OPERANDS = (0xB800, 0xC800)
BOTTOM_OPERANDS = (0x10000, 0x10800)
OUTPUT_BANKS = (C_MEMORY, SECOND_OUTPUT, C_MEMORY + 0x10000, SECOND_OUTPUT + 0x10000)
TILEGRID_VOCAB = 1 << 17
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
ADD32I_OP = "ADD32I"
LOAD_INDEX_HI = "LOAD_INDEX_HI"
STORE_INDEX_HI = "STORE_INDEX_HI"
SHARED32_OPS = (LOAD_INDEX_HI, STORE_INDEX_HI)
SHARED32_WIRE_OFFSET = 7
ADD_INDEX_LO = "ADD_INDEX_LO"
ADD_INDEX_HI = "ADD_INDEX_HI"
ADDMEM32_OPS = (ADD_INDEX_LO, ADD_INDEX_HI)
ADDMEM32_WIRE_OFFSET = 64
PAIR_MASK_COUNT = 64
RESIDENT_ADDMEM32_LAYOUT = "qwengpu-resident-addmem32-9b-v1"
RESIDENT_BRANCH32_LAYOUT = "qwengpu-resident-branch32-9b-v1"
RESIDENT_LOOP32_LAYOUT = "qwengpu-resident-loop32-9b-v1"
RESIDENT_TILEBANKS_LAYOUT = "qwengpu-resident-tilebanks-9b-v1"
RESIDENT_TILEPIPE_LAYOUT = "qwengpu-resident-tilepipe-9b-v1"
RESIDENT_TILEGRID_LAYOUT = "qwengpu-resident-tilegrid-9b-v1"
RESIDENT_GLOBALGRID_LAYOUT = "qwengpu-resident-globalgrid-9b-v1"
RESIDENT_COMPUTE_LAYOUT = "qwengpu-resident-compute-9b-v1"
ID_OPS = ("LANE_ID", "WORKGROUP_ID", "THREAD_ID", "GLOBAL_ID")
GLOBAL_OPS = ID_OPS + ("WARP_SUM", "SHL32I", "MASK_LANES")
COMPUTE_OPS = ("MASK_GLOBAL_LT", "MATRIX_MAC8")
GLOBAL_WIRE = 192
GLOBAL_GRID = 128
RESET_WARP = (2064, 2080)
WHILE_HI_OP = "WHILE_HI_BIT"
WHILE_HI_WIRE_OFFSET = 9
RESIDENT_SHARED32_LAYOUT = "qwengpu-resident-shared32-9b-v1"
ADD32I_ZERO_WIRE = 256 * OPS.index("MAC_K") + 8
READ_PLAN = ROM_BANK + 2
PLAN_TAG = 0xD000
GRID = 0xDE00
ROW_PAIRS = 4
PLAN_BYTES = 8
READ_HIGH = READ_PLAN + 16
HIGH_REGISTER = 0x9400
HIGH_NIBBLES = 32
WIDE_DIRTY_OFFSET = WARP_DIRTY_OFFSET + 4 * HIGH_NIBBLES
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
WIDE_WARP_TAGS = WARP_TAGS + tuple(
    HIGH_REGISTER + 16 * nib for nib in range(HIGH_NIBBLES)
)
NO_WRITE = 0
DONE = TICKET + 4
DEADLOCK = TICKET + 5
ADVANCE = 32
NEXT_BANK = TICKET + 64 + ADVANCE
NEXT_WORKGROUP = TICKET + 128 + ADVANCE
WORKGROUP_TICKETS = (NEXT_WORKGROUP, NEXT_WORKGROUP + 64, NEXT_WORKGROUP + 256)
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
        if op == "MASK_GLOBAL_LT":
            return OPS.index("BARRIER") * 256 + 32 + arg
        if op == "MATRIX_MAC8":
            return OPS.index("BARRIER") * 256 + 128 + arg
        if op in INDEX_OPS:
            return halt + 1 + INDEX_OPS.index(op)
        if op in SHARED32_OPS:
            return halt + SHARED32_WIRE_OFFSET + SHARED32_OPS.index(op)
        if op in ADDMEM32_OPS:
            if arg >= 128:
                return PAIR_WIRES[(ADDMEM32_OPS.index(op), arg)]
            return (
                halt
                + ADDMEM32_WIRE_OFFSET
                + PAIR_MASK_COUNT * ADDMEM32_OPS.index(op)
                + arg // 2
            )
        if op in GLOBAL_OPS:
            if op == "SHL32I":
                return halt + GLOBAL_WIRE + 5 + arg
            if op == "MASK_LANES":
                return halt + GLOBAL_WIRE + 13 + arg
            return halt + GLOBAL_WIRE + GLOBAL_OPS.index(op)
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
            if arg >= 16:
                # MAC_K 0 through 7 are live and 8 encodes ADD32I 0.
                # The first possible exit is PC 2, after WHILE and ENDWHILE.
                return (
                    OPS.index("MAC_K") * 256
                    + WHILE_HI_WIRE_OFFSET
                    + 16 * (links[address] - 2)
                    + arg
                    - 16
                )
            return OPS.index("LOAD_TILE") * 256 + 16 * links[address] + arg
        if op == "ENDWHILE":
            return OPS.index("BARRIER") * 256 + 1 + links[address]
        if op == ADD32I_OP:
            # STORE_TILE only uses immediate zero. Reserve MAC_K 8 for ADD32I 0.
            return OPS.index("STORE_TILE") * 256 + arg if arg else ADD32I_ZERO_WIRE
        return OPS.index(op) * 256 + arg

    return [
        ROM + address * STRIDE + wire(address, op, arg)
        for address, (op, arg) in enumerate(padded)
    ]


def rom_payload(
    token: int,
    *,
    indexed: bool,
    masked: bool,
    divergent: bool,
    looping: bool,
    resident_add32: bool = False,
    resident_shared32: bool = False,
    resident_addmem32: bool = False,
    resident_branch32: bool = False,
    resident_loop32: bool = False,
    resident_globalgrid: bool = False,
    resident_compute: bool = False,
    resident_memory512: bool = False,
) -> int:
    if not ROM <= token < ROM + 16 * STRIDE:
        return 0
    wire = (token - ROM) % STRIDE
    if resident_memory512 and wire in PAIR_DECODE:
        return PAIR_DECODE[wire]
    tag, arg = divmod(wire, 256)
    if resident_compute and OPS[tag] == "BARRIER":
        if 32 <= arg <= 96:
            return 29 * 256 + arg - 32
        if 128 <= arg < 192:
            return 30 * 256 + arg - 128
    if (
        resident_globalgrid
        and OPS[tag] == "HALT"
        and GLOBAL_WIRE <= arg < GLOBAL_WIRE + 22
    ):
        index = arg - GLOBAL_WIRE
        if index < 5:
            return (22 + index) * 256
        if index < 13:
            return 27 * 256 + index - 5
        return 28 * 256 + index - 13
    if resident_add32 and (
        (OPS[tag] == "STORE_TILE" and arg) or wire == ADD32I_ZERO_WIRE
    ):
        return (1 << ROM_PAYLOAD_BITS) + (0 if wire == ADD32I_ZERO_WIRE else arg)
    if (
        resident_shared32
        and OPS[tag] == "HALT"
        and SHARED32_WIRE_OFFSET <= arg < SHARED32_WIRE_OFFSET + len(SHARED32_OPS)
    ):
        return (1 << ROM_PAYLOAD_BITS) + 256 * (arg - SHARED32_WIRE_OFFSET + 1)
    if (
        resident_addmem32
        and OPS[tag] == "HALT"
        and ADDMEM32_WIRE_OFFSET
        <= arg
        < ADDMEM32_WIRE_OFFSET + PAIR_MASK_COUNT * len(ADDMEM32_OPS)
    ):
        op, pair_mask = divmod(arg - ADDMEM32_WIRE_OFFSET, PAIR_MASK_COUNT)
        return (
            (1 << ROM_PAYLOAD_BITS) + 256 * (1 + len(SHARED32_OPS) + op) + 2 * pair_mask
        )
    if (
        resident_loop32
        and OPS[tag] == "MAC_K"
        and WHILE_HI_WIRE_OFFSET <= arg < WHILE_HI_WIRE_OFFSET + 14 * 16
    ):
        return (
            (1 << ROM_PAYLOAD_BITS)
            + 256 * (1 + len(SHARED32_OPS) + len(ADDMEM32_OPS))
            + 32
            + arg
            - WHILE_HI_WIRE_OFFSET
        )
    if indexed and OPS[tag] == "HALT" and 1 <= arg <= len(INDEX_OPS):
        return (len(OPS) + arg - 1) * 256
    if masked and OPS[tag] == "HALT":
        if arg == 4:
            return (len(OPS) + len(INDEX_OPS)) * 256
        if 16 <= arg < 32:
            return (len(OPS) + len(INDEX_OPS) + 1) * 256 + arg - 16
    if divergent and OPS[tag] == "HALT":
        branch_tag = len(OPS) + len(INDEX_OPS) + len(MASK_OPS)
        if 32 <= arg < (64 if resident_branch32 else 48):
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
    resident_wide: bool = False,
    resident_add32: bool = False,
    resident_shared32: bool = False,
    resident_addmem32: bool = False,
    resident_branch32: bool = False,
    resident_loop32: bool = False,
    resident_tilebanks: bool = False,
    resident_tilepipe: bool = False,
    resident_tilegrid: bool = False,
    resident_globalgrid: bool = False,
    resident_compute: bool = False,
    resident_memory512: bool = False,
) -> Circ:
    resident_compute = resident_compute or resident_memory512
    resident_globalgrid = resident_globalgrid or resident_compute
    resident_tilegrid = resident_tilegrid or resident_globalgrid
    resident_tilepipe = resident_tilepipe or resident_tilegrid
    resident_tilebanks = resident_tilebanks or resident_tilepipe
    resident_loop32 = resident_loop32 or resident_tilebanks
    resident_branch32 = resident_branch32 or resident_loop32
    resident_addmem32 = resident_addmem32 or resident_branch32
    resident_shared32 = resident_shared32 or resident_addmem32
    resident_add32 = resident_add32 or resident_shared32
    resident_wide = resident_wide or resident_add32
    resident_grid = resident_grid or resident_wide
    warp_tags = WIDE_WARP_TAGS if resident_wide else WARP_TAGS
    dirty_offset = WIDE_DIRTY_OFFSET if resident_wide else WARP_DIRTY_OFFSET
    geometry = WIDE_MEMORY if resident_memory512 else LEGACY_MEMORY
    if resident_memory512:
        dirty_offset = geometry.dirty_offset
    plan_bytes = len(warp_tags) // 8
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
    c.resident_shared32 = resident_shared32
    c.resident_addmem32 = resident_addmem32
    c.resident_branch32 = resident_branch32
    c.resident_loop32 = resident_loop32
    c.resident_tilebanks = resident_tilebanks
    c.resident_tilepipe = resident_tilepipe
    c.resident_tilegrid = resident_tilegrid
    c.resident_globalgrid = resident_globalgrid
    c.resident_compute = resident_compute
    c.resident_memory512 = resident_memory512
    c.memory_geometry = geometry
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
    high_reads = (
        [match(f"high.read.{nib}", READ_HIGH + nib) for nib in range(HIGH_NIBBLES)]
        if resident_wide
        else []
    )
    plan_reads = (
        [
            match(f"state.plan.{byte}", READ_PLAN + byte)
            for byte in range(plan_bytes + int(resident_io))
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
        for bit in range(16 + int(resident_tilegrid) + int(resident_memory512))
    ]
    c.code = code
    c.keep = {next(iter(expr.terms)) for expr in code}
    for expr in code:
        c.ffn([(-1, one, expr), (1, one, -expr)], "code.clear", into=expr)

    raw = [
        c.token_feature(f"raw.{bit}", lambda t, bit=bit: (t >> bit) & 1, full=True)
        for bit in range(6 + int(resident_tilegrid) + int(resident_globalgrid))
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
                {**{2 + bit: value for bit, value in enumerate(raw[:6])}, 8: load},
                load,
            )
        )
    load_ticket = between(
        "ticket.load",
        TICKET,
        TICKET + (512 if resident_tilegrid else 128 if resident_tilepipe else 64),
    )
    last, tile0, tile1 = c.slot_head(
        "ticket",
        {0: load_ticket},
        {0: schedule + tick + sum(resets, Expr())},
        {2: raw[0], 3: raw[3], 4: raw[4]},
        load_ticket,
    )
    both_halted = c.and_([controls[0][4], controls[1][4]], "schedule.both_halted")
    switch = Expr()
    group_switch = Expr()
    if resident_grid:
        load_grid = between(
            "grid.load",
            GRID,
            GRID
            + ROW_PAIRS
            * (
                1
                << (
                    int(resident_tilebanks)
                    + int(resident_tilepipe)
                    + 3 * int(resident_tilegrid)
                    + int(resident_globalgrid)
                )
            ),
        )
        grid_queries = schedule + sum(resets, Expr())
        grid_value = {2: raw[0], 3: raw[1]}
        if resident_tilebanks:
            grid_queries += tick
            grid_value[4] = raw[2]
        if resident_tilepipe:
            grid_value[5] = raw[3]
        if resident_tilegrid:
            grid_value[6] = raw[4]
            grid_value[7] = raw[5]
            grid_value[8] = raw[6]
        if resident_globalgrid:
            grid_value[9] = raw[7]
        grid = c.slot_head(
            "grid", {0: load_grid}, {0: grid_queries}, grid_value, load_grid
        )
        difference = from_bits([tile0, tile1]) - from_bits(grid[:2])
        last_tile = c.node(c.t_eq(difference, 0, -3, 3), "schedule.last_tile")
        if resident_tilepipe:
            relayed_switch = match("operand.bank.advance", NEXT_BANK)
            group_tickets = (
                [
                    match(f"workgroup.advance.{index}", token)
                    for index, token in enumerate(WORKGROUP_TICKETS)
                ]
                if resident_tilegrid
                else []
            )
            relayed_group = sum(group_tickets, Expr())
            load_bank = load_grid + relayed_switch + relayed_group
            (bank_bit,) = c.slot_head(
                "operand.bank",
                {0: load_bank},
                {0: grid_queries},
                {
                    2: c.and_([load_grid, raw[2]], "operand.bank.initial")
                    + relayed_switch
                },
                load_bank,
            )
            switch = c.and_(
                [both_halted, last_tile, grid[3], one - bank_bit], "schedule.next_bank"
            )
            if resident_tilegrid:
                load_group = load_grid + relayed_group
                group_low, group_high = c.slot_head(
                    "workgroup",
                    {0: load_group},
                    {0: grid_queries},
                    {
                        2: group_tickets[0] + group_tickets[2],
                        3: group_tickets[1] + group_tickets[2],
                    },
                    load_group,
                )
                more_rows = c.and_([grid[6], one - group_high], "workgroup.more_rows")
                last_group = c.and_([group_low, one - more_rows], "workgroup.last")
                group_ready = (
                    c.and_(
                        [grid[4], bank_bit, one - last_group], "workgroup.tile_ready"
                    )
                    if resident_globalgrid
                    else None
                )
                if resident_globalgrid:
                    generic = grid[7]
                    count_difference = from_bits([group_low, group_high]) - from_bits(
                        grid[2:4]
                    )
                    last_global = c.node(
                        c.t_eq(count_difference, 0, -3, 3), "workgroup.global_last"
                    )
                    group_ready = c.and_(
                        [group_ready, one - generic], "workgroup.tile_mode"
                    ) + c.and_([generic, one - last_global], "workgroup.global_mode")
                    switch = c.and_([switch, one - generic], "schedule.tile_bank")
                group_switch = c.and_(
                    [both_halted, last_tile, group_ready]
                    if resident_globalgrid
                    else [both_halted, last_tile, grid[4], bank_bit, one - last_group],
                    "schedule.next_workgroup",
                )
                second_output = c.and_([group_low, grid[5]], "workgroup.right_output")
                bottom_output = c.and_([group_high, grid[6]], "workgroup.bottom_output")
        elif resident_tilebanks:
            bank_bit = grid[2]
    else:
        last_tile = c.and_([tile0, tile1], "schedule.last_tile")
    row_advance = c.and_([both_halted, one - last_tile], "schedule.advance")
    restart = switch + group_switch
    advance = row_advance + restart
    release = c.and_([controls[0][5], controls[1][5]], "schedule.release")
    ready = [one - ctrl[4] - ctrl[5] + advance + release for ctrl in controls]
    choose1 = c.and_([ready[1], one - ready[0]], "schedule.only1") + c.and_(
        [ready[1], ready[0], one - last], "schedule.prefer1"
    )
    stop = c.and_([one - ready[0], one - ready[1]], "schedule.stop")
    finished = c.and_([both_halted, last_tile], "schedule.finished") - restart
    carry = c.and_([tile0, row_advance], "schedule.tile.carry")
    new_tile = [
        tile0 + row_advance - 2 * carry,
        tile1 + carry - 2 * c.and_([tile1, carry], "schedule.tile.wrap"),
    ]
    if resident_tilepipe:
        choose1 = c.and_([choose1, one - restart], "schedule.bank.first_warp")
        new_tile = [
            c.and_([bit, one - restart], f"schedule.bank.reset_row.{i}")
            for i, bit in enumerate(new_tile)
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
    if resident_tilepipe:
        decision.append(switch)
    if resident_tilegrid:
        decision[-1] += c.and_([group_switch, group_low], "schedule.group.ticket.low")
        decision.extend(
            [
                group_switch,
                c.and_([group_switch, group_high], "schedule.group.ticket.high"),
            ]
        )

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
        + ((ADD32I_OP,) if resident_add32 else ())
        + (SHARED32_OPS if resident_shared32 else ())
        + (ADDMEM32_OPS if resident_addmem32 else ())
        + ((WHILE_HI_OP,) if resident_loop32 else ())
        + (GLOBAL_OPS if resident_globalgrid else ())
        + (COMPUTE_OPS if resident_compute else ())
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
                        resident_add32=resident_add32,
                        resident_shared32=resident_shared32,
                        resident_addmem32=resident_addmem32,
                        resident_branch32=resident_branch32,
                        resident_loop32=resident_loop32,
                        resident_globalgrid=resident_globalgrid,
                        resident_compute=resident_compute,
                        resident_memory512=resident_memory512,
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
        name: str,
        base: int,
        addresses: list[int],
        selects: list[Expr],
        *,
        reset: bool = False,
        reset_ones: bool = False,
    ) -> list[Expr]:
        keys = {}
        for slot, address in enumerate(addresses):
            start = base + address * 16
            if start not in input_keys:
                input_keys[start] = between(f"{name}.load.{slot}", start, start + 16)
            keys[slot] = input_keys[start]
        data = raw_nibble
        if reset:
            keys = {slot: key + warp_resets[slot] for slot, key in keys.items()}
            if reset_ones:
                data = {
                    bit: value + sum(warp_resets, Expr()) for bit, value in data.items()
                }
        return c.slot_head(
            name, keys, dict(enumerate(selects)), data, sum(keys.values(), Expr())
        )

    if resident_memory512:
        ram_word_bits = [
            c.token_feature(
                f"ram.word_bit.{bit}",
                lambda t, bit=bit: int(
                    geometry.base <= t < geometry.vocab
                    and (((t - geometry.base) // 64) >> bit) & 1
                ),
                full=True,
            )
            for bit in range(7)
        ]
        ram_nib = [
            c.token_feature(
                f"ram.nibble.{nib}",
                lambda t, nib=nib: int(
                    geometry.base <= t < geometry.vocab
                    and ((t - geometry.base) // 16) % 4 == nib
                ),
                full=True,
            )
            for nib in range(4)
        ]
        ram_bank_load = [
            c.token_feature(
                f"ram.load_bank.{bank}",
                lambda t, bank=bank: int(
                    geometry.base + bank * 8192 <= t < geometry.base + (bank + 1) * 8192
                ),
                full=True,
            )
            for bank in range(2)
        ]

        def ram_lookup(name, addresses, selects):
            bank, nib = addresses[0] // 512, addresses[0] % 4
            assert all(
                address // 512 == bank and address % 4 == nib for address in addresses
            )
            # Seven address-bit matches and the nibble/bank matches give
            # exactly one for a matching row and <= 0 otherwise. This keeps
            # the existing slot-head key contract without 128 one-hot rows.
            keys = {
                slot: sum(
                    (
                        ram_word_bits[bit]
                        if (address // 4) >> bit & 1
                        else one - ram_word_bits[bit]
                        for bit in range(7)
                    ),
                    Expr(),
                )
                + ram_nib[nib]
                + ram_bank_load[bank]
                - 8 * one
                for slot, address in enumerate(addresses)
            }
            return c.slot_head(
                name,
                keys,
                dict(enumerate(selects)),
                raw_nibble,
                ram_nib[nib] + ram_bank_load[bank] - one,
            )

    warp_resets = (
        [match(f"warp.reset.{i}", token) for i, token in enumerate(RESET_WARP)]
        if resident_globalgrid
        else []
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
            offset = warp_tags.index(base) + nib
            return lookup(
                name,
                0,
                [bank // 16 + offset for bank in WARP_BANKS],
                state_queries,
                reset=resident_globalgrid,
                reset_ones=base == MASK_REGISTER,
            )
        return lookup(name, base, [nib], state_queries)

    # Two A heads select the nibbles of one row element. Each B column
    # has its own pair of heads, addressed by the same K index.
    bank_gates = [one]
    if resident_tilebanks:
        bank_gates = [one - bank_bit, bank_bit]
    a_queries = (
        [
            c.and_([tick, gate, rows[row], k_hot[k]], f"A.query.{bank}.{row}.{k}")
            for bank, gate in enumerate(bank_gates)
            for row in range(8)
            for k in range(8)
        ]
        if resident_tilebanks
        else [
            c.and_([tick, rows[row], k_hot[k]], f"A.query.{row}.{k}")
            for row in range(8)
            for k in range(8)
        ]
    )
    b_queries = (
        [
            c.and_([tick, gate, k_hot[k]], f"B.query.{bank}.{k}")
            for bank, gate in enumerate(bank_gates)
            for k in range(8)
        ]
        if resident_tilebanks
        else [c.and_([tick, k_hot[k]], f"B.query.{k}") for k in range(8)]
    )
    bank_offsets = (
        [0, (ALT_OPERAND - operand_base) // 16] if resident_tilebanks else [0]
    )
    b_offsets = [offset + 128 for offset in bank_offsets]
    if resident_tilegrid:
        b_offsets.extend((base - operand_base) // 16 for base in RIGHT_OPERANDS)
        b_queries = [
            c.and_([query, gate], f"B.query.panel.{panel}.{slot}")
            for panel, gate in enumerate((one - second_output, second_output))
            for slot, query in enumerate(b_queries)
        ]
    row_queries = [
        c.and_([tick, hot], f"C.query.{row}") for row, hot in enumerate(rows)
    ]
    output_offsets = [0]
    if resident_tilegrid:
        output_offsets = [(base - C_MEMORY) // 16 for base in OUTPUT_BANKS]
        row_queries = [
            c.and_([query, horizontal, vertical], f"C.query.bank.{y}.{x}.{row}")
            for y, vertical in enumerate((one - bottom_output, bottom_output))
            for x, horizontal in enumerate((one - second_output, second_output))
            for row, query in enumerate(row_queries)
        ]
    top_a_queries = (
        [
            c.and_([query, one - bottom_output], f"A.top.{i}")
            for i, query in enumerate(a_queries)
        ]
        if resident_tilegrid
        else a_queries
    )
    a = [
        bit
        for nib in range(2)
        for bit in lookup(
            f"A.{nib}",
            operand_base,
            [offset + 2 * i + nib for offset in bank_offsets for i in range(64)],
            top_a_queries,
        )
    ]
    if resident_tilegrid:
        bottom_a_queries = [
            c.and_([query, bottom_output], f"A.bottom.{i}")
            for i, query in enumerate(a_queries)
        ]
        bottom_a = [
            bit
            for nib in range(2)
            for bit in lookup(
                f"A.bottom.{nib}",
                0,
                [
                    base // 16 + 2 * i + nib
                    for base in BOTTOM_OPERANDS
                    for i in range(64)
                ],
                bottom_a_queries,
            )
        ]
        a = [top + bottom for top, bottom in zip(a, bottom_a, strict=True)]
    b = [
        [
            bit
            for nib in range(2)
            for bit in lookup(
                f"B.{col}.{nib}",
                operand_base,
                [
                    offset + 2 * (8 * k + col) + nib
                    for offset in b_offsets
                    for k in range(8)
                ],
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
    old_high = [
        bit
        for nib in range(HIGH_NIBBLES if resident_wide else 0)
        for bit in state_lookup(f"high.{nib}", HIGH_REGISTER, nib)
    ]
    c_row = [
        bit
        for nib in range(32)
        for bit in lookup(
            f"C.{nib}",
            C_MEMORY,
            [offset + 32 * row + nib for offset in output_offsets for row in range(8)],
            row_queries,
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

        while_high = ops.get(WHILE_HI_OP, Expr())
        while_op = ops.get("WHILE_BIT", Expr()) + while_high
        high_predicate = (
            c.and_([ops["IF_BIT"], arg[4]], "branch.high")
            if resident_branch32
            else Expr()
        ) + while_high

        def selected_bit(lane: int, gate: Expr) -> Expr:
            start = 16 * lane
            terms = [
                term
                for selector, value in zip(
                    selectors, old[start : start + 16], strict=True
                )
                for term in c.t_and([selector, value, gate - high_predicate])
            ]
            if resident_branch32:
                terms += [
                    term
                    for selector, value in zip(
                        selectors, old_high[start : start + 16], strict=True
                    )
                    for term in c.t_and([selector, value, high_predicate])
                ]
            return c.node(terms, f"mask.selected.{lane}")

        if divergent:
            predicate_op = ops["MASK_BIT"] + ops["IF_BIT"] + while_op
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
            if resident_globalgrid:
                mask_sources.append(
                    (
                        [ops["MASK_LANES"]],
                        [
                            c.node(
                                c.t_ge(
                                    from_bits(arg[:4]), lane + 1, 0, 15, lits=[enabled]
                                ),
                                f"mask.first.{lane}",
                            )
                            for lane, enabled in enumerate(active)
                        ],
                    )
                )
                keep_mask -= ops["MASK_LANES"]
            if resident_compute:
                mask_sources.append(
                    (
                        [ops["MASK_GLOBAL_LT"]],
                        [
                            c.node(
                                c.t_lt(
                                    16 * from_bits([group_low, group_high])
                                    + 8 * warp_bit
                                    + lane
                                    - from_bits(arg[:7]),
                                    0,
                                    -127,
                                    63,
                                    lits=[enabled],
                                ),
                                f"mask.global.{lane}",
                            )
                            for lane, enabled in enumerate(active)
                        ],
                    )
                )
                keep_mask -= ops["MASK_GLOBAL_LT"]
            if looping:
                loop = [
                    bit
                    for nib in range(LOOP_NIBBLES)
                    for bit in state_lookup(f"loop.{nib}", LOOP_REGISTER, nib)
                ]
                any_lane = c.node(c.t_ge(sum(then_mask, Expr()), 1, 0, 8), "loop.any")
                enter = c.and_([while_op, any_lane], "loop.enter")
                leave = c.and_([while_op, one - any_lane], "loop.leave")
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
    next_index_high = []
    prior_index_high = []
    grants = []
    indexed_values = []
    load_high = ops.get(LOAD_INDEX_HI, Expr())
    store_high = ops.get(STORE_INDEX_HI, Expr())
    add_low = ops.get(ADD_INDEX_LO, Expr())
    add_high = ops.get(ADD_INDEX_HI, Expr())
    if indexed:
        load_index = ops["LOAD_INDEX"] + load_high + add_low + add_high
        read_high = load_high + add_high
        pair_mask = (
            [
                c.and_([add_low + add_high, bit], f"gather.mask.{i}")
                for i, bit in enumerate(arg[1 : geometry.index_bits], 1)
            ]
            if resident_addmem32
            else []
        )
        store_index = ops["STORE_INDEX"] + store_high
        prior_addresses = []
        for lane in range(8):
            address = [
                bit
                for nib, width in enumerate((4, 4 if resident_memory512 else 3))
                for bit in state_lookup(
                    f"index.{lane}.{nib}", INDEX_REGISTER, 2 * lane + nib
                )[:width]
            ]
            if resident_memory512:
                address[7] = c.and_([address[7], generic], f"index.global_high.{lane}")
                prior_index_high.append(address[7])
            old_indices.extend(address[:7])
            # Both limbs share the saved base. The high limb toggles its low bit.
            gather_address = address
            if resident_shared32:
                paired_bit = c.node(
                    c.t_and([address[0], one - read_high])
                    + c.t_and([one - address[0], read_high]),
                    f"gather.pair.{lane}",
                )
                gather_address = [paired_bit, *address[1:]]
            if resident_addmem32:
                gather_address[1:] = [
                    c.node(
                        c.t_and([addr, one - mask]) + c.t_and([one - addr, mask]),
                        f"gather.address.{lane}.{bit}",
                    )
                    for bit, (addr, mask) in enumerate(
                        zip(address[1:], pair_mask, strict=True), 1
                    )
                ]
            queries = [
                c.and_(
                    [
                        load_index,
                        *[
                            value if word >> bit & 1 else one - value
                            for bit, value in enumerate(gather_address[:7])
                        ],
                    ],
                    f"gather.{lane}.{word}",
                )
                for word in range(128)
            ]
            if resident_memory512:
                ram_bank = [
                    c.and_([generic, one - gather_address[7]], f"gather.bank.{lane}.0"),
                    c.and_([generic, gather_address[7]], f"gather.bank.{lane}.1"),
                ]
            for nib in range(4):
                slots = [
                    base // 16 + 4 * word + nib
                    for base in (INDEX_OPERAND, C_MEMORY)
                    for word in range(INDEX_WORDS // 2)
                ]
                value = lookup(f"shared.{lane}.{nib}", 0, slots, queries)
                if resident_memory512:
                    sources = [([one - generic], value)]
                    for bank in range(2):
                        start = bank * 128
                        other = ram_lookup(
                            f"ram.{lane}.{nib}.{bank}",
                            [4 * word + nib for word in range(start, start + 128)],
                            queries,
                        )
                        sources.append(([ram_bank[bank]], other))
                    value = c.select_bits(sources, 4, f"gather.value.{lane}.{nib}")
                indexed_values.extend(value)
            save_index = (
                c.and_([ops["SAVE_INDEX"], active[lane]], f"index.save.{lane}")
                if masked
                else ops["SAVE_INDEX"]
            )
            next_index.extend(
                c.select_bits(
                    [
                        ([save_index], old[16 * lane : 16 * lane + 7]),
                        ([one - save_index], address[:7]),
                    ],
                    7,
                    f"index.next.{lane}",
                )
            )
            if resident_memory512:
                high = c.and_(
                    [old[16 * lane + 7], generic], f"index.source_high.{lane}"
                )
                next_index_high.extend(
                    c.select_bits(
                        [([save_index], [high]), ([one - save_index], [address[7]])],
                        1,
                        f"index.next_high.{lane}",
                    )
                )
            # STORE_INDEX retains the old address, so it can use the same snapshot.
            addr = from_bits(address)
            earlier = [
                c.node(
                    c.t_eq(
                        addr - prior,
                        0,
                        -(geometry.words - 1),
                        geometry.words - 1,
                        lits=[active[i]] if masked else (),
                    ),
                    f"scatter.same.{lane}.{i}",
                )
                for i, prior in enumerate(prior_addresses)
            ]
            grants.append(
                c.and_(
                    [
                        store_index,
                        *([active[lane]] if masked else []),
                        *[one - equal for equal in earlier],
                    ],
                    f"scatter.grant.{lane}",
                )
            )
            prior_addresses.append(addr)

    if resident_globalgrid:
        registers32 = [
            old[16 * lane : 16 * (lane + 1)] + old_high[16 * lane : 16 * (lane + 1)]
            for lane in range(8)
        ]
        warp_sum = _word_from_columns(
            c,
            [
                sum(
                    (
                        c.and_([active[lane], word[bit]], f"sum.input.{lane}.{bit}")
                        for lane, word in enumerate(registers32)
                    ),
                    Expr(),
                )
                for bit in range(32)
            ],
            [8] * 32,
            "warp.sum",
        )
        shifts = c.onehot_of_bits(arg[:3], "shift.amount", 8)
        group_ops = sum((ops[op] for op in GLOBAL_OPS[:-1]), Expr())
        special_low = []
        special_high = []
        for lane, word in enumerate(registers32):
            lane_bits = [(lane >> bit) & 1 for bit in range(3)]
            id_bits = (
                lane_bits,
                [group_low, group_high],
                lane_bits + [warp_bit],
                lane_bits + [warp_bit, group_low, group_high],
            )
            shifted = c.select_bits(
                [
                    ([shift], [Expr()] * n + word[: 32 - n])
                    for n, shift in enumerate(shifts)
                ],
                32,
                f"shift.lane.{lane}",
            )
            special = c.select_bits(
                [
                    ([ops[op]], bits + [Expr()] * (32 - len(bits)))
                    for op, bits in zip(ID_OPS, id_bits, strict=True)
                ]
                + [([ops["SHL32I"]], shifted), ([ops["WARP_SUM"]], warp_sum)],
                32,
                f"global.result.{lane}",
            )
            special_low.extend(special[:16])
            special_high.extend(special[16:])
    if resident_compute:
        from qwengpu.compiled_matrix import operands as matrix_operands

        def matrix_lookup(name, base, addresses, selects):
            if resident_memory512:
                return ram_lookup(name, addresses, selects)
            return lookup(name, base, addresses, selects)

        matrix_a, matrix_b = matrix_operands(
            c,
            ops["MATRIX_MAC8"],
            arg,
            active,
            [group_low, group_high],
            warp_bit,
            matrix_lookup,
        )
    updated = []
    next_high = [] if resident_add32 else old_high
    add_immediate = ops["ADD16I"] + ops.get(ADD32I_OP, Expr())
    for lane in range(8):
        columns = []
        limits = []
        gathered = indexed_values[16 * lane : 16 * (lane + 1)]
        for bit in range(16):
            pairs = [(i, bit - i) for i in range(8) if 0 <= bit - i < 8]
            terms = [
                term
                for i, j in pairs
                for term in c.t_and([ops["MAC_K"], a[i], b[lane][j]])
            ]
            if resident_compute:
                terms += [
                    term
                    for i, j in pairs
                    for term in c.t_and(
                        [ops["MATRIX_MAC8"], matrix_a[lane][i], matrix_b[lane][j]]
                    )
                ]
            if bit < 8:
                terms += c.t_and([add_immediate, arg[bit]])
            if resident_addmem32:
                terms += c.t_and([add_low, gathered[bit]])
            columns.append(
                old[16 * lane + bit] + c.node(terms, f"lane.{lane}.column.{bit}")
            )
            limits.append(
                max(1 if resident_addmem32 else 0, len(pairs) + int(bit < 8)) + 1
            )
        if resident_add32:
            upper = old_high[16 * lane : 16 * (lane + 1)]
            upper_columns = upper
            if resident_addmem32:
                upper_columns = [
                    bit + c.and_([add_high, rhs], f"lane.{lane}.add_high.{i}")
                    for i, (bit, rhs) in enumerate(zip(upper, gathered, strict=True))
                ]
            word = _word_from_columns(
                c,
                columns + upper_columns,
                limits + [1 + int(resident_addmem32)] * 16,
                f"lane.{lane}.word",
            )
            updated.extend(word[:16])
            enabled = c.and_(
                [
                    ops[ADD32I_OP]
                    + add_low
                    + add_high
                    + ops.get("MATRIX_MAC8", Expr()),
                    active[lane],
                ],
                f"wide.active.{lane}",
            )
            high_sources = [([enabled], word[16:])]
            preserve = one - enabled
            if resident_globalgrid:
                special_enabled = c.and_(
                    [group_ops, active[lane]], f"global.active.{lane}"
                )
                high_sources.append(
                    ([special_enabled], special_high[16 * lane : 16 * (lane + 1)])
                )
                preserve -= special_enabled
            if resident_shared32:
                gather_high = c.and_([load_high, active[lane]], f"wide.gather.{lane}")
                high_sources.append(([gather_high], gathered))
                preserve -= gather_high
            high_sources.append(([preserve], upper))
            next_high.extend(
                c.select_bits(
                    high_sources,
                    16,
                    f"wide.next.{lane}",
                )
            )
        else:
            updated.extend(_word_from_columns(c, columns, limits, f"lane.{lane}.word"))
    sources = [([ops["LOAD_TILE"]], c_row)]
    if indexed:
        sources.append(([ops["LOAD_INDEX"]], indexed_values))
    if resident_globalgrid:
        sources.append(([group_ops], special_low))
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
        *next_high,
    ]
    if resident_memory512:
        bits.extend([*next_index_high, generic, Expr(), Expr(), Expr()])
    if resident_delta:

        def padded_indices(indices: list[Expr], high: list[Expr]) -> list[Expr]:
            return [
                bit
                for start in range(0, len(indices), 7)
                for bit in (
                    *indices[start : start + 7],
                    high[start // 7] if resident_memory512 else Expr(),
                )
            ]

        prior_state = (
            old + padded_indices(old_indices, prior_index_high) + active + stack + loop
        )
        updated_state = (
            result
            + padded_indices(next_index, next_index_high)
            + next_mask
            + next_branch
            + next_loop
        )
        if resident_add32:
            prior_state += old_high
            updated_state += next_high
        for nib in range(len(prior_state) // 4):
            start = 4 * nib
            difference = from_bits(updated_state[start : start + 4]) - from_bits(
                prior_state[start : start + 4]
            )
            equal = c.node(c.t_eq(difference, 0, -15, 15), f"state.equal.{nib}")
            bits.append(one - equal)
        # This layout preserves the high limb until it has a writing opcode.
        if resident_wide and not resident_add32:
            bits.extend(Expr() for _ in range(HIGH_NIBBLES))
    if resident_shared32:
        bits.append(store_high)
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
                        + high_reads
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

    if resident_tilegrid:
        store_second_output, store_bottom_output = c.slot_head(
            "workgroup.output",
            {0: tick},
            {0: sum(writes, Expr())},
            {2: second_output, 3: bottom_output},
            tick,
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
        dirty_reads = iter(saved[dirty_offset:])

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
        high_reads = only_dirty(high_reads, "high")

    def state_tag(base: int, nib: int) -> int:
        if resident_state:
            return 16 * (warp_tags.index(base) + nib)
        return base + 16 * nib

    state_reads = sum(
        reads + index_reads + mask_reads + branch_reads + loop_reads + high_reads,
        Expr(),
    )
    floor = max(node.level for node in c.nodes)
    for bit, expr in enumerate(code):
        terms = c.t_const(-KAPPA)
        control_terms = []
        value = execution[bit] if bit < len(execution) else (CONTROL >> bit) & 1
        control_terms += c.t_and([tick, value], 2 * KAPPA)
        value = decision[bit] if bit < len(decision) else (TICKET >> bit) & 1
        control_terms += c.t_and([schedule, value], 2 * KAPPA)
        for warp, reset in enumerate(resets):
            if bit < 6:
                gates = [reset, controls[warp][bit], one - advance]
                if bit == 5:
                    gates.append(one - release)
                control_terms += c.t_and(gates, 2 * KAPPA)
            elif control_token(warp) >> bit & 1:
                control_terms += c.t_and([reset], 2 * KAPPA)
        if not resident_wide:
            terms += control_terms
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
                elif bit == 15 and resident_tilegrid:
                    terms += c.t_and([*tile_write, store_second_output], 2 * KAPPA)
                elif bit == 16 and resident_tilegrid:
                    terms += c.t_and([*tile_write, store_bottom_output], 2 * KAPPA)
                elif (C_MEMORY + 16 * nib) >> bit & 1:
                    terms += c.t_and(tile_write, 2 * KAPPA)
        if indexed:
            for nib, read in enumerate(index_reads):
                lane, half = divmod(nib, 2)
                if bit < (4 if half == 0 or resident_memory512 else 3):
                    index_bit = (
                        saved[geometry.extra_index_offset + lane]
                        if resident_memory512 and half == 1 and bit == 3
                        else saved[SNAPSHOT_BITS + 7 * lane + 4 * half + bit]
                    )
                    terms += c.t_and([read, index_bit], 2 * KAPPA)
                elif state_tag(INDEX_REGISTER, nib) >> bit & 1:
                    terms += c.t_and([read], 2 * KAPPA)
            scatter_terms = [] if resident_shared32 else terms
            for nib, write in enumerate(writes):
                lane, digit = divmod(nib, 4)
                index_start = SNAPSHOT_BITS + 7 * lane
                address = saved[index_start : index_start + 7]
                if resident_memory512:
                    address = [*address, saved[geometry.extra_index_offset + lane]]
                gate = [write, saved[SNAPSHOT_BITS + 56 + lane]]
                if bit < 4:
                    low_value = saved[4 * nib + bit]
                    if resident_shared32:
                        high = saved[-1]
                        high_value = saved[WARP_DIRTY_OFFSET + 4 * nib + bit]
                        scatter_terms += c.t_and(
                            [*gate, one - high, low_value], 2 * KAPPA
                        )
                        scatter_terms += c.t_and([*gate, high, high_value], 2 * KAPPA)
                    else:
                        scatter_terms += c.t_and([*gate, low_value], 2 * KAPPA)
                elif bit < 6:
                    if digit >> (bit - 4) & 1:
                        scatter_terms += c.t_and(gate, 2 * KAPPA)
                elif bit == 6 and resident_shared32:
                    high = saved[-1]
                    scatter_terms += c.t_and([*gate, one - high, address[0]], 2 * KAPPA)
                    scatter_terms += c.t_and([*gate, high, one - address[0]], 2 * KAPPA)
                elif bit < 12:
                    scatter_terms += c.t_and([*gate, address[bit - 6]], 2 * KAPPA)
                else:
                    old_gate = (
                        [*gate, one - saved[geometry.mode_offset]]
                        if resident_memory512
                        else gate
                    )
                    if INDEX_OPERAND >> bit & 1:
                        scatter_terms += c.t_and(
                            [*old_gate, one - address[6]], 2 * KAPPA
                        )
                    if C_MEMORY >> bit & 1:
                        scatter_terms += c.t_and([*old_gate, address[6]], 2 * KAPPA)
                    if resident_memory512:
                        new_gate = [*gate, saved[geometry.mode_offset]]
                        if bit in (12, 13):
                            scatter_terms += c.t_and(
                                [*new_gate, address[bit - 6]], 2 * KAPPA
                            )
                        elif geometry.base >> bit & 1:
                            scatter_terms += c.t_and(new_gate, 2 * KAPPA)
            if resident_shared32 and scatter_terms:
                c.ffn(
                    scatter_terms, f"exit.scatter.{bit}", into=expr, min_level=floor + 1
                )
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
            for nib, read in enumerate(high_reads):
                if bit < 4:
                    branch_terms += c.t_and(
                        [read, saved[WARP_DIRTY_OFFSET + 4 * nib + bit]], 2 * KAPPA
                    )
                elif state_tag(HIGH_REGISTER, nib) >> bit & 1:
                    branch_terms += c.t_and([read], 2 * KAPPA)
            if branch_terms:
                c.ffn(
                    branch_terms, f"exit.branch.{bit}", into=expr, min_level=floor + 1
                )
        plan_terms = control_terms if resident_wide else []
        for byte, read in enumerate(plan_reads):
            if bit < 8:
                if byte == plan_bytes:
                    # Tile stores use the active mask. Indexed stores use
                    # the collision arbiter's grants.
                    plan_terms += c.t_and(
                        [read, saved[128], saved[MASK_OFFSET + bit]], 2 * KAPPA
                    )
                    plan_terms += c.t_and(
                        [read, saved[SNAPSHOT_BITS + 56 + bit]], 2 * KAPPA
                    )
                else:
                    plan_terms += c.t_and(
                        [read, saved[dirty_offset + 8 * byte + bit]], 2 * KAPPA
                    )
            elif (PLAN_TAG + 256 * byte) >> bit & 1:
                plan_terms += c.t_and([read], 2 * KAPPA)
        if resident_wide and plan_terms:
            c.ffn(
                plan_terms, f"exit.control_plan.{bit}", into=expr, min_level=floor + 3
            )
        else:
            terms += plan_terms
        if resident_state:
            if WARP_BANKS[0] >> bit & 1:
                terms += c.t_and([state_reads, one - saved[129]], 2 * KAPPA)
            if WARP_BANKS[1] >> bit & 1:
                terms += c.t_and([state_reads, saved[129]], 2 * KAPPA)
        c.ffn(
            terms,
            f"exit.code.{bit}",
            into=expr,
            min_level=floor + (5 if resident_wide else 3 if divergent else 1),
        )
    if resident_tilegrid:
        # The compiler clears live chip slots at the last exit layer. Give
        # cleanup its own layer to fit the standalone 3,584-unit MLP.
        c.ffn([], "exit.cleanup", into=code[0], min_level=max(c.units))
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
    resident_wide: bool = False,
    resident_add32: bool = False,
    resident_shared32: bool = False,
    resident_addmem32: bool = False,
    resident_branch32: bool = False,
    resident_loop32: bool = False,
    resident_tilebanks: bool = False,
    resident_tilepipe: bool = False,
    resident_tilegrid: bool = False,
    resident_globalgrid: bool = False,
    resident_compute: bool = False,
    resident_memory512: bool = False,
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
        resident_wide=resident_wide,
        resident_add32=resident_add32,
        resident_shared32=resident_shared32,
        resident_addmem32=resident_addmem32,
        resident_branch32=resident_branch32,
        resident_loop32=resident_loop32,
        resident_tilebanks=resident_tilebanks,
        resident_tilepipe=resident_tilepipe,
        resident_tilegrid=resident_tilegrid,
        resident_globalgrid=resident_globalgrid,
        resident_compute=resident_compute,
        resident_memory512=resident_memory512,
    )
    return build_qwen(
        circuit,
        list(
            range(
                WIDE_MEMORY.vocab
                if resident_memory512
                else TILEGRID_VOCAB
                if resident_tilegrid
                or resident_globalgrid
                or resident_compute
                or resident_memory512
                else VOCAB
            )
        ),
        [next(iter(expr.terms)) for expr in circuit.code],
        share=True,
    ), circuit
