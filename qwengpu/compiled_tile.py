"""A model scheduler and a two-warp 4x4 integer tile engine."""

from __future__ import annotations

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target
from chipformer.q35.prims import Circ

from qwengpu.compiled_rom import CODE_AMP, KAPPA, _increment4

VOCAB = 65536
CONTROL = 1024
TICKET = 1536
DONE = TICKET + 4
DEADLOCK = TICKET + 5
SCHEDULE, READY0, READY1, TICK = range(2048, 2052)
EXPORT = 2112
ROM = 4096
PACKET = 32768
REGISTER_BYTES = 16
SHARED_BYTES = 64
PACKET_BYTES = REGISTER_BYTES + SHARED_BYTES
CHUNK_BITS = 120
OPS = ("MAC_K", "ADD16I", "STORE_TILE", "BARRIER", "HALT")
STRIDE = 256 * len(OPS)


def control_token(warp: int, pc: int = 0, status: int = 0) -> int:
    return CONTROL + 64 * warp + pc + 16 * status


def program_tokens(program: tuple[tuple[str, int], ...]) -> list[int]:
    padded = program + (("HALT", 0),) * (16 - len(program))
    return [
        ROM + address * STRIDE + 256 * OPS.index(op) + arg
        for address, (op, arg) in enumerate(padded)
    ]


def packet_tokens(data: list[int]) -> list[int]:
    return [PACKET + 256 * address + value for address, value in enumerate(data)]


def _word_from_columns(
    c: Circ, columns: list[Expr], limits: list[int], name: str
) -> list[Expr]:
    """Decode four nibbles with exact bounded threshold sums."""
    result = []
    carry = Expr()
    carry_limit = 0
    for start in range(0, 16, 4):
        prefix = carry
        upper = carry_limit
        for bit in range(4):
            weight = 1 << bit
            prefix += columns[start + bit] * weight
            upper += limits[start + bit] * weight
            terms = []
            for threshold in range(weight, upper + 1, weight):
                sign = 1 if (threshold // weight) & 1 else -1
                terms += c.t_ge(prefix, threshold, 0, upper, coef=sign)
            result.append(c.node(terms, f"{name}.bit.{start + bit}"))
        if start < 12:
            carry = c.node(
                [
                    term
                    for threshold in range(16, upper + 1, 16)
                    for term in c.t_ge(prefix, threshold, 0, upper)
                ],
                f"{name}.carry.{start}",
            )
            carry_limit = upper // 16
    return result


def build(target: Target | None = None) -> Circ:
    c = Circ(target or Target())
    one = c.one
    feature = c.token_feature

    def request(name, token):
        return feature(name, lambda t: int(t == token), full=True)

    def load_range(name, start, size):
        return feature(name, lambda t: int(start <= t < start + size), full=True)

    tick = request("tick", TICK)
    schedule = request("schedule", SCHEDULE)
    resets = [request("ready.0", READY0), request("ready.1", READY1)]
    exports = [
        request(f"export.{index}", EXPORT + index) for index in range(PACKET_BYTES)
    ]
    code = [
        feature(
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

    raw_byte = [
        feature(f"byte.{bit}", lambda t, bit=bit: (t >> bit) & 1, full=True)
        for bit in range(8)
    ]
    control_query = tick + schedule + sum(resets, Expr())
    controls = []
    for warp in range(2):
        load = load_range(f"control.load.{warp}", CONTROL + 64 * warp, 64)
        controls.append(
            c.slot_head(
                f"control.{warp}",
                {0: load},
                {0: control_query},
                {**{2 + bit: raw_byte[bit] for bit in range(6)}, 8: load},
                load,
            )
        )
    last_ticket = load_range("ticket.load", TICKET, 4)
    (last_warp,) = c.slot_head(
        "last.warp", {0: last_ticket}, {0: schedule}, {2: raw_byte[0]}, last_ticket
    )
    release = c.and_([controls[0][5], controls[1][5]], "barrier.release")
    ready = [one - state[4] - state[5] + release for state in controls]
    choose1 = c.and_([ready[1], one - ready[0]], "schedule.only1") + c.and_(
        [ready[1], ready[0], one - last_warp], "schedule.prefer1"
    )
    terminal = c.and_([one - ready[0], one - ready[1]], "schedule.stop")
    both_halted = c.and_([controls[0][4], controls[1][4]], "schedule.halted")
    ticket_bits = [choose1 + terminal - both_halted, release, terminal]

    # Only one warp control packet is present during an execution call.
    pc = [
        c.node(
            c.t_and([tick, controls[0][bit]]) + c.t_and([tick, controls[1][bit]]),
            f"exec.pc.{bit}",
        )
        for bit in range(4)
    ]
    warp = controls[1][6]
    keys = {
        address: load_range(f"rom.load.{address}", ROM + address * STRIDE, STRIDE)
        for address in range(16)
    }
    is_rom = load_range("rom.load", ROM, 16 * STRIDE)
    raw_rom = [
        feature(
            f"rom.bit.{bit}",
            lambda t, bit=bit: (
                (((t - ROM) % STRIDE) >> bit) & 1 if ROM <= t < ROM + 16 * STRIDE else 0
            ),
            full=True,
        )
        for bit in range(11)
    ]
    pc_hot = c.onehot_of_bits(pc, "pc", 16)
    queries = {
        address: c.and_([tick, active], f"rom.query.{address}")
        for address, active in enumerate(pc_hot)
    }
    payload = c.slot_head(
        "rom",
        keys,
        queries,
        {2 + bit: expr for bit, expr in enumerate(raw_rom)},
        is_rom,
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
    k_hot = c.onehot_of_bits(arg[:2], "tile.k", 4)

    data = []
    for index in range(PACKET_BYTES):
        load = load_range(f"packet.load.{index}", PACKET + index * 256, 256)
        data.append(
            c.slot_head(
                f"packet.{index}",
                {0: load},
                {0: tick},
                {2 + bit: expr for bit, expr in enumerate(raw_byte)},
                load,
            )
        )
    memory = data[REGISTER_BYTES:]
    left = [
        c.select_bits(
            [
                ([k_hot[k], warp if w else one - warp], memory[8 * w + row * 4 + k])
                for w in range(2)
                for k in range(4)
            ],
            8,
            f"tile.A.{row}",
        )
        for row in range(2)
    ]
    right = [
        c.select_bits(
            [([k_hot[k]], memory[16 + k * 4 + col]) for k in range(4)],
            8,
            f"tile.B.{col}",
        )
        for col in range(4)
    ]
    accumulators = [data[2 * lane] + data[2 * lane + 1] for lane in range(8)]
    updated = []
    for lane, old in enumerate(accumulators):
        a, b = left[lane // 4], right[lane % 4]
        columns = []
        limits = []
        for bit in range(16):
            pairs = [(i, bit - i) for i in range(8) if 0 <= bit - i < 8]
            terms = [
                term for i, j in pairs for term in c.t_and([ops["MAC_K"], a[i], b[j]])
            ]
            if bit < 8:
                terms += c.t_and([ops["ADD16I"], arg[bit]])
            columns.append(old[bit] + c.node(terms, f"lane.{lane}.column.{bit}"))
            limits.append(len(pairs) + 1 + int(bit < 8))
        updated.extend(_word_from_columns(c, columns, limits, f"lane.{lane}.acc"))

    shared = [bit for byte in memory[:32] for bit in byte]
    for owner in range(2):
        store = c.and_(
            [ops["STORE_TILE"], warp if owner else one - warp], f"tile.store.{owner}"
        )
        for lane, old in enumerate(accumulators):
            index = 32 + 16 * owner + 2 * lane
            shared.extend(
                c.select_bits(
                    [
                        ([store], old),
                        ([one - store], memory[index] + memory[index + 1]),
                    ],
                    16,
                    f"tile.C.{owner}.{lane}",
                )
            )
    bits = updated + shared
    c.snapshot_bits = len(bits)
    export_query = sum(exports, Expr())
    saved = []
    for chunk, start in enumerate(range(0, len(bits), CHUNK_BITS)):
        saved.extend(
            c.slot_head(
                f"stage.0.{chunk}",
                {0: tick},
                {0: export_query},
                {
                    2 + bit: expr
                    for bit, expr in enumerate(bits[start : start + CHUNK_BITS])
                },
                tick,
            )
        )

    halted = ops["HALT"]
    next_pc = c.select_bits(
        [([halted], pc), ([one - halted], _increment4(c, pc))], 4, "exec.next_pc"
    )
    execution = next_pc + [halted, ops["BARRIER"], warp]
    floor = max(node.level for node in c.nodes)
    for bit, expr in enumerate(code):
        terms = c.t_const(-KAPPA)
        value = execution[bit] if bit < len(execution) else (CONTROL >> bit) & 1
        terms += c.t_and([tick, value], 2 * KAPPA)
        decision = ticket_bits[bit] if bit < len(ticket_bits) else (TICKET >> bit) & 1
        terms += c.t_and([schedule, decision], 2 * KAPPA)
        for w, reset in enumerate(resets):
            if bit == 5:
                terms += c.t_and([reset, controls[w][5], one - release], 2 * KAPPA)
            else:
                value = controls[w][bit] if bit < 6 else (control_token(w) >> bit) & 1
                terms += c.t_and([reset, value], 2 * KAPPA)
        for index, request_bit in enumerate(exports):
            if bit < 8:
                terms += c.t_and([request_bit, saved[index * 8 + bit]], 2 * KAPPA)
            elif (PACKET + 256 * index) >> bit & 1:
                terms += c.t_and([request_bit], 2 * KAPPA)
        c.ffn(terms, f"exit.code.{bit}", into=expr, min_level=floor + 1)
    return c


def build_standalone():
    from chipformer.q35.tiny import build as build_qwen

    circuit = build()
    dims = [next(iter(expr.terms)) for expr in circuit.code]
    return build_qwen(circuit, list(range(VOCAB)), dims, share=True), circuit
