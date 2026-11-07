"""Two sequential workgroups operating on one resident vector image."""

from __future__ import annotations

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target
from chipformer.q35.prims import Circ, from_bits

from qwengpu.compiled_add32 import LANES, WORD_BYTES, add32
from qwengpu.compiled_rom import CODE_AMP, KAPPA

GROUPS = 2
WORDS = GROUPS * LANES
BYTE_SLOTS = WORDS * WORD_BYTES
REGION_SIZE = 256 * BYTE_SLOTS
A = 0
B = A + REGION_SIZE
C = B + REGION_SIZE
OUT = C + REGION_SIZE
READ = OUT + 256
LENGTH = READ + BYTE_SLOTS
START = LENGTH + WORDS + 1
STEP = START + 1
GROUP = STEP + 1
DONE = GROUP + GROUPS
VOCAB = 1 << 16


def build(target: Target | None = None) -> Circ:
    c = Circ(target or Target())
    one = c.one

    def token_range(name: str, base: int, count: int = 1) -> Expr:
        return c.token_feature(
            name, lambda token: int(base <= token < base + count), full=True
        )

    start = token_range("grid.start", START)
    step = token_range("grid.step", STEP)
    descriptor = token_range("grid.length.load", LENGTH, WORDS + 1)
    length_bits = [
        c.token_feature(
            f"grid.length.{bit}", lambda t, bit=bit: ((t - LENGTH) >> bit) & 1,
            full=True,
        )
        for bit in range(WORDS.bit_length())
    ]
    length = from_bits(c.slot_head(
        "grid.length", {0: descriptor}, {0: start + step},
        {2 + bit: value for bit, value in enumerate(length_bits)}, descriptor,
    ))
    group_one = token_range("grid.group.1", GROUP + 1)
    ticket = token_range("grid.ticket", GROUP, GROUPS + 1)
    runnable = token_range("grid.runnable", GROUP, GROUPS)
    group, live = c.slot_head(
        "grid.cursor", {0: ticket}, {0: step}, {2: group_one, 3: runnable}, ticket,
    )
    issue = c.and_([step, live], "grid.issue")
    first = c.node(c.t_ge(length, 1, 0, WORDS, lits=[start]), "grid.first")
    second = c.node(
        c.t_ge(length, LANES + 1, 0, WORDS, lits=[issue, one - group]), "grid.second"
    )
    done = start + step - first - second

    raw_byte = [
        c.token_feature(f"byte.{bit}", lambda t, bit=bit: (t >> bit) & 1, full=True)
        for bit in range(8)
    ]
    selected = [Expr() for _ in range(8)]
    reading = token_range("grid.read", READ, BYTE_SLOTS)
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
    for value in code:
        c.ffn([(-1, one, value), (1, one, -value)], "code.clear", into=value)

    def upload_keys(name: str, base: int, lane: int, byte: int) -> dict[int, Expr]:
        return {
            slot: token_range(
                f"{name}.{lane}.{byte}.load.{slot}",
                base + 256 * ((slot * LANES + lane) * WORD_BYTES + byte), 256,
            )
            for slot in range(GROUPS)
        }

    for lane in range(LANES):
        # Each lane owns two physical rows. The model's group bit supplies the
        # high address bit, and the fixed lane number supplies the low bits.
        address = LANES * group + lane
        active = c.node(
            c.t_ge(length - address, 1, -(WORDS - 1), WORDS, lits=[issue]),
            f"grid.active.{lane}",
        )
        access = {
            0: c.and_([active, one - group], f"grid.access.{lane}.0"),
            1: c.and_([active, group], f"grid.access.{lane}.1"),
        }
        operands = []
        for name, base in (("a", A), ("b", B)):
            word = []
            for byte in range(WORD_BYTES):
                keys = upload_keys(name, base, lane, byte)
                word.extend(c.slot_head(
                    f"{name}.{lane}.{byte}", keys, access,
                    {2 + bit: value for bit, value in enumerate(raw_byte)},
                    sum(keys.values(), Expr()),
                ))
            operands.append(word)
        result = add32(c, *operands, f"grid.add.{lane}")
        for byte in range(WORD_BYTES):
            keys = upload_keys("c", C, lane, byte)
            upload = sum(keys.values(), Expr())
            slots = [(slot * LANES + lane) * WORD_BYTES + byte for slot in range(GROUPS)]
            reads = {slot: token_range(f"c.read.{address}", READ + address)
                     for slot, address in enumerate(slots)}
            data = c.select_bits(
                [([upload], raw_byte), ([active], result[8 * byte : 8 * (byte + 1)])],
                8, f"c.data.{lane}.{byte}",
            )
            saved = c.slot_head(
                f"c.{lane}.{byte}", {slot: key + access[slot] for slot, key in keys.items()},
                reads, {2 + bit: value for bit, value in enumerate(data)}, upload + active,
            )
            read = sum(reads.values(), Expr())
            for bit, value in enumerate(saved):
                selected[bit] += c.and_([read, value], f"c.output.{lane}.{byte}.{bit}")

    floor = max(node.level for node in c.nodes)
    for bit, value in enumerate(code):
        output = selected[bit] if bit < 8 else reading if OUT >> bit & 1 else Expr()
        for token, gate in ((GROUP, first), (GROUP + 1, second), (DONE, done)):
            if token >> bit & 1:
                output += gate
        c.ffn(c.t_const(-KAPPA) + c.t_and([output], 2 * KAPPA),
              f"exit.code{bit}", into=value, min_level=floor)
    return c


def build_standalone():
    from chipformer.q35.tiny import build as build_qwen

    circuit = build()
    dims = [next(iter(expr.terms)) for expr in circuit.code]
    return build_qwen(circuit, list(range(VOCAB)), dims, share=True), circuit
