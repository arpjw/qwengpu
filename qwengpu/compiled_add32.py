"""Eight masked 32-bit lane additions compiled into stock Qwen3.5."""

from __future__ import annotations

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target
from chipformer.q35.prims import Circ

from qwengpu.compiled_rom import CODE_AMP, KAPPA
from qwengpu.compiled_tile import _word_from_columns

LANES = 8
WORD_BITS = 32
WORD_BYTES = WORD_BITS // 8
BYTE_SLOTS = LANES * WORD_BYTES
VOCAB = 0x8000
MASK = 0x1000
EXEC = 0x1100
READ = 0x1120
A = 0x2000
B = 0x4000
OUT = 0x6000
LAYOUT = "qwengpu-add32-9b-v1"


def add32(c: Circ, left: list[Expr], right: list[Expr], name: str) -> list[Expr]:
    """Add two words with carry from bit 15 into bit 16."""
    columns = [a + b for a, b in zip(left, right, strict=True)]
    return _word_from_columns(c, columns, [2] * WORD_BITS, name)


def build(target: Target | None = None) -> Circ:
    c = Circ(target or Target())
    one = c.one

    def token_range(name: str, base: int, count: int) -> Expr:
        return c.token_feature(
            name, lambda token: int(base <= token < base + count), full=True
        )

    execute = token_range("add32.execute", EXEC, 1)
    reads = [
        token_range(f"add32.read.{slot}", READ + slot, 1) for slot in range(BYTE_SLOTS)
    ]
    code = [
        c.token_feature(
            f"code.{bit}",
            lambda token, bit=bit: CODE_AMP * (2 * ((token >> bit) & 1) - 1),
            full=True,
        )
        for bit in range((VOCAB - 1).bit_length())
    ]
    c.code = code
    c.keep = {next(iter(expr.terms)) for expr in code}
    for expr in code:
        c.ffn([(-1, one, expr), (1, one, -expr)], "code.clear", into=expr)

    raw = [
        c.token_feature(
            f"raw.{bit}", lambda token, bit=bit: (token >> bit) & 1, full=True
        )
        for bit in range(8)
    ]
    raw_cols = {2 + bit: value for bit, value in enumerate(raw)}
    mask_write = token_range("mask.load", MASK, 256)
    mask = c.slot_head("mask", {0: mask_write}, {0: execute}, raw_cols, mask_write)

    def operand(name: str, base: int, lane: int) -> list[Expr]:
        word = []
        for byte in range(WORD_BYTES):
            slot = WORD_BYTES * lane + byte
            head = f"{name}.{lane}.{byte}"
            load = token_range(f"{head}.load", base + 256 * slot, 256)
            word.extend(c.slot_head(head, {0: load}, {0: execute}, raw_cols, load))
        return word

    selected = [Expr() for _ in range(8)]
    for lane in range(LANES):
        left = operand("a", A, lane)
        right = operand("b", B, lane)
        summed = add32(c, left, right, f"add32.lane.{lane}")
        result = c.select_bits(
            [([mask[lane]], summed), ([one - mask[lane]], left)],
            WORD_BITS,
            f"masked.{lane}",
        )
        start = lane * WORD_BYTES
        lane_reads = reads[start : start + WORD_BYTES]
        saved = c.slot_head(
            f"result.{lane}",
            {0: execute},
            {0: sum(lane_reads, Expr())},
            {2 + bit: value for bit, value in enumerate(result)},
            execute,
        )
        for byte, read in enumerate(lane_reads):
            for bit in range(8):
                selected[bit] += c.and_(
                    [read, saved[8 * byte + bit]], f"output.{lane}.{byte}.{bit}"
                )

    floor = max(node.level for node in c.nodes)
    for bit, expr in enumerate(code):
        output = (
            selected[bit]
            if bit < 8
            else sum(
                (
                    read
                    for slot, read in enumerate(reads)
                    if (OUT + 256 * slot) >> bit & 1
                ),
                Expr(),
            )
        )
        c.ffn(
            c.t_const(-KAPPA) + c.t_and([output], 2 * KAPPA),
            f"exit.code{bit}",
            into=expr,
            min_level=floor,
        )
    return c


def build_standalone():
    from chipformer.q35.tiny import build as build_qwen

    circuit = build()
    dims = [next(iter(expr.terms)) for expr in circuit.code]
    return build_qwen(circuit, list(range(VOCAB)), dims, share=True), circuit
