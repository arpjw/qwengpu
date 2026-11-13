"""Integer opcode encoding for the resident 32-bit arithmetic layout."""

ALU32_LAYOUT = "qwengpu-resident-alu32-9b-v1"
ALU_ROM = 0x28000
ALU_IMMEDIATE = 0x2C000
ALU_VOCAB = ALU_IMMEDIATE + 32768
ALU_PAYLOAD = 1 << 14

SHIFTS = ("SHL32", "SHR32", "SAR32")
LOGIC = ("AND_INDEX", "OR_INDEX", "XOR_INDEX")
LOGIC_OPS = tuple(f"{name}_{limb}" for name in LOGIC for limb in ("LO", "HI"))
COMPACT_OPS = (*SHIFTS, "MUL32I", *LOGIC_OPS)
IMMEDIATE_OPS = ("SUB32I", "SLTU32I", "SLT32I", "ADDIDX")
ALU_OPS = (*COMPACT_OPS, *IMMEDIATE_OPS)
ALU_LIMITS = {
    **dict.fromkeys(SHIFTS, 31),
    "MUL32I": 31,
    **dict.fromkeys(IMMEDIATE_OPS, 255),
}
ALU_SIGNED_ARGS = ("SLT32I", "ADDIDX")


def alu_token(address: int, op: str, arg: int, *, scoped=False) -> int:
    if scoped:
        from qwengpu.scoped_memory import SCOPED_OPS

        if op in SCOPED_OPS:
            return (
                ALU_ROM + address * 512 + 32 * (len(COMPACT_OPS) + SCOPED_OPS.index(op))
            )
    if op in COMPACT_OPS:
        return ALU_ROM + address * 512 + 32 * COMPACT_OPS.index(op) + arg
    if op == "SUB32I":
        arg = -arg
    return ALU_IMMEDIATE + 8192 * IMMEDIATE_OPS.index(op) + 256 * address + (arg & 255)


def decode_alu_token(token: int, *, scoped=False) -> tuple[int, int] | None:
    if ALU_ROM <= token < ALU_IMMEDIATE:
        address, wire = divmod(token - ALU_ROM, 512)
        tag, arg = divmod(wire, 32)
        if tag < len(COMPACT_OPS):
            return address, ALU_PAYLOAD + 256 * tag + arg
        if scoped and arg == 0 and len(COMPACT_OPS) <= tag < len(COMPACT_OPS) + 4:
            return address, ALU_PAYLOAD + 256 * (tag + len(IMMEDIATE_OPS))
    if ALU_IMMEDIATE <= token < ALU_VOCAB:
        variant, wire = divmod(token - ALU_IMMEDIATE, 8192)
        address, arg = divmod(wire, 256)
        return address, ALU_PAYLOAD + 256 * (len(COMPACT_OPS) + variant) + arg
    return None


def lane_results(c, ops, arg, word, gathered, lane):
    """Return full-word candidates for the masked register writeback."""
    from chipformer.dsl.graph import Expr
    from chipformer.q35.prims import from_bits

    from qwengpu.compiled_tile import _word_from_columns

    one = c.one
    name = f"alu.{lane}"

    # Reverse around one right barrel shifter to also shift left.
    shift_left = ops["SHL32"] + ops["SHL32I"]
    shift_fill = c.and_([ops["SAR32"], word[31]], f"{name}.sign")
    shifted = c.select_bits(
        [([shift_left], word[::-1]), ([one - shift_left], word)],
        32,
        f"{name}.reverse_in",
    )
    for bit, distance in enumerate((1, 2, 4, 8, 16)):
        shifted = c.select_bits(
            [
                ([arg[bit]], shifted[distance:] + [shift_fill] * distance),
                ([one - arg[bit]], shifted),
            ],
            32,
            f"{name}.shift.{bit}",
        )
    shifted = c.select_bits(
        [([shift_left], shifted[::-1]), ([one - shift_left], shifted)],
        32,
        f"{name}.reverse_out",
    )
    mul_columns = [
        c.ffn(
            [
                term
                for i in range(min(5, j + 1))
                for term in c.t_and([word[j - i], arg[i]])
            ],
            f"{name}.mul_column.{j}",
            min_level=14 + 2 * (lane // 4),
        )
        for j in range(32)
    ]
    product = _word_from_columns(
        c, mul_columns, [min(i + 1, 5) for i in range(32)], f"{name}.product"
    )

    # Compare against an 8-bit immediate using byte-wise lexicographic order.
    signed = ops["SLT32I"]
    extend = c.and_([signed, arg[7]], f"{name}.extend")
    cmp = Expr()
    for i in range(4):
        a = word[8 * i : 8 * i + 8]
        b = arg if i == 0 else [extend] * 8
        diff = from_bits(a) - from_bits(b)
        if i == 3:
            # Sign disagreement reverses the unsigned ordering.
            sign_diff = c.node(
                c.t_and([a[7], one - extend]) + c.t_and([one - a[7], extend]),
                f"{name}.sign_diff",
            )
            invert = c.and_([signed, sign_diff], f"{name}.invert")
        less = c.node(c.t_lt(diff, 0, -255, 255), f"{name}.less.{i}")
        equal = c.node(c.t_eq(diff, 0, -255, 255), f"{name}.equal.{i}")
        cmp = less + c.and_([equal, cmp], f"{name}.compare.{i}")
    compare = c.node(
        c.t_and([cmp, one - invert]) + c.t_and([one - cmp, invert]),
        f"{name}.comparison",
    )

    logic = []
    for limb, suffix in enumerate(("LO", "HI")):
        for i, rhs in enumerate(gathered):
            lhs = word[16 * limb + i]
            both = c.and_([lhs, rhs], f"{name}.logic.{limb}.{i}")
            logic.append(
                c.node(
                    c.t_and([ops[f"AND_INDEX_{suffix}"], both])
                    + c.t_and([ops[f"OR_INDEX_{suffix}"], lhs + rhs - both])
                    + c.t_and([ops[f"XOR_INDEX_{suffix}"], lhs + rhs - 2 * both])
                    + c.t_and(
                        [
                            one - sum((ops[f"{op}_{suffix}"] for op in LOGIC), Expr()),
                            lhs,
                        ]
                    ),
                    f"{name}.logic_result.{limb}.{i}",
                )
            )
    shift = ops["SHL32I"] + sum((ops[op] for op in SHIFTS), Expr())
    mul = ops["MUL32I"]
    arithmetic = [
        c.node(
            c.t_and([shift, lhs]) + c.t_and([mul, rhs]),
            f"{name}.arithmetic.{bit}",
        )
        for bit, (lhs, rhs) in enumerate(zip(shifted, product, strict=True))
    ]
    return [
        ([shift + mul], arithmetic),
        ([ops["SLTU32I"] + ops["SLT32I"]], [compare] + [Expr()] * 31),
        ([sum((ops[op] for op in LOGIC_OPS), Expr())], logic),
    ]
