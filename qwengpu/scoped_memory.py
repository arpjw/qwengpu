"""Word addresses for workgroup shared and lane private memory."""

SCOPED_LAYOUT = "qwengpu-resident-scoped-9b-v1"
SCOPED_BASE = 0x18000
SCOPED_WORDS = 128
SCOPED_END = SCOPED_BASE + 64 * SCOPED_WORDS
SCOPED_LOADS = ("LOAD_SHARED", "LOAD_PRIVATE")
SCOPED_STORES = ("STORE_SHARED", "STORE_PRIVATE")
SCOPED_OPS = (*SCOPED_LOADS, *SCOPED_STORES)


def physical_word(op: str, group: int, warp: int, lane: int, index: int) -> int:
    if op.endswith("_PRIVATE"):
        return 64 + 16 * group + 8 * warp + lane
    return 16 * group + (index & 15)


def canonical_nibble(token: int, operand_base: int, output_base: int) -> int:
    """Alias scoped token addresses onto the physical rows of the tiled memory ports."""
    if SCOPED_BASE <= token < SCOPED_END:
        slot = (token - SCOPED_BASE) // 16
        base = output_base if slot >= 256 else operand_base
        return base // 16 + slot % 256
    return token // 16
