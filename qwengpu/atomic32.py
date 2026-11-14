"""Reference transition for a future 32-bit global atomic-add instruction.

This module does not add an opcode to a compiled checkpoint. It defines the
instruction-level result that a new circuit must reproduce.
"""

from collections.abc import Sequence


def atomic_add32(
    global_words: Sequence[int],
    indices: Sequence[int],
    addends: Sequence[int],
    active: int,
) -> tuple[list[int], list[int]]:
    """Apply one warp instruction, visiting active lanes from zero through seven.

    The eight-bit index selects a 32-bit pair by clearing its low bit. Each
    lane returns the value observed before its addition. Inactive lanes keep
    their register values. Additions wrap modulo 2**32. The returned memory
    is a copy; callers commit it only after the whole instruction completes.
    """
    if len(global_words) != 256 or any(
        type(word) is not int or not 0 <= word < 65536 for word in global_words
    ):
        raise ValueError("global_words must contain 256 unsigned 16-bit words")
    if len(indices) != 8 or any(
        type(index) is not int or not 0 <= index < 256 for index in indices
    ):
        raise ValueError("indices must contain eight unsigned 8-bit addresses")
    if len(addends) != 8 or any(
        type(value) is not int or not 0 <= value < 1 << 32 for value in addends
    ):
        raise ValueError("addends must contain eight unsigned 32-bit values")
    if type(active) is not int or not 0 <= active < 256:
        raise ValueError("active must be an 8-bit lane mask")

    words = list(global_words)
    returned = list(addends)
    for lane, (index, value) in enumerate(zip(indices, addends, strict=True)):
        if not active >> lane & 1:
            continue
        address = index & ~1
        old = words[address] | words[address + 1] << 16
        new = (old + value) & 0xFFFFFFFF
        returned[lane] = old
        words[address] = new & 0xFFFF
        words[address + 1] = new >> 16
    return words, returned
