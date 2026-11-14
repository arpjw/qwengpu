"""Symbolic lane results for a 32-bit global atomic-add instruction.

The caller supplies the same pre-instruction memory snapshot to every lane.
This primitive computes old values, final values, and the last active writer
for each aliased pair. It does not yet connect those values to ROM or RAM.
"""

from chipformer.dsl.graph import Expr
from chipformer.q35.prims import Circ, from_bits

from qwengpu.compiled_tile import _word_from_columns


def atomic_add32_results(
    c: Circ,
    pair_addresses: list[list[Expr]],
    addends: list[list[Expr]],
    memory: list[list[Expr]],
    active: list[Expr],
) -> tuple[list[list[Expr]], list[list[Expr]], list[Expr]]:
    """Return per-lane old words, final words, and distinct-pair commit grants.

    Pair addresses omit bit zero. Within an instruction, lower lane IDs win
    earlier positions in the read-modify-write order. Each granted lane's
    final word contains all active additions to its pair.
    """
    lanes = len(active)
    if not (
        len(pair_addresses) == len(addends) == len(memory) == lanes
        and all(len(bits) == 7 for bits in pair_addresses)
        and all(len(bits) == 32 for bank in (addends, memory) for bits in bank)
    ):
        raise ValueError("atomic inputs require matching lanes, 7-bit pairs, and 32-bit words")

    equal: dict[tuple[int, int], Expr] = {}
    for lane in range(lanes):
        for earlier in range(lane):
            difference = from_bits(pair_addresses[earlier]) - from_bits(pair_addresses[lane])
            equal[earlier, lane] = c.node(
                c.t_eq(difference, 0, -127, 127),
                f"atomic.same.{earlier}.{lane}",
            )

    old_words = []
    final_words = []
    grants = []
    for lane in range(lanes):
        prior = [
            c.and_([active[earlier], equal[earlier, lane]],
                   f"atomic.prior.{earlier}.{lane}")
            for earlier in range(lane)
        ]
        if prior:
            columns = [
                memory[lane][bit]
                + sum(
                    (c.and_([enabled, addends[earlier][bit]],
                            f"atomic.contribution.{earlier}.{lane}.{bit}")
                     for earlier, enabled in enumerate(prior)),
                    Expr(),
                )
                for bit in range(32)
            ]
            old = _word_from_columns(
                c, columns, [1 + len(prior)] * 32, f"atomic.old.{lane}"
            )
        else:
            old = memory[lane]
        final = _word_from_columns(
            c,
            [left + right for left, right in zip(old, addends[lane], strict=True)],
            [2] * 32,
            f"atomic.final.{lane}",
        )
        later = [
            c.and_([active[other], equal[lane, other]],
                   f"atomic.later.{lane}.{other}")
            for other in range(lane + 1, lanes)
        ]
        grants.append(
            c.and_(
                [active[lane], *(c.one - same for same in later)],
                f"atomic.grant.{lane}",
            )
        )
        old_words.append(old)
        final_words.append(final)
    return old_words, final_words, grants


def atomic_nibble_token_bits(
    c: Circ,
    address: list[Expr],
    word: list[Expr],
    nibble: list[Expr],
) -> list[Expr]:
    """Encode one value-bearing 512-byte RAM write token for generation.

    The token format is 0x20000 + 64 * address + 16 * nibble + value_nibble.
    A later generation step can feed that token back into the RAM head using
    only token features, avoiding a same-layer RAM read/write dependency.
    """
    if len(address) != 8 or len(word) != 16 or len(nibble) != 2:
        raise ValueError("RAM token needs an 8-bit address, 16-bit word, and 2-bit nibble")
    selected = c.select_bits(
        [([hot], word[4 * index : 4 * index + 4])
         for index, hot in enumerate(c.onehot_of_bits(nibble, "atomic.token.nibble", 4))],
        4,
        "atomic.token.value",
    )
    return [*selected, *nibble, *address, Expr(), Expr(), Expr(), c.one]


def atomic_nibble_step(
    c: Circ,
    command: Expr,
    memory_token: Expr,
    input_nibble: list[Expr],
    address: list[Expr],
    word: list[Expr],
) -> tuple[Expr, list[Expr], Expr]:
    """Emit nibble zero after a command, then nibbles one through three.

    Return the emission gate, generated memory token bits, and a completion
    gate. Completion after nibble three lets the scheduler select the next
    committed limb. The caller keeps address and word in a staged head and
    enables memory_token only while an atomic writeback is active.
    """
    if len(input_nibble) != 2:
        raise ValueError("input memory nibble must have two bits")
    hot = c.onehot_of_bits(input_nibble, "atomic.stream.input_nibble", 4)
    continuation = c.and_([memory_token, c.one - hot[3]], "atomic.stream.continue")
    next_nibble = [
        c.and_([memory_token, hot[0] + hot[2]], "atomic.stream.next_low"),
        c.and_([memory_token, hot[1] + hot[2]], "atomic.stream.next_high"),
    ]
    return (
        command + continuation,
        atomic_nibble_token_bits(c, address, word, next_nibble),
        c.and_([memory_token, hot[3]], "atomic.stream.done"),
    )


def generated_memory_gate(
    c: Circ, raw_memory_token: Expr, launch_start: Expr, schedule: Expr
) -> Expr:
    """Distinguish generated RAM tags from the same tags in a launch upload.

    A launch starts with a reset marker, uploads memory, then supplies a
    schedule token. The one-row head is zero before that schedule and one
    afterward. A new launch marker resets it again in the retained cache.
    """
    phase = c.slot_head(
        "atomic.generation_phase",
        {0: launch_start + schedule},
        {0: raw_memory_token},
        {2: schedule},
        launch_start + schedule,
    )
    return c.and_([raw_memory_token, phase[0]], "atomic.generated_memory")
