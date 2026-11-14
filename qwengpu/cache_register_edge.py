"""Optional inference-runtime edge from the model snapshot to its register banks."""

from __future__ import annotations

from qwengpu.eval_compiled_rom import _head_location


def _names():
    from qwengpu.compiled_tile8 import BRANCH_NIBBLES, LOOP_NIBBLES

    return (
        [f"register.{nib}" for nib in range(32)]
        + [f"index.{lane}.{nib}" for lane in range(8) for nib in range(2)]
        + [f"mask.{nib}" for nib in range(2)]
        + [f"branch.{nib}" for nib in range(BRANCH_NIBBLES)]
        + [f"loop.{nib}" for nib in range(LOOP_NIBBLES)]
        + [f"high.{nib}" for nib in range(32)]
    )


def snapshot_locations(circuit) -> list[tuple[int, ...]]:
    """Map each register nibble to the four stored snapshot bit positions."""
    from qwengpu.compiled_tile8 import MASK_OFFSET, SNAPSHOT_BITS, WARP_DIRTY_OFFSET

    bits: list[int] = list(range(128))
    for lane in range(8):
        bits.extend(range(SNAPSHOT_BITS + 7 * lane, SNAPSHOT_BITS + 7 * (lane + 1)))
        bits.append(circuit.memory_geometry.extra_index_offset + lane)
    bits.extend(range(MASK_OFFSET, WARP_DIRTY_OFFSET))
    bits.extend(range(WARP_DIRTY_OFFSET, WARP_DIRTY_OFFSET + 128))
    return [tuple(bits[i : i + 4]) for i in range(0, len(bits), 4)]


class CacheRegisterEdge:
    """Copy exact model tensors, using a fixed map and a model-issued warp ID."""

    def __init__(self, circuit):
        from qwengpu.compiled_tile8 import CHUNK_BITS

        ratio = circuit.target.value_heads // circuit.target.key_heads
        moves = []
        for name, bits in zip(_names(), snapshot_locations(circuit), strict=True):
            dst_layer, dst_head = _head_location(circuit, name)
            columns = []
            for bit in bits:
                chunk, col = divmod(bit, CHUNK_BITS)
                layer, head = _head_location(circuit, f"stage.0.{chunk}")
                columns.append((layer, head * ratio, 2 + col))
            src_layer, src_head, first = columns[0]
            contiguous = columns == [(src_layer, src_head, first + i) for i in range(4)]
            moves.append((dst_layer, dst_head * ratio, tuple(columns), contiguous))
        self.moves = tuple(moves)

    def apply(self, cache, warp: int) -> None:
        for dst_layer, dst_head, columns, contiguous in self.moves:
            dst = cache.layers[dst_layer].recurrent_states[0][0, dst_head, warp]
            src_layer, src_head, first = columns[0]
            if contiguous:
                source = cache.layers[src_layer].recurrent_states[0][0, src_head, 0]
                dst[2:6].copy_(source[first : first + 4])
                continue
            for column, (src_layer, src_head, src_col) in enumerate(columns, 2):
                source = cache.layers[src_layer].recurrent_states[0][0, src_head, 0]
                dst[column].copy_(source[src_col])
