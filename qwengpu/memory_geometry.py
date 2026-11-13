"""Memory and snapshot geometry for the compatible indexed processor layouts."""

from dataclasses import dataclass

from qwengpu.alu32 import ALU32_LAYOUT
from qwengpu.scoped_memory import SCOPED_LAYOUT
from qwengpu.resident_scheduler import RESIDENT_LAYOUT

MEMORY512_LAYOUT = "qwengpu-resident-memory512-9b-v1"
ROM32_LAYOUT = "qwengpu-resident-rom32-9b-v1"
ALU32_LAYOUTS = (ALU32_LAYOUT, SCOPED_LAYOUT, RESIDENT_LAYOUT)
ROM32_LAYOUTS = (ROM32_LAYOUT, *ALU32_LAYOUTS)
MEMORY512_LAYOUTS = (MEMORY512_LAYOUT, *ROM32_LAYOUTS)
MEMORY512_BASE = 0x20000
MEMORY512_VOCAB = 0x24000


@dataclass(frozen=True)
class MemoryGeometry:
    words: int
    base: int | None
    vocab: int

    @property
    def index_bits(self) -> int:
        return (self.words - 1).bit_length()

    @property
    def extra_index_offset(self) -> int:
        return 388

    @property
    def dirty_offset(self) -> int:
        return 400 if self.words == 256 else 388

    @property
    def mode_offset(self) -> int:
        return 396

    @property
    def upload_tokens(self) -> int:
        return 4 * self.words

    def validate(self) -> None:
        if self.words not in (128, 256):
            raise ValueError("supported memory geometries contain 128 or 256 words")
        if self.base is not None:
            if self.base < 0x20000 or self.base % 16:
                raise ValueError("expanded memory must not overlap legacy token ranges")
            if self.base + 64 * self.words > self.vocab:
                raise ValueError("expanded memory exceeds the chip vocabulary")


LEGACY_MEMORY = MemoryGeometry(128, None, 1 << 17)
WIDE_MEMORY = MemoryGeometry(256, MEMORY512_BASE, MEMORY512_VOCAB)
LEGACY_MEMORY.validate()
WIDE_MEMORY.validate()

# Old ADD_INDEX immediates retain their wire representation. Only the upper
# half of the new even-mask domain uses these previously unused immediates.
PAIR_WIRES = {
    (high, 128 + 2 * index): wire
    for high, wires in enumerate(
        (
            [3 * 256 + value for value in range(192, 256)],
            [3 * 256 + value for value in range(97, 128)]
            + [4 * 256 + value for value in range(214, 247)],
        )
    )
    for index, wire in enumerate(wires)
}
PAIR_DECODE = {
    wire: (19 + high) * 256 + mask for (high, mask), wire in PAIR_WIRES.items()
}
assert len(PAIR_WIRES) == len(PAIR_DECODE) == 128
