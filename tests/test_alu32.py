"""Check ALU32 program limits and round-trip the extended ROM payloads."""

import unittest

from qwengpu.alu32 import ALU_LIMITS, ALU_OPS, ALU_PAYLOAD, ALU_SIGNED_ARGS
from qwengpu.compiled_tile8 import program_tokens, rom_address, rom_payload
from qwengpu.eval_tile8 import parse_launch


class ALU32Tests(unittest.TestCase):
    def test_extended_instructions_at_each_rom_address(self):
        flags = dict(
            indexed=True,
            masked=True,
            divergent=True,
            looping=True,
            resident_rom32=True,
            resident_alu32=True,
        )
        for address in range(31):
            for tag, op in enumerate(ALU_OPS):
                low = -128 if op in ALU_SIGNED_ARGS else 0
                high = 127 if op in ALU_SIGNED_ARGS else ALU_LIMITS.get(op, 0)
                for arg in {low, high, 0}:
                    instructions = (("ADD32I", 0),) * address + ((op, arg), ("HALT", 0))
                    token = program_tokens(instructions, resident_alu32=True)[address]
                    self.assertEqual(
                        rom_address(token, resident_rom32=True, resident_alu32=True),
                        address,
                    )
                    immediate = -arg if op == "SUB32I" else arg
                    self.assertEqual(
                        rom_payload(token, **flags),
                        ALU_PAYLOAD | tag << 8 | (immediate & 255),
                    )
                    self.assertIsNone(rom_address(token, resident_rom32=True))

    def test_parser_rejects_values_outside_instruction_domains(self):
        for op, arg in (
            ("SHR32", 32),
            ("SLT32I", 128),
            ("ADDIDX", -129),
            ("SLTU32I", -1),
            ("SUB32I", 256),
            ("MUL32I", 32),
            ("XOR_INDEX_HI", 1),
            ("SAR32", True),
        ):
            launch = {
                "programs": [[{"op": op, "arg": arg}, {"op": "HALT"}], [{"op": "HALT"}]]
            }
            with self.assertRaisesRegex(ValueError, f"invalid {op} argument"):
                parse_launch(launch, resident_alu32=True)
        launch = {
            "programs": [
                [{"op": "ADDIDX", "arg": -128}, {"op": "HALT"}],
                [{"op": "HALT"}],
            ]
        }
        with self.assertRaisesRegex(ValueError, "invalid tile opcode"):
            parse_launch(launch, resident_rom32=True)
        self.assertEqual(
            parse_launch(launch, resident_alu32=True)["programs"][0][0],
            ("ADDIDX", -128),
        )
