"""ROM upload and link encoding at the 16-instruction boundary."""

import unittest

from qwengpu.compiled_tile8 import program_tokens, rom_address, rom_payload
from qwengpu.eval_tile8 import parse_launch


class Rom32Tests(unittest.TestCase):
    def test_loop_targets_and_predicates(self):
        flags = {
            "indexed": True,
            "masked": True,
            "divergent": True,
            "looping": True,
            "resident_loop32": True,
            "resident_compute": True,
            "resident_memory512": True,
            "resident_rom32": True,
        }
        for exit_pc in range(2, 32):
            for start in range(exit_pc - 1):
                for predicate in (0, 15, 16, 31):
                    program = [("ADD32I", 0)] * (exit_pc + 1)
                    program[start] = ("WHILE_BIT", predicate)
                    program[exit_pc - 1] = ("ENDWHILE", 0)
                    program[exit_pc] = ("HALT", 0)
                    tokens = program_tokens(tuple(program), resident_rom32=True)
                    head = rom_payload(tokens[start], **flags)
                    tail = rom_payload(tokens[exit_pc - 1], **flags)
                    self.assertEqual(
                        rom_address(tokens[start], resident_rom32=True), start
                    )
                    self.assertEqual(
                        rom_address(tokens[exit_pc - 1], resident_rom32=True),
                        exit_pc - 1,
                    )
                    self.assertEqual((head >> 8) & 31, 14 if predicate < 16 else 21)
                    self.assertEqual(head & 15, predicate % 16)
                    self.assertEqual(((head >> 4) & 15) | ((head >> 13) << 4), exit_pc)
                    self.assertEqual(tail >> 8, 15)
                    self.assertEqual(tail & 255, start)

    def test_launch_length_is_layout_specific(self):
        program = [{"op": "ADD32I", "arg": 1}] * 31 + [{"op": "HALT"}]
        data = {"programs": [program, [{"op": "HALT"}]]}
        self.assertEqual(
            len(parse_launch(data, resident_rom32=True)["programs"][0]), 32
        )
        with self.assertRaisesRegex(ValueError, "through 16"):
            parse_launch(data, resident_memory512=True)
        with self.assertRaisesRegex(ValueError, "through 32"):
            parse_launch(
                {"programs": [[{"op": "ADD32I"}, *program], program]},
                resident_rom32=True,
            )
