"""Check the frozen cross-workgroup dependency and token range separation."""

import json
import unittest
from pathlib import Path

from qwengpu import resident_generation as gen
from qwengpu.compiled_tile8 import WIDE_WARP_TAGS, rom_address
from qwengpu.eval_resident_generation import expected_tokens, launch_prompt, resident_reference
from qwengpu.eval_tile8 import oracle, parse_launch
from qwengpu.resident_scheduler import resident_control


class ResidentGenerationTests(unittest.TestCase):
    def test_single_workgroup_leaves_two_warps_halted(self):
        root = Path(__file__).resolve().parents[1] / "eval/frozen/full_integer_v1"
        raw = json.loads((root / "indexed_words.json").read_text())
        case = parse_launch(raw, resident_globalgrid=True, full_integer=True)
        prompt = launch_prompt(case)
        inactive = [resident_control(warp, status=1) for warp in (2, 3)]
        self.assertEqual(prompt[3:5], inactive)
        reference = resident_reference(oracle(case, 256, looping=True))
        tokens, controls, _ = expected_tokens(reference, active_warps=2)
        self.assertEqual(controls[2:], inactive)
        self.assertIn(gen.STEP_CONTROL + 16, tokens)
        self.assertFalse(any(gen.TICKET + 2 <= token < gen.TICKET + 4 for token in tokens))

    def test_cross_group_high_limb(self):
        fixture = Path(__file__).resolve().parents[1] / (
            "eval/frozen/resident_interleaved_v1/cross_group_high_limb.json"
        )
        case = parse_launch(json.loads(fixture.read_text()), resident_globalgrid=True,
                            resident_compute=True, full_integer=True)
        reference = oracle(case, 256, looping=True)
        self.assertEqual(reference["shared_words"][:4], [71, 1, 72, 1])
        self.assertEqual(reference["warp_registers"][2], [0x10048] * 8)
        tokens, _, _ = expected_tokens(reference)
        prompt = launch_prompt(case)
        self.assertEqual(set(tokens) & set(prompt), {gen.SCHEDULE})
        high = gen.REGISTER + 2 * gen.REGISTER_STRIDE + 16 * (len(WIDE_WARP_TAGS) - 32)
        self.assertGreater(sum(high <= token < high + 512 for token in tokens), 0)
        for token in set(tokens):
            self.assertIsNone(rom_address(token, resident_rom32=True, resident_alu32=True,
                                          resident_scoped=True, full_integer=True))
