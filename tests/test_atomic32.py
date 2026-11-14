"""Independent outcomes for the proposed 32-bit global atomic operation."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from qwengpu.atomic32 import ATOMIC_ADD_OP, atomic_add32
from qwengpu.assemble import assemble
from qwengpu.compiled_tile8 import ATOMIC32_BASE_TAG, program_tokens, rom_payload
from qwengpu.eval_tile8 import execute, oracle, parse_launch


class Atomic32Tests(unittest.TestCase):
    def test_frozen_cross_warp_barrier_case(self):
        fixture = (
            Path(__file__).resolve().parents[1]
            / "eval/frozen/atomic32_v1/cross_warp.json"
        )
        case = parse_launch(json.loads(fixture.read_text()), atomic32=True)
        result = oracle(case, 64, looping=True)
        self.assertEqual(result["outcome"], "halted")
        self.assertEqual(result["shared_words"][:8], [11, 0, 7, 0, 11, 0, 15, 0])
        self.assertEqual(result["warp_registers"][1], list(range(3, 11)))
        self.assertEqual(
            [entry["warp"] for entry in result["trace"] if "atomic_patch" in entry],
            [0, 1],
        )

    def test_experimental_opcode_keeps_existing_rom_decode(self):
        for address in range(32):
            program = (("HALT", 0),) * address + ((ATOMIC_ADD_OP, 0),)
            token = program_tokens(
                program, resident_rom32=True, full_integer=True, atomic32=True
            )[address]
            options = dict(
                indexed=True,
                masked=True,
                divergent=True,
                looping=True,
                resident_add32=True,
                resident_shared32=True,
                resident_addmem32=True,
                resident_loop32=True,
                resident_globalgrid=True,
                resident_compute=True,
                resident_memory512=True,
                resident_rom32=True,
                resident_alu32=True,
                resident_scoped=True,
                full_integer=True,
            )
            self.assertEqual(ATOMIC32_BASE_TAG, 31)
            self.assertEqual(
                rom_payload(token, **options, atomic32=True), ATOMIC32_BASE_TAG * 256
            )
            self.assertNotEqual(rom_payload(token, **options), ATOMIC32_BASE_TAG * 256)
        with self.assertRaises(ValueError):
            program_tokens(((ATOMIC_ADD_OP, 0),), resident_rom32=True, full_integer=True)

    def test_assembled_launch_reaches_reference_interpreter(self):
        source = (
            ".warp 0\nLANE_ID\nSAVE_INDEX\nADD32I 1\n"
            "ATOMIC_ADD_INDEX32\nHALT\n.warp 1\nHALT\n"
        )
        raw = assemble(
            source,
            {"workgroups": 1, "global_words": [0] * 256},
            atomic32=True,
        )
        case = parse_launch(raw, atomic32=True)
        result = oracle(case, 32, looping=True)
        self.assertEqual(result["outcome"], "halted")
        self.assertEqual(result["shared_words"][:8], [3, 0, 7, 0, 11, 0, 15, 0])
        self.assertEqual(result["warp_registers"][0], [0, 1, 0, 3, 0, 5, 0, 7])
        atomic_trace = next(item for item in result["trace"] if "atomic_patch" in item)
        self.assertEqual(len(atomic_trace["atomic_patch"]), 32)
        with self.assertRaises(ValueError):
            parse_launch(raw, full_integer=True)
        with self.assertRaisesRegex(ValueError, "compiled atomic32 checkpoint"):
            execute(None, SimpleNamespace(), case, "cpu", 32)
        with self.assertRaises(ValueError):
            parse_launch({"programs": raw["programs"]}, atomic32=True)

    def test_colliding_lanes_return_ordered_old_values(self):
        initial = [0] * 256
        initial[0:2] = [0xFFFE, 0xFFFF]
        words, returned = atomic_add32(
            initial, [1, 0, 1, 0, 2, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8], 0b00010111
        )
        self.assertEqual(returned, [0xFFFFFFFE, 0xFFFFFFFF, 1, 4, 0, 6, 7, 8])
        self.assertEqual(words[:4], [4, 0, 5, 0])
        self.assertEqual(initial[:2], [0xFFFE, 0xFFFF])

    def test_mask_and_last_pair_boundary(self):
        initial = [0xBEEF] * 256
        words, returned = atomic_add32(
            initial, [254, 255, 0, 0, 0, 0, 0, 0], [0x10002, 3, 4, 5, 6, 7, 8, 9], 3
        )
        self.assertEqual(returned[:3], [0xBEEFBEEF, 0xBEF0BEF1, 4])
        self.assertEqual(words[254:256], [0xBEF4, 0xBEF0])
        self.assertEqual(words[:254], initial[:254])

    def test_next_warp_sees_completed_instruction(self):
        initial = [0] * 256
        first, first_old = atomic_add32(initial, [20] * 8, [1] * 8, 255)
        second, second_old = atomic_add32(first, [21] * 8, [10] * 8, 1)
        self.assertEqual(first_old, list(range(8)))
        self.assertEqual(second_old[0], 8)
        self.assertEqual(second[20:22], [18, 0])

    def test_rejects_non_word_or_non_lane_inputs(self):
        with self.assertRaises(ValueError):
            atomic_add32([0] * 255, [0] * 8, [0] * 8, 1)
        with self.assertRaises(ValueError):
            atomic_add32([0] * 256, [256] * 8, [0] * 8, 1)
        with self.assertRaises(ValueError):
            atomic_add32([0] * 256, [0] * 8, [1 << 32] * 8, 1)
        with self.assertRaises(ValueError):
            atomic_add32([0] * 256, [0] * 8, [0] * 8, 256)
