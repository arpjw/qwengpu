"""Check the full integer instruction boundary and uploaded runtime extent."""

import json
import unittest
from pathlib import Path

from qwengpu.alu32 import FULL_INTEGER_OPS, alu_token, decode_alu_token
from qwengpu.eval_tile8 import oracle, parse_launch
from qwengpu.resident_scheduler import reference_dispatch

FIXTURE = (
    Path(__file__).resolve().parents[1] / "eval/frozen/full_integer_v1/indexed_words.json"
)


class FullIntegerTests(unittest.TestCase):
    def test_indexed_ops_have_distinct_wires_at_every_rom_address(self):
        seen = set()
        for address in range(32):
            for op in FULL_INTEGER_OPS:
                token = alu_token(address, op, 0, scoped=True, full_integer=True)
                self.assertNotIn(token, seen)
                seen.add(token)
                self.assertIsNone(decode_alu_token(token, scoped=True))
                self.assertEqual(
                    decode_alu_token(token, scoped=True, full_integer=True)[0], address
                )

    def test_same_program_runs_with_different_extent_descriptors(self):
        instructions = [
            {"op": name}
            for name in ("GLOBAL_ID", "SAVE_INDEX", "MASK_EXTENT", "STORE_INDEX", "HALT")
        ]
        programs = [instructions, instructions]
        words = [65535] * 256
        for extent in (0, 1, 19, 63, 64):
            raw = dict(workgroups=4, extent=extent, programs=programs, global_words=words)
            result = oracle(parse_launch(raw, full_integer=True), 256, looping=True)
            self.assertEqual(result["outcome"], "halted")
            self.assertEqual(result["shared_words"], list(range(extent)) + words[extent:])
        for extent in (-1, True, 65, None):
            with self.assertRaises(ValueError):
                parse_launch({**raw, "extent": extent}, full_integer=True)

    def test_indexed_word_opcodes_require_the_full_integer_layout(self):
        raw = json.loads(FIXTURE.read_bytes())
        with self.assertRaises(ValueError):
            parse_launch(raw, resident_interleaved=True)
        self.assertEqual(oracle(parse_launch(raw, full_integer=True), 256)["outcome"], "halted")

    def test_barrier_status_is_local_to_each_pair(self):
        self.assertEqual(reference_dispatch((2, 2, 2, 1), 1).warp, 0)
        self.assertEqual(reference_dispatch((2, 2, 2, 1), 0).warp, 1)
        self.assertIsNone(reference_dispatch((2, 1, 2, 1), 2).warp)
        self.assertTrue(reference_dispatch((1, 1, 1, 1), 0).halted)
