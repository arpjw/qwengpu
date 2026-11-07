"""Boundary and workload checks for the eight-bit memory geometry."""

import random
import unittest

from qwengpu.compiled_tile8 import ADDMEM32_OPS, program_tokens, rom_payload
from qwengpu.eval_tile8 import oracle, parse_launch
from qwengpu.memory_geometry import PAIR_WIRES, WIDE_MEMORY
from qwengpu.run_compute import assemble_launch, compile_request, parse_request, words16


def run_reference(request):
    launches = compile_request(parse_request(request, memory512=True), memory512=True)
    words = launches[0]["global_words"][:]
    for launch in launches:
        case = parse_launch({**launch, "global_words": words}, resident_memory512=True)
        result = oracle(case, 256, looping=True)
        assert result["outcome"] == "halted"
        words = result["shared_words"]
    return words, launches[0]["global_words"]


class Memory512Tests(unittest.TestCase):
    def test_geometry_and_pair_wire_domain(self):
        WIDE_MEMORY.validate()
        self.assertEqual(len(set(PAIR_WIRES.values())), 128)
        for high, op in enumerate(ADDMEM32_OPS):
            for mask in range(0, 256, 2):
                token = program_tokens(((op, mask), ("HALT", 0)))[0]
                decoded = rom_payload(
                    token,
                    indexed=True,
                    masked=True,
                    divergent=True,
                    looping=True,
                    resident_add32=True,
                    resident_shared32=True,
                    resident_addmem32=True,
                    resident_compute=True,
                    resident_memory512=True,
                )
                self.assertEqual(decoded, (19 + high) * 256 + mask)

    def test_every_word_address(self):
        initial = [(0xA000 + 257 * i) & 65535 for i in range(256)]
        for address in range(256):
            body = [
                f"ADD32I {address}",
                "SAVE_INDEX",
                "LOAD_INDEX",
                "ADD32I 1",
                "STORE_INDEX",
                "HALT",
            ]
            case = parse_launch(
                assemble_launch(body, initial, 1, ["HALT"]), resident_memory512=True
            )
            observed = oracle(case, 256, looping=True)["shared_words"]
            expected = initial[:]
            expected[address] = (expected[address] + 1) & 65535
            self.assertEqual(observed, expected)

    def test_larger_vector_and_reduction_lengths(self):
        rng = random.Random(512128)
        for count in range(129):
            a = [rng.getrandbits(32) for _ in range(count)]
            if count <= 64:
                b = [rng.getrandbits(32) for _ in range(count)]
                actual, initial = run_reference(
                    {"kernel": "vector_add", "a": a, "b": b}
                )
                expected = initial[:]
                expected[: 2 * count] = words16(
                    [(x + y) & 0xFFFFFFFF for x, y in zip(a, b, strict=True)]
                )
                self.assertEqual(actual, expected)
            actual, initial = run_reference({"kernel": "reduce", "values": a})
            expected = initial[:]
            for start in range(0, min(count, 64), 8):
                partial = (
                    sum(a[start : start + 8]) + sum(a[64 + start : 72 + start])
                ) & 0xFFFFFFFF
                expected[2 * start : 2 * start + 2] = words16([partial])
            expected[:2] = words16([sum(a) & 0xFFFFFFFF])
            self.assertEqual(actual, expected)

    def test_cross_bank_collision_priority(self):
        initial = [0xABCD] * 256
        body = ["LANE_ID", "SHL32I 7", "SAVE_INDEX", "LANE_ID", "STORE_INDEX", "HALT"]
        case = parse_launch(
            assemble_launch(body, initial, 1, ["HALT"]), resident_memory512=True
        )
        observed = oracle(case, 256, looping=True)["shared_words"]
        expected = initial[:]
        expected[0], expected[128] = 0, 1
        self.assertEqual(observed, expected)
