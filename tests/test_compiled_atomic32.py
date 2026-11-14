"""Evaluate the atomic collision arithmetic as an isolated Qwenputer graph."""

import unittest

from chipformer.q35.ir import Target, ev, evaluate
from chipformer.q35.prims import Circ

from qwengpu.compiled_atomic32 import atomic_add32_results


def bits(c, cases, name, width, value):
    return [
        c.token_feature(
            f"{name}.{bit}",
            lambda token, bit=bit: (value(cases[token]) >> bit) & 1,
            full=True,
        )
        for bit in range(width)
    ]


def number(expressions, residual, position):
    return sum(
        int(ev(value, residual)[position]) << bit
        for bit, value in enumerate(expressions)
    )


class CompiledAtomic32Tests(unittest.TestCase):
    def assert_cases(self, cases):
        lanes = len(cases[0]["indices"])
        c = Circ(Target())
        addresses = [
            bits(
                c, cases, f"address.{lane}", 7,
                lambda case, lane=lane: case["indices"][lane] >> 1,
            )
            for lane in range(lanes)
        ]
        addends = [
            bits(
                c, cases, f"addend.{lane}", 32,
                lambda case, lane=lane: case["addends"][lane],
            )
            for lane in range(lanes)
        ]
        memory = [
            bits(
                c, cases, f"memory.{lane}", 32,
                lambda case, lane=lane: case["memory"][lane],
            )
            for lane in range(lanes)
        ]
        active = [
            bits(
                c, cases, f"active.{lane}", 1,
                lambda case, lane=lane: case["active"] >> lane,
            )[0]
            for lane in range(lanes)
        ]
        old, final, grants = atomic_add32_results(c, addresses, addends, memory, active)
        residual = evaluate(c, list(range(len(cases))), check=True)
        for position, case in enumerate(cases):
            with self.subTest(position=position):
                actual_grants = [number([grant], residual, position) for grant in grants]
                self.assertEqual(actual_grants, case["grants"])
                for lane, expected in enumerate(case["old"]):
                    if expected is not None:
                        self.assertEqual(number(old[lane], residual, position), expected)
                self.assertEqual(
                    [number(final[lane], residual, position)
                     for lane, granted in enumerate(actual_grants) if granted],
                    case["final"],
                )

    def test_collisions_wrap_masks_and_last_pair(self):
        self.assert_cases([
            {
                "indices": [0, 1, 0],
                "addends": [1, 2, 3],
                "memory": [0, 0, 0],
                "active": 7,
                "old": [0, 1, 3],
                "final": [6],
                "grants": [0, 0, 1],
            },
            {
                "indices": [254, 4, 255],
                "addends": [1, 0xFFFFFFFF, 2],
                "memory": [0xFFFFFFFE, 0, 0xFFFFFFFE],
                "active": 7,
                "old": [0xFFFFFFFE, 0, 0xFFFFFFFF],
                "final": [0xFFFFFFFF, 1],
                "grants": [0, 1, 1],
            },
            {
                "indices": [8, 9, 8],
                "addends": [2, 0xFFFFFFFF, 3],
                "memory": [5, 5, 5],
                "active": 5,
                "old": [5, None, 7],
                "final": [10],
                "grants": [0, 0, 1],
            },
        ])

    def test_all_eight_lanes_collide_with_wrap(self):
        self.assert_cases([{
            "indices": [0, 1, 0, 1, 0, 1, 0, 1],
            "addends": [0xFFFFFFFF, 1, 2, 3, 4, 5, 6, 7],
            "memory": [0] * 8,
            "active": 255,
            "old": [0, 0xFFFFFFFF, 0, 2, 5, 9, 14, 20],
            "final": [27],
            "grants": [0, 0, 0, 0, 0, 0, 0, 1],
        }])
