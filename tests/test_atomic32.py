"""Independent outcomes for the proposed 32-bit global atomic operation."""

import unittest

from qwengpu.atomic32 import atomic_add32


class Atomic32Tests(unittest.TestCase):
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
