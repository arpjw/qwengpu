"""Behavioral checks for lane masks, reconvergence, and scratch memory."""

import unittest

from qwengpu.reference import Instruction as I
from qwengpu.reference import WarpState, run


class ReferenceTests(unittest.TestCase):
    def test_vector_add_and_sum(self) -> None:
        state = run((I("ADDI", 9), I("SUM"), I("HALT")),
                    state=WarpState(lanes=tuple(range(8))))
        self.assertEqual(state.lanes, tuple(range(9, 17)))
        self.assertEqual(state.scalar, sum(range(9, 17)))

    def test_divergence_reconverges(self) -> None:
        state = run((I("IF_LT", 4), I("ADDI", 10), I("ELSE"), I("XORI", 1),
                     I("ENDIF"), I("STORE", 8), I("HALT")),
                    state=WarpState(lanes=tuple(range(8))))
        self.assertEqual(state.lanes, (10, 11, 12, 13, 5, 4, 7, 6))
        self.assertEqual(state.active_mask, 255)
        self.assertEqual(state.scratch[8:16], bytes(state.lanes))

    def test_masked_memory_and_nested_branch(self) -> None:
        program = (I("IF_LT", 6), I("IF_LT", 3), I("ADDI", 7), I("ENDIF"),
                   I("STORE", 0), I("ENDIF"), I("LOAD", 0), I("HALT"))
        state = run(program, state=WarpState(lanes=tuple(range(8))))
        self.assertEqual(state.lanes, (7, 8, 9, 3, 4, 5, 0, 0))

    def test_invalid_memory_address(self) -> None:
        with self.assertRaises(ValueError):
            run((I("STORE", 57), I("HALT")))


if __name__ == "__main__":
    unittest.main()
