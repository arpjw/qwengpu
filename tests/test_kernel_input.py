"""Reject programs that exceed the compiled circuit's supported subset."""

import json
import unittest
from pathlib import Path

from qwengpu.run_kernel import parse_kernel


class KernelInputTests(unittest.TestCase):
    def test_example_is_valid(self) -> None:
        path = Path(__file__).resolve().parents[1] / "examples/mixed_kernel.json"
        initial, program = parse_kernel(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(initial, tuple(range(8)))
        self.assertEqual(program[-1].op, "HALT")

    def test_nested_branch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "one branch level"):
            parse_kernel({"initial": [0] * 8, "program": [
                {"op": "IF_LT", "arg": 4}, {"op": "IF_LT", "arg": 2},
                {"op": "ENDIF"}, {"op": "ENDIF"}, {"op": "HALT"}]})

    def test_unavailable_scratch_address_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "scratch addresses"):
            parse_kernel({"initial": [0] * 8, "program": [
                {"op": "STORE", "arg": 32}, {"op": "HALT"}]})


if __name__ == "__main__":
    unittest.main()
