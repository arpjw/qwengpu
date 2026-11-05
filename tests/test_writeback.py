"""Validate internally latched state against independent instruction semantics."""

import importlib.util
import json
import unittest
from pathlib import Path
from qwengpu.reference import WarpState, step
from qwengpu.run_kernel import parse_kernel


@unittest.skipUnless(
    importlib.util.find_spec("chipformer"), "Qwenputer backend unavailable"
)
class WritebackTests(unittest.TestCase):
    def test_parallel_add_all_byte_pairs(self) -> None:
        import numpy as np
        from chipformer.q35.ir import evaluate
        from chipformer.q35.prims import Circ
        from qwengpu.compiled_writeback import _add_byte

        c = Circ()
        left = [
            c.token_feature(f"a.{b}", lambda t, b=b: (t >> b) & 1) for b in range(8)
        ]
        right = [
            c.token_feature(f"b.{b}", lambda t, b=b: (t >> (8 + b)) & 1)
            for b in range(8)
        ]
        summed = _add_byte(c, left, right, "sum")
        residual = evaluate(c, list(range(65536)), check=True, kernel=True)
        observed = np.zeros(65536, dtype=np.int64)
        for b, expr in enumerate(summed):
            values = sum(
                (residual[:, d] * coefficient for d, coefficient in expr.terms.items())
            )
            observed += values.astype(np.int64) << b
        tokens = np.arange(65536)
        np.testing.assert_array_equal(observed, ((tokens & 255) + (tokens >> 8)) & 255)

    def test_frozen_internal_snapshots(self):
        from chipformer.q35.ir import evaluate
        from qwengpu.compiled_writeback import build
        from qwengpu.compiled_rom import (
            TICK,
            KAPPA,
            load_a,
            program_tokens,
            state_token,
        )

        c = build()
        root = Path(__file__).resolve().parents[1] / "eval/frozen/writeback_gate_v1"
        for path in sorted(root.glob("*.json")):
            with self.subTest(fixture=path.name):
                initial, program = parse_kernel(json.loads(path.read_text()))
                tokens = [load_a(lane, value) for lane, value in enumerate(initial)]
                tokens += program_tokens(program) + [state_token(0)]
                state = WarpState(lanes=initial)
                checks = []
                while not state.halted:
                    stage = state.pc
                    state = step(state, program)
                    checks.append((len(tokens), stage, state))
                    tokens.extend((TICK, state_token(state.pc, state.halted)))
                residual = evaluate(c, tokens, check=True, kernel=True)
                for position, stage, state in checks:
                    predicted = sum(
                        int(residual[position, next(iter(expr.terms))] > 0) << bit
                        for bit, expr in enumerate(c.code)
                    )
                    self.assertEqual(predicted, state_token(state.pc, state.halted))
                    expected = list(state.lanes) + list(state.scratch[:32])
                    for bit, expr in enumerate(c.snapshots[stage]):
                        value = residual[position, next(iter(expr.terms))]
                        self.assertEqual(value, (expected[bit // 8] >> (bit % 8)) & 1)


if __name__ == "__main__":
    unittest.main()
