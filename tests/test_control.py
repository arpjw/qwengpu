"""Frozen loop/divergence traces and loader boundaries."""

import importlib.util
import json
import unittest
from pathlib import Path
from qwengpu.control_reference import (
    ControlState,
    MAX_STEPS,
    parse_kernel,
    snapshot_bits,
    step,
)
from qwengpu.reference import WarpState

FIXTURES = Path(__file__).resolve().parents[1] / "eval/frozen/control_gate_v1"


class ControlLoaderTests(unittest.TestCase):
    def test_reference_outcomes(self):
        for path in sorted(FIXTURES.glob("*.json")):
            with self.subTest(fixture=path.name):
                data = json.loads(path.read_text())
                initial, program = parse_kernel(data)
                state = ControlState(WarpState(lanes=initial))
                for _ in range(MAX_STEPS):
                    state = step(state, program)
                    if state.warp.halted:
                        break
                self.assertEqual(
                    "halted" if state.warp.halted else "exhausted",
                    data["expected_outcome"],
                )

    def test_invalid_control_blocks(self):
        invalid = [
            ["ELSE", "HALT"],
            ["ENDIF", "HALT"],
            ["ENDWHILE", "HALT"],
            ["IF_LT", "WHILE_LT", "ENDIF", "ENDWHILE", "HALT"],
            ["IF_LT", "ELSE", "ELSE", "ENDIF", "HALT"],
            ["IF_LT"] * 3 + ["ENDIF"] * 3 + ["HALT"],
            ["WHILE_LT"] * 3 + ["ENDWHILE"] * 3 + ["HALT"],
            ["WHILE_LT", "HALT", "ENDWHILE", "HALT"],
        ]
        for ops in invalid:
            with self.subTest(ops=ops), self.assertRaises(ValueError):
                parse_kernel(
                    {"initial": [0] * 8, "program": [{"op": op} for op in ops]}
                )


@unittest.skipUnless(
    importlib.util.find_spec("chipformer"), "Qwenputer backend unavailable"
)
class CompiledControlTests(unittest.TestCase):
    def test_frozen_traces(self):
        from chipformer.q35.ir import evaluate
        from qwengpu.compiled_control import build, program_tokens, state_token, TICK
        from qwengpu.compiled_rom import load_a

        c = build()
        for path in sorted(FIXTURES.glob("*.json")):
            with self.subTest(fixture=path.name):
                initial, program = parse_kernel(json.loads(path.read_text()))
                state = ControlState(WarpState(lanes=initial))
                tokens = [load_a(i, v) for i, v in enumerate(initial)]
                tokens += program_tokens(program) + [state_token(0, 0)]
                checks = []
                for stage in range(MAX_STEPS):
                    state = step(state, program)
                    halt = state.warp.halted
                    exhausted = not halt and stage == MAX_STEPS - 1
                    token = state_token(
                        state.warp.pc,
                        stage if halt or exhausted else stage + 1,
                        halt,
                        exhausted,
                    )
                    checks.append((len(tokens), stage, token, snapshot_bits(state)))
                    tokens.extend((TICK, token))
                    if halt or exhausted:
                        break
                residual = evaluate(c, tokens, check=True, kernel=True)
                for pos, stage, expected, bits in checks:
                    predicted = sum(
                        int(residual[pos, next(iter(e.terms))] > 0) << b
                        for b, e in enumerate(c.code)
                    )
                    self.assertEqual(predicted, expected)
                    observed = [
                        int(residual[pos, next(iter(e.terms))])
                        for e in c.snapshots[stage]
                    ]
                    self.assertEqual(observed, bits)


if __name__ == "__main__":
    unittest.main()
