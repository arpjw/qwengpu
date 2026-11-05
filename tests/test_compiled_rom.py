"""One circuit must fetch and execute the frozen straight-line kernels."""

import importlib.util
import json
import unittest
from pathlib import Path

from qwengpu.reference import WarpState, step
from qwengpu.run_kernel import parse_kernel


@unittest.skipUnless(importlib.util.find_spec("chipformer"), "Qwenputer backend unavailable")
class CompiledROMTests(unittest.TestCase):
    def test_frozen_programs_one_graph(self) -> None:
        from chipformer.q35.ir import evaluate

        from qwengpu.compiled_rom import (KAPPA, TICK, build, load_a, out_token,
                                          program_tokens, read_lane, state_token)

        circuit = build()
        fixtures = Path(__file__).resolve().parents[1] / "eval/frozen/rom_gate_v1"
        for path in sorted(fixtures.glob("*.json")):
            with self.subTest(fixture=path.name):
                initial, program = parse_kernel(json.loads(path.read_text()))
                state = WarpState(lanes=initial)
                tokens = [load_a(lane, value) for lane, value in enumerate(initial)]
                tokens.extend(program_tokens(program))
                tokens.append(state_token(0))
                checks = []
                while not state.halted:
                    state = step(state, program)
                    generated_state = state_token(state.pc, state.halted)
                    checks.append((len(tokens), generated_state))
                    tokens.extend((TICK, generated_state))
                    if not state.halted:
                        for lane, value in enumerate(state.lanes):
                            checks.append((len(tokens), out_token(lane, value)))
                            tokens.extend((read_lane(lane), out_token(lane, value)))
                residual = evaluate(circuit, tokens, check=True, kernel=True)
                for position, expected in checks:
                    predicted = 0
                    for bit, expr in enumerate(circuit.code):
                        value = residual[position, next(iter(expr.terms))]
                        self.assertEqual(abs(value), KAPPA)
                        predicted |= int(value > 0) << bit
                    self.assertEqual(predicted, expected)


if __name__ == "__main__":
    unittest.main()
