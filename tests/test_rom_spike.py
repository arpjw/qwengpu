"""A shared graph fetches different uploaded instructions and advances PC."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("chipformer"), "Qwenputer backend unavailable")
class RomSpikeTests(unittest.TestCase):
    def test_three_programs_one_graph(self) -> None:
        from chipformer.q35.ir import evaluate

        from qwengpu.rom_spike import (ADVANCE, KAPPA, PAD, ROM_SLOTS,
                                      SAMPLE_PROGRAMS, TICK, build,
                                      instruction_token, load_token)

        circuit = build()
        for program in SAMPLE_PROGRAMS:
            with self.subTest(program=program):
                tokens = [PAD, *(load_token(i, value)
                                 for i, value in enumerate(program)), 0]
                checks = []
                for pc, payload in enumerate(program):
                    checks.append((len(tokens), instruction_token(payload)))
                    tokens.extend((TICK, instruction_token(payload)))
                    if pc < ROM_SLOTS - 1:
                        checks.append((len(tokens), pc + 1))
                        tokens.extend((ADVANCE, pc + 1))
                residual = evaluate(circuit, tokens, check=True)
                for position, expected in checks:
                    predicted = 0
                    for bit, expr in enumerate(circuit.code):
                        value = residual[position, next(iter(expr.terms))]
                        self.assertEqual(abs(value), KAPPA)
                        predicted |= int(value > 0) << bit
                    self.assertEqual(predicted, expected)


if __name__ == "__main__":
    unittest.main()
