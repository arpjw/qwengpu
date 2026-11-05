"""The compiled graph must calculate eight masked byte sums exactly."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("chipformer"), "Qwenputer backend unavailable")
class CompiledAddTests(unittest.TestCase):
    def test_masked_add_multiple_inputs_one_graph(self) -> None:
        from chipformer.q35.ir import evaluate

        from qwengpu.compiled_add import (EXEC_ADD, KAPPA, build,
                                          load_a, load_b, mask_token, out_token,
                                          read_lane)

        circuit = build()
        self.assertEqual(circuit.stats()["heads"], 18)
        for a, b, mask in (
            (tuple(range(8)), (10,) * 8, 255),
            ((255, 0, 1, 2, 3, 4, 5, 6), (1, 255, 254, 200, 1, 2, 3, 4), 0b01010101),
            ((100,) * 8, (50,) * 8, 0),
        ):
            with self.subTest(mask=mask, a=a, b=b):
                expected = tuple((a[lane] + b[lane]) & 255 if mask >> lane & 1
                                 else a[lane] for lane in range(8))
                tokens = []
                for lane in range(8):
                    tokens.extend((load_a(lane, a[lane]), load_b(lane, b[lane])))
                tokens.extend((mask_token(mask), EXEC_ADD))
                reads = []
                for lane, value in enumerate(expected):
                    reads.append(len(tokens))
                    tokens.extend((read_lane(lane), out_token(value)))
                residual = evaluate(circuit, tokens, check=True)
                for lane, position in enumerate(reads):
                    predicted = 0
                    for bit, expr in enumerate(circuit.code):
                        value = residual[position, next(iter(expr.terms))]
                        self.assertEqual(abs(value), KAPPA)
                        predicted |= int(value > 0) << bit
                    self.assertEqual(predicted, out_token(expected[lane]))

    def test_comparison_produces_lane_mask(self) -> None:
        from chipformer.q35.ir import evaluate

        from qwengpu.compiled_add import (KAPPA, build, compare_token, load_a,
                                          mask_token)

        circuit = build()
        for a, threshold in ((tuple(range(8)), 4),
                             ((255, 0, 128, 127, 0, 1, 200, 199), 128)):
            with self.subTest(a=a, threshold=threshold):
                tokens = [load_a(lane, value) for lane, value in enumerate(a)]
                position = len(tokens)
                tokens.append(compare_token(threshold))
                expected_mask = sum(1 << lane for lane, value in enumerate(a)
                                    if value < threshold)
                residual = evaluate(circuit, tokens, check=True)
                predicted = 0
                for bit, expr in enumerate(circuit.code):
                    value = residual[position, next(iter(expr.terms))]
                    self.assertEqual(abs(value), KAPPA)
                    predicted |= int(value > 0) << bit
                self.assertEqual(predicted, mask_token(expected_mask))


if __name__ == "__main__":
    unittest.main()
