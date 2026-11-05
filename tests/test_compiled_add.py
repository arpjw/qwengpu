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
                    tokens.extend((read_lane(lane), out_token(lane, value)))
                residual = evaluate(circuit, tokens, check=True)
                for lane, position in enumerate(reads):
                    predicted = 0
                    for bit, expr in enumerate(circuit.code):
                        value = residual[position, next(iter(expr.terms))]
                        self.assertEqual(abs(value), KAPPA)
                        predicted |= int(value > 0) << bit
                    self.assertEqual(predicted, out_token(lane, expected[lane]))

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

    def test_result_tokens_write_back_for_second_instruction(self) -> None:
        from chipformer.q35.ir import evaluate

        from qwengpu.compiled_add import (EXEC_ADD, KAPPA, build, load_a, load_b,
                                          mask_token, out_token, read_lane)

        circuit = build()
        a = tuple(range(8))
        b = (17,) * 8
        tokens = [token for lane in range(8)
                  for token in (load_a(lane, a[lane]), load_b(lane, b[lane]))]
        tokens.append(mask_token(0b10101010))
        positions = []
        expected = []
        for _ in range(2):
            tokens.append(EXEC_ADD)
            a = tuple((a[lane] + b[lane]) & 255 if lane % 2 else a[lane]
                      for lane in range(8))
            for lane, value in enumerate(a):
                positions.append(len(tokens))
                expected.append(out_token(lane, value))
                tokens.extend((read_lane(lane), out_token(lane, value)))
        residual = evaluate(circuit, tokens, check=True)
        for position, token in zip(positions, expected, strict=True):
            predicted = 0
            for bit, expr in enumerate(circuit.code):
                value = residual[position, next(iter(expr.terms))]
                self.assertEqual(abs(value), KAPPA)
                predicted |= int(value > 0) << bit
            self.assertEqual(predicted, token)

    def test_one_level_divergence_and_reconvergence(self) -> None:
        from chipformer.q35.ir import evaluate

        from qwengpu.compiled_add import (ELSE_TOKEN, ENDIF_TOKEN, EXEC_ADD,
                                          KAPPA, build, compare_token, load_a,
                                          load_b, mask_token, out_token,
                                          read_lane)

        circuit = build()
        tokens = [token for lane in range(8)
                  for token in (load_a(lane, lane), load_b(lane, 10))]
        tokens.append(mask_token(255))
        predicted_positions = []

        def expect(token: int, next_token: int) -> None:
            predicted_positions.append((len(tokens), next_token))
            tokens.extend((token, next_token))

        def add_and_read(values: tuple[int, ...]) -> None:
            tokens.append(EXEC_ADD)
            for lane, value in enumerate(values):
                expect(read_lane(lane), out_token(lane, value))

        expect(compare_token(4), mask_token(0b00001111))
        add_and_read((10, 11, 12, 13, 4, 5, 6, 7))
        expect(ELSE_TOKEN, mask_token(0b11110000))
        tokens.extend(load_b(lane, 20) for lane in range(8))
        add_and_read((10, 11, 12, 13, 24, 25, 26, 27))
        expect(ENDIF_TOKEN, mask_token(255))
        tokens.extend(load_b(lane, 1) for lane in range(8))
        add_and_read((11, 12, 13, 14, 25, 26, 27, 28))

        residual = evaluate(circuit, tokens, check=True)
        for position, expected in predicted_positions:
            predicted = 0
            for bit, expr in enumerate(circuit.code):
                value = residual[position, next(iter(expr.terms))]
                self.assertEqual(abs(value), KAPPA)
                predicted |= int(value > 0) << bit
            self.assertEqual(predicted, expected)


if __name__ == "__main__":
    unittest.main()
