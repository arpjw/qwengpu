"""Compare generated compute programs with independent scalar results."""

import random
import unittest

from qwengpu.eval_tile8 import oracle, parse_launch
from qwengpu.run_compute import compile_request, parse_request, words16, words32


def interpret(request):
    programs = compile_request(parse_request(request))
    words = programs[0]["global_words"]
    for program in programs:
        case = parse_launch({**program, "global_words": words}, resident_compute=True)
        result = oracle(case, 256, looping=True)
        assert result["outcome"] == "halted"
        # Matrix output addressing must not change the scheduler's tile row.
        assert all(
            (int(step["snapshot"], 16) >> 129) & 7 == step["warp"]
            for step in result["trace"]
        )
        words = result["shared_words"]
    return words, programs[0]["global_words"]


class ComputeTests(unittest.TestCase):
    def test_vector_and_reduction_boundaries(self):
        rng = random.Random(81827)
        for count in range(65):
            a = [rng.getrandbits(32) for _ in range(count)]
            b = [rng.getrandbits(32) for _ in range(count)]
            if count <= 32:
                words, initial = interpret({"kernel": "vector_add", "a": a, "b": b})
                expected = initial[:]
                expected[: 2 * count] = words16(
                    [(x + y) & 0xFFFFFFFF for x, y in zip(a, b)]
                )
                self.assertEqual(words, expected)
            with self.subTest(count=count, kernel="reduce"):
                words, initial = interpret({"kernel": "reduce", "values": a})
                expected = initial[:]
                for start in range(0, count, 8):
                    expected[2 * start : 2 * start + 2] = words16(
                        [sum(a[start : start + 8]) & 0xFFFFFFFF]
                    )
                expected[:2] = words16([sum(a) & 0xFFFFFFFF])
                self.assertEqual(words, expected)

    def test_matrix_dimensions_and_overflow(self):
        rng = random.Random(8218)
        for m in (1, 3, 5, 8):
            for k in (1, 3, 4):
                for n in (1, 2, 3, 4):
                    a = [[rng.randrange(256) for _ in range(k)] for _ in range(m)]
                    b = [[rng.randrange(256) for _ in range(n)] for _ in range(k)]
                    c = [[0xFFFFFFFE for _ in range(n)] for _ in range(m)]
                    with self.subTest(m=m, k=k, n=n):
                        words, initial = interpret(
                            {"kernel": "matmul", "a": a, "b": b, "c": c}
                        )
                        expected = initial[:]
                        scalar = [
                            (c[i][j] + sum(a[i][t] * b[t][j] for t in range(k)))
                            & 0xFFFFFFFF
                            for i in range(m)
                            for j in range(n)
                        ]
                        expected[64 : 64 + 2 * m * n] = words16(scalar)
                        self.assertEqual(words, expected)
                        self.assertEqual(words32(words[64 : 64 + 2 * m * n]), scalar)

    def test_invalid_requests(self):
        for request in (
            {"kernel": "vector_add", "a": [1], "b": []},
            {"kernel": "reduce", "values": [0] * 65},
            {"kernel": "matmul", "a": [[1]], "b": [[256]]},
            {"kernel": "matmul", "a": [[1]], "b": [[1] * 5]},
            {"kernel": "matmul", "a": [[1, 2]], "b": [[1]]},
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                parse_request(request)
