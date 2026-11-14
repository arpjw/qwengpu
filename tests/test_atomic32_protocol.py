"""Check old and expanded model-owned store token selection."""

import unittest

from chipformer.dsl.graph import Expr, as_expr
from chipformer.q35.ir import Target, ev, evaluate
from chipformer.q35.prims import Circ

from qwengpu.model_stores import ATOMIC_HIGH_COMMIT_BASE, COMMIT_BASE, COMMIT_COUNT
from qwengpu import resident_generation as gen


def selected_commit_tokens(c, grants, token_ids):
    request = c.token_feature("request", lambda token: int(token == 1), full=True)
    inputs = [
        c.token_feature(
            f"commit.{index}",
            lambda token, expected=expected: int(token == expected),
            full=True,
        )
        for index, expected in enumerate(token_ids)
    ]
    raw = [
        c.token_feature(
            f"raw.{bit}", lambda token, bit=bit: token >> bit & 1, full=True
        )
        for bit in range(11)
    ]
    return list(
        gen.generated_writeback(
            c,
            request,
            Expr(),
            inputs,
            raw,
            [Expr(), Expr()],
            [],
            [],
            [c.one if index in grants else Expr() for index in range(len(token_ids))],
            token_ids if len(token_ids) > COMMIT_COUNT else None,
        )
    )


class Atomic32ProtocolTests(unittest.TestCase):
    def test_extra_commit_tokens_do_not_overlap_controls(self):
        legacy = set(range(COMMIT_BASE, COMMIT_BASE + COMMIT_COUNT))
        legacy_high = set(range(ATOMIC_HIGH_COMMIT_BASE, ATOMIC_HIGH_COMMIT_BASE + 8))
        generated = set(range(gen.COMMIT, gen.COMMIT + COMMIT_COUNT))
        generated_high = set(range(gen.ATOMIC_HIGH_COMMIT, gen.ATOMIC_HIGH_COMMIT + 8))
        controls = (
            {gen.SCHEDULE, *gen.RESET}
            | set(range(gen.TICKET, gen.TICKET + 4))
            | set(range(gen.GROUP, gen.GROUP + 4))
            | set(range(gen.ROW, gen.ROW + 4))
            | set(range(gen.PANEL, gen.PANEL + 8))
        )
        self.assertEqual(len(legacy | legacy_high | generated | generated_high), 32)
        self.assertFalse((legacy | legacy_high | generated | generated_high) & controls)

    def check_sequence(self, token_ids, granted):
        c = Circ(Target())
        outputs = selected_commit_tokens(c, granted, token_ids)
        expected = [token_ids[index] for index in granted] + [gen.SCHEDULE]
        inputs = [1, *expected[:-1]]
        residual = evaluate(c, inputs, check=True)
        observed = []
        for position in range(len(inputs)):
            selected = []
            for gates, bits in outputs:
                if all(int(ev(gate, residual)[position]) for gate in gates):
                    selected.append(
                        sum(
                            int(ev(as_expr(bit), residual)[position]) << offset
                            for offset, bit in enumerate(bits)
                        )
                    )
            self.assertEqual(len(selected), 1)
            observed.extend(selected)
        self.assertEqual(observed, expected)

    def test_existing_eight_lane_commit_sequence(self):
        self.check_sequence([gen.COMMIT + lane for lane in range(8)], [0, 7])

    def test_new_sixteen_limb_commit_sequence(self):
        token_ids = [
            *[gen.COMMIT + lane for lane in range(8)],
            *[gen.ATOMIC_HIGH_COMMIT + lane for lane in range(8)],
        ]
        self.check_sequence(token_ids, [0, 7, 8, 15])
