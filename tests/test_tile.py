"""Exercise all scheduler status pairs and both priorities in the compiler."""

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("chipformer"), "Qwenputer backend unavailable"
)
class TileSchedulerTests(unittest.TestCase):
    def test_every_status_pair_and_priority(self):
        from chipformer.q35.ir import evaluate
        from qwengpu.compiled_tile import (
            DEADLOCK,
            DONE,
            READY0,
            READY1,
            SCHEDULE,
            TICKET,
            build,
            control_token,
        )

        circuit = build()
        # Entries are indexed by ready, halted, waiting for each warp.
        choices = (
            ("prefer", 0, 0),
            (1, DONE, DEADLOCK),
            (1, DEADLOCK, "release"),
        )
        for status0, row in enumerate(choices):
            for status1, choice in enumerate(row):
                for previous in (0, 1):
                    with self.subTest(
                        status0=status0, status1=status1, previous=previous
                    ):
                        controls = [
                            control_token(0, 3, status0),
                            control_token(1, 7, status1),
                        ]
                        if choice == "prefer":
                            ticket = TICKET + 1 - previous
                        elif choice == "release":
                            ticket = TICKET + 3 - previous
                            controls_after = [control_token(0, 3), control_token(1, 7)]
                        else:
                            ticket = TICKET + choice if choice in (0, 1) else choice
                        if choice != "release":
                            controls_after = controls
                        output = evaluate(
                            circuit,
                            [*controls, TICKET + previous, SCHEDULE, READY0, READY1],
                            check=True,
                            kernel=True,
                        )
                        actual = [
                            sum(
                                int(output[position, next(iter(expr.terms))] > 0) << bit
                                for bit, expr in enumerate(circuit.code)
                            )
                            for position in range(3, 6)
                        ]
                        self.assertEqual(actual, [ticket, *controls_after])
