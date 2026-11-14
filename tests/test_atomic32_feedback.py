"""Replay a generated atomic value stream into recurrent RAM heads."""

import unittest

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target, ev, evaluate
from chipformer.q35.prims import Circ

from qwengpu.compiled_atomic32 import (
    atomic_add32_results,
    atomic_nibble_step,
    generated_memory_gate,
)
from qwengpu.memory_geometry import WIDE_MEMORY
from qwengpu.resident_scheduler import RESIDENT_ENABLE
from qwengpu import resident_generation as gen

TICK = gen.TICKET
LOW_COMMAND = gen.COMMIT
HIGH_COMMAND = gen.ATOMIC_HIGH_COMMIT
LAUNCH_START = RESIDENT_ENABLE
SCHEDULE = gen.SCHEDULE


def word_tags(base: int, address: int, word: int) -> list[int]:
    return [
        base + 64 * address + 16 * nibble + (word >> (4 * nibble) & 15)
        for nibble in range(4)
    ]


def number(bits, residual, position):
    return sum(
        int(ev(bit, residual)[position]) << shift
        for shift, bit in enumerate(bits)
    )


class Atomic32FeedbackTests(unittest.TestCase):
    def test_generation_phase_resets_before_a_dependent_launch_upload(self):
        c = Circ(Target())
        start = c.token_feature(
            "feedback.phase.start", lambda token: int(token == LAUNCH_START), full=True
        )
        schedule = c.token_feature(
            "feedback.phase.schedule", lambda token: int(token == SCHEDULE), full=True
        )
        memory = c.token_feature(
            "feedback.phase.memory",
            lambda token: int(WIDE_MEMORY.base <= token < WIDE_MEMORY.vocab),
            full=True,
        )
        gated = generated_memory_gate(c, memory, start, schedule)
        tag = word_tags(WIDE_MEMORY.base, 0, 1)[0]
        tokens = [
            LAUNCH_START, tag, SCHEDULE, tag,
            LAUNCH_START, tag, SCHEDULE, tag,
        ]
        residual = evaluate(c, tokens, check=True)
        self.assertEqual(
            [int(ev(gated, residual)[position]) for position in range(len(tokens))],
            [0, 0, 0, 1, 0, 0, 0, 1],
        )

    def test_generated_tokens_update_ram_before_the_next_instruction(self):
        c = Circ(Target())
        tick = c.token_feature("feedback.tick", lambda token: int(token == TICK), full=True)
        low_command = c.token_feature(
            "feedback.low_command", lambda token: int(token == LOW_COMMAND), full=True
        )
        high_command = c.token_feature(
            "feedback.high_command", lambda token: int(token == HIGH_COMMAND), full=True
        )
        command = low_command + high_command
        launch_start = c.token_feature(
            "feedback.launch_start", lambda token: int(token == LAUNCH_START), full=True
        )
        schedule = c.token_feature(
            "feedback.schedule", lambda token: int(token == SCHEDULE), full=True
        )
        raw_memory_token = c.token_feature(
            "feedback.memory_token",
            lambda token: int(WIDE_MEMORY.base <= token < WIDE_MEMORY.vocab),
            full=True,
        )
        memory_token = generated_memory_gate(
            c, raw_memory_token, launch_start, schedule
        )
        raw = [
            c.token_feature(
                f"feedback.raw.{bit}",
                lambda token, bit=bit: token >> bit & 1,
                full=True,
            )
            for bit in range(6)
        ]
        memory = []
        for address in (0, 1):
            for nibble in range(4):
                load = c.token_feature(
                    f"feedback.load.{address}.{nibble}",
                    lambda token, address=address, nibble=nibble: int(
                        WIDE_MEMORY.base <= token < WIDE_MEMORY.vocab
                        and (token & 0x3FFF) // 64 == address
                        and (token >> 4) & 3 == nibble
                    ),
                    full=True,
                )
                memory.extend(
                    c.slot_head(
                        f"feedback.ram.{address}.{nibble}",
                        {0: load},
                        {0: tick},
                        {2 + bit: value for bit, value in enumerate(raw[:4])},
                        load,
                    )
                )
        old, final, grants = atomic_add32_results(
            c,
            [[Expr()] * 7],
            [[c.one, c.one, *([Expr()] * 30)]],
            [memory],
            [c.one],
        )
        self.assertEqual(len(grants), 1)
        staged = c.slot_head(
            "feedback.staged_result",
            {0: tick},
            {0: command},
            {2 + bit: value for bit, value in enumerate(final[0])},
            tick,
        )
        selected_word = c.select_bits(
            [([low_command], staged[:16]), ([high_command], staged[16:])],
            16,
            "feedback.selected_word",
        )
        selected_address = [high_command, *([Expr()] * 7)]
        current = c.slot_head(
            "feedback.current_limb",
            {0: command},
            {0: command + memory_token},
            {2 + bit: value
             for bit, value in enumerate([*selected_address, *selected_word])},
            command,
        )
        emit, output, done = atomic_nibble_step(
            c, command, memory_token, raw[4:6], current[:8], current[8:]
        )

        low_tags = word_tags(WIDE_MEMORY.base, 0, 1)
        high_tags = word_tags(WIDE_MEMORY.base, 1, 0)
        inputs = [
            LAUNCH_START,
            *word_tags(WIDE_MEMORY.base, 0, 0xFFFE),
            *word_tags(WIDE_MEMORY.base, 1, 0xFFFF),
            SCHEDULE,
            TICK,
            LOW_COMMAND,
            *low_tags,
            HIGH_COMMAND,
            *high_tags,
            TICK,
        ]
        residual = evaluate(c, inputs, check=True)
        first_tick, second_tick = 10, len(inputs) - 1
        self.assertEqual(number(old[0], residual, first_tick), 0xFFFFFFFE)
        self.assertEqual(number(final[0], residual, first_tick), 1)
        self.assertEqual(number(old[0], residual, second_tick), 1)
        self.assertEqual(number(final[0], residual, second_tick), 4)
        emissions = [
            number(output, residual, position)
            for position in range(len(inputs))
            if int(ev(emit, residual)[position])
        ]
        self.assertEqual(emissions, [*low_tags, *high_tags])
        completed = [
            position for position in range(len(inputs))
            if int(ev(done, residual)[position])
        ]
        self.assertEqual(completed, [15, 20])
