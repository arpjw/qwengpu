"""Check the frozen cross-workgroup dependency and token range separation."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from chipformer.q35.ir import A_BIT, A_REF

from qwengpu import resident_generation as gen
from qwengpu.compiled_tile8 import C_MEMORY, WIDE_WARP_TAGS, rom_address
from qwengpu.eval_compiled_rom import head_rows
from qwengpu.eval_resident_generation import (
    cache_words, expected_tokens, launch_prompt, launch_registers, resident_reference,
)
from qwengpu.eval_tile8 import oracle, parse_launch
from qwengpu.resident_scheduler import resident_control


class ResidentGenerationTests(unittest.TestCase):
    def test_batched_ram_read_preserves_banks_nibbles_and_mirrors(self):
        fixture = SimpleNamespace(
            nodes=[], target=SimpleNamespace(value_heads=2, key_heads=1),
        )
        names = [
            f"{port}.{lane}.{nib}{suffix}"
            for port, suffix in (("shared", ""), ("ram", ".0"), ("ram", ".1"))
            for lane in range(8)
            for nib in range(4)
        ]
        fixture._qwengpu_head_location_cache = (0, {name: (i, 0) for i, name in enumerate(names)})
        words = [0x8000 + 113 * i for i in range(256)]
        layers = []
        for i, _ in enumerate(names):
            bank = int(i >= 64)
            nib = i % 4
            digits = [(word >> (4 * nib)) & 15 for word in words[128 * bank:128 * (bank + 1)]]
            state = torch.tensor(
                [[A_REF, A_REF, *(A_BIT if digit >> bit & 1 else 0 for bit in range(4))]
                 for digit in digits],
            ).reshape(1, 1, 128, 6)
            layers.append(SimpleNamespace(recurrent_states=(state,)))
        cache = SimpleNamespace(layers=layers)
        self.assertEqual(cache_words(cache, fixture), (words, True))
        self.assertEqual(cache_words(cache, fixture, scoped=True), (words[:128], True))
        layers[0].recurrent_states[0][0, 0, 3] = 0
        self.assertFalse(cache_words(cache, fixture, scoped=True)[1])
        layers[0].recurrent_states[0][0, 0, 3, 2] = A_BIT
        with self.assertRaisesRegex(ValueError, r"unwritten shared\.0\.0\[3\]"):
            head_rows(cache, fixture, "shared.0.0", 1, 4, 4)

    def test_tile_relaunch_uploads_operands_without_output_words(self):
        case = parse_launch({"c": [123] * 64}, full_integer=True)
        initial = launch_prompt(case)
        continued = launch_prompt(case, retain_tile_output=True)
        self.assertEqual(
            continued, [token for token in initial if not C_MEMORY <= token < C_MEMORY + 4096],
        )
        self.assertEqual(len(initial) - len(continued), 256)

    def test_tile_row_transition_preserves_seeded_registers(self):
        case = parse_launch({
            "programs": [[{"op": "ADD16I", "arg": 1}, {"op": "HALT"}],
                         [{"op": "HALT"}]],
            "first_row_pair": 1, "row_pairs": 2,
            "initial_registers": [[0x12340001] * 8, [0x45670001] * 8],
        }, full_integer=True)
        reference = resident_reference(oracle(case, 256, looping=True), tiled=True)
        tokens, controls, regs = expected_tokens(
            reference, active_warps=2, registers=launch_registers(case),
        )
        self.assertIn(gen.ROW + 1, launch_prompt(case))
        row = tokens.index(gen.ROW + 2)
        self.assertEqual(tokens[row:row + 3], [gen.ROW + 2, gen.SCHEDULE, gen.TICKET + 1])
        self.assertTrue(set(tokens).isdisjoint(gen.RESET))
        self.assertEqual(regs[0][:4], [3, 0, 0, 0])
        self.assertEqual(regs[0][-32:-28], [4, 3, 2, 1])
        self.assertEqual(controls[2:], [resident_control(w, status=1) for w in (2, 3)])

    def test_four_sequential_groups_reset_both_active_warps(self):
        path = Path(__file__).resolve().parents[1] / "eval/frozen/scoped_memory_v1/isolation.json"
        case = parse_launch(json.loads(path.read_text()), full_integer=True)
        reference = resident_reference(oracle(case, 256, looping=True))
        tokens, controls, _ = expected_tokens(reference, active_warps=2)
        self.assertIn(gen.GROUP, launch_prompt(case))
        transitions = [i for i, token in enumerate(tokens) if gen.GROUP <= token < gen.GROUP + 4]
        self.assertEqual([tokens[i] for i in transitions], [gen.GROUP + i for i in (1, 2, 3)])
        for index in transitions:
            self.assertEqual(tokens[index + 1:index + 5],
                             [*gen.RESET, gen.SCHEDULE, gen.TICKET])
        self.assertEqual(controls[2:], [resident_control(w, status=1) for w in (2, 3)])

    def test_single_workgroup_leaves_two_warps_halted(self):
        root = Path(__file__).resolve().parents[1] / "eval/frozen/full_integer_v1"
        raw = json.loads((root / "indexed_words.json").read_text())
        case = parse_launch(raw, resident_globalgrid=True, full_integer=True)
        prompt = launch_prompt(case)
        inactive = [resident_control(warp, status=1) for warp in (2, 3)]
        self.assertEqual(prompt[3:5], inactive)
        reference = resident_reference(oracle(case, 256, looping=True))
        tokens, controls, _ = expected_tokens(reference, active_warps=2)
        self.assertEqual(controls[2:], inactive)
        self.assertIn(gen.STEP_CONTROL + 16, tokens)
        self.assertFalse(any(gen.TICKET + 2 <= token < gen.TICKET + 4 for token in tokens))

    def test_cross_group_high_limb(self):
        fixture = Path(__file__).resolve().parents[1] / (
            "eval/frozen/resident_interleaved_v1/cross_group_high_limb.json"
        )
        case = parse_launch(json.loads(fixture.read_text()), resident_globalgrid=True,
                            resident_compute=True, full_integer=True)
        reference = oracle(case, 256, looping=True)
        self.assertEqual(reference["shared_words"][:4], [71, 1, 72, 1])
        self.assertEqual(reference["warp_registers"][2], [0x10048] * 8)
        tokens, _, _ = expected_tokens(reference)
        prompt = launch_prompt(case)
        self.assertEqual(set(tokens) & set(prompt), {gen.SCHEDULE})
        high = gen.REGISTER + 2 * gen.REGISTER_STRIDE + 16 * (len(WIDE_WARP_TAGS) - 32)
        self.assertGreater(sum(high <= token < high + 512 for token in tokens), 0)
        for token in set(tokens):
            self.assertIsNone(rom_address(token, resident_rom32=True, resident_alu32=True,
                                          resident_scoped=True, full_integer=True))
