"""Replay one stock-generation launch of eight resident matrix panels."""

import argparse
import hashlib
import json
from pathlib import Path

from chipformer.q35.emit import allocate
from chipformer.q35.ir import A_BIT, A_REF, Head, Target, evaluate
from chipformer.q35.merge import pick_hidden

from qwengpu import resident_generation as gen
from qwengpu.compiled_tile8 import ALT_OPERAND, BOTTOM_OPERANDS, GRID, OUTPUT_BANKS, RIGHT_OPERANDS, build
from qwengpu.eval_resident_generation import expected_tokens, launch_prompt, resident_reference
from qwengpu.eval_tile8 import _digits, _words, nibbles, oracle, parse_launch
from qwengpu.run_matmul16 import TILE, _tile, parse_inputs


def panel_memory_plan(a, b, c):
    """Return preloaded tiles in the circuit's physical slot order."""
    yield ALT_OPERAND, _tile(a, 0, TILE) + _tile(b, TILE, 0), 8
    for base, k in zip(BOTTOM_OPERANDS, (0, TILE), strict=True):
        yield base, _tile(a, TILE, k), 8
    for base, k in zip(RIGHT_OPERANDS, (0, TILE), strict=True):
        yield base, _tile(b, k, TILE), 8
    for base, (row, col) in zip(
        OUTPUT_BANKS[1:], ((0, TILE), (TILE, 0), (TILE, TILE)), strict=True,
    ):
        yield base, _tile(c, row, col), 16


def matrix_protocol(a, b, c):
    first = parse_launch(
        {"a": _tile(a, 0, 0), "b": _tile(b, 0, 0), "c": _tile(c, 0, 0)},
        full_integer=True,
    )
    prompt = launch_prompt(first)
    prompt[prompt.index(GRID + 3)] |= 16 | 32 | 64
    for base, values, width in panel_memory_plan(a, b, c):
        prompt[-1:-1] = nibbles(base, _digits(values, width))

    expected, final = [], []
    for group in range(4):
        row, col = TILE * (group // 2), TILE * (group % 2)
        output = _tile(c, row, col)
        for k in (0, TILE):
            case = parse_launch(
                {"a": _tile(a, row, k), "b": _tile(b, k, col), "c": output},
                full_integer=True,
            )
            result = oracle(case, 256, looping=True)
            assert result["outcome"] == "halted"
            tokens, _, _ = expected_tokens(resident_reference(result, tiled=True), 2)
            panel = 2 * group + k // TILE
            if panel:
                expected.extend((gen.PANEL + panel, *gen.RESET, gen.SCHEDULE))
            expected.extend(tokens if panel == 7 else tokens[:-1])
            output = result["shared_words"][64:]
        final.append(output)
    return prompt, expected, final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--allocate-only", action="store_true")
    args = parser.parse_args()
    payload = args.fixture.read_bytes()
    matrices = parse_inputs(json.loads(payload))
    print(f"{args.fixture.name} SHA-256 {hashlib.sha256(payload).hexdigest()}", flush=True)
    circuit = build(Target(16, 32, 12288), generate=True, generated_tilegrid=True)
    code = [next(iter(value.terms)) for value in circuit.code]
    allocation = allocate(circuit, None, set(code), circuit.target.intermediate)
    print(circuit.stats(), allocation.n_slots, pick_hidden(4096, 1 + allocation.n_slots), flush=True)
    if args.allocate_only:
        return
    prompt, expected, final = matrix_protocol(*matrices)
    assert [token for token in expected if gen.PANEL <= token < gen.PANEL + 8] == [
        gen.PANEL + panel for panel in range(1, 8)
    ]
    assert sum(gen.TICKET <= token < gen.TICKET + 4 for token in expected) == 928
    stream = prompt + expected[:-1]
    state = {}
    for start in range(0, len(stream), 256):
        residual = evaluate(circuit, stream[start:start + 256], check=True, carry=state)
        for index, row in enumerate(residual, start):
            if index >= len(prompt) - 1:
                actual = sum(int(row[dim] > 0) << bit for bit, dim in enumerate(code))
                wanted = expected[index - len(prompt) + 1]
                assert actual == wanted, (index, stream[index], actual, wanted)
                if gen.PANEL < wanted < gen.PANEL + 8:
                    print(f"Matched transition to panel {wanted - gen.PANEL}", flush=True)
    locations = {node.name: i for i, node in enumerate(circuit.nodes) if isinstance(node, Head)}
    for group, target in enumerate(final):
        digits = []
        for row in range(TILE):
            for nib in range(4 * TILE):
                value = state[locations[f"C.{nib}"], "S", 0][TILE * group + row]
                assert tuple(value[:2]) == (A_REF, A_REF)
                assert all(bit in (0, A_BIT) for bit in value[2:6])
                digits.append(sum(int(bit == A_BIT) << i for i, bit in enumerate(value[2:6])))
        assert _words(digits) == target, group
    print(f"{len(expected)} exact generated tokens and four exact output banks")


if __name__ == "__main__":
    main()
