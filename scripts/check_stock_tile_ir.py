"""Replay frozen stock-generation token streams through the compiler IR."""

import argparse
import hashlib
import json
from pathlib import Path

from chipformer.q35.emit import allocate
from chipformer.q35.ir import A_BIT, A_REF, Head, Target, evaluate
from chipformer.q35.merge import pick_hidden
from qwengpu.compiled_tile8 import build
from qwengpu.eval_resident_generation import (
    expected_tokens, launch_prompt, launch_registers, resident_reference,
)
from qwengpu.eval_tile8 import oracle, parse_launch

BASE_HIDDEN = 4096
MERGE_TARGET = Target(16, 32, 12288)


def memory_words(circuit, state, *, global_ram):
    locations = {node.name: i for i, node in enumerate(circuit.nodes)
                 if isinstance(node, Head)}
    words = []
    for word in range(256 if global_ram else 128):
        value = 0
        for nib in range(4):
            mirrors = []
            for lane in range(8):
                name = (f"ram.{lane}.{nib}.{word // 128}" if global_ram else
                        f"shared.{lane}.{nib}")
                row = state[locations[name], "S", 0][word % 128]
                assert tuple(row[:2]) == (A_REF, A_REF), (name, word)
                assert all(bit in (0, A_BIT) for bit in row[2:6]), (name, word)
                mirrors.append(sum(int(bit == A_BIT) << i for i, bit in enumerate(row[2:6])))
            assert mirrors == mirrors[:1] * 8, (word, nib, mirrors)
            value |= mirrors[0] << (4 * nib)
        words.append(value)
    return words

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-tilegrid", action="store_true",
                        help="check the ordinary launch path on the v4 circuit")
    parser.add_argument("fixtures", type=Path, nargs="+")
    args = parser.parse_args()
    circuit = build(MERGE_TARGET, generate=True, generated_tilegrid=args.generated_tilegrid)
    code = [next(iter(value.terms)) for value in circuit.code]
    allocation = allocate(circuit, None, set(code), MERGE_TARGET.intermediate)
    hidden, ballast = pick_hidden(BASE_HIDDEN, max(3269, 1 + allocation.n_slots))
    print(circuit.stats(), allocation.n_slots, (hidden, ballast), flush=True)
    for path in args.fixtures:
        payload = path.read_bytes()
        raw = json.loads(payload)
        case = parse_launch(raw, full_integer=True)
        reference = oracle(case, 256, looping=True)
        if "expected_outcome" in raw:
            assert reference["outcome"] == raw["expected_outcome"], path
        active_warps = 4 if case.get("interleaved") else 2
        if active_warps == 2:
            reference = resident_reference(reference, tiled="workgroups" not in case)
        expected, _, _ = expected_tokens(reference, active_warps, registers=launch_registers(case))
        prompt = launch_prompt(case)
        carry = {}
        residual = evaluate(circuit, prompt + expected[:-1], check=True, carry=carry)
        actual = [
            sum(int(value[dim] > 0) << bit for bit, dim in enumerate(code))
            for value in residual[len(prompt) - 1:]
        ]
        for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
            if got != want:
                raise ValueError(f"{path}: token {i} {got=} {want=} previous={expected[max(0, i-8):i]}")
        actual_words = memory_words(circuit, carry, global_ram="workgroups" in case)
        assert actual_words == reference["shared_words"], path
        if "scoped_words" in case:
            assert memory_words(circuit, carry, global_ram=False) == reference["scoped_words"], path
        print(path, len(reference["trace"]), len(expected), hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
