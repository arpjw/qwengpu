"""Differentially check cached audit locations against the pinned old reader."""

from __future__ import annotations

import argparse
import json
import subprocess
import types
from dataclasses import replace
from pathlib import Path

import torch
from chipformer.q35.ir import A_BIT, A_REF, Head

from qwengpu.compiled_tile8 import build
from qwengpu.eval_compiled_rom import head_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="9948a57")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    old = types.ModuleType("previous_audit")
    old.__file__ = str(Path("qwengpu/eval_compiled_rom.py").resolve())
    source = subprocess.check_output(
        ["git", "show", f"{args.ref}:qwengpu/eval_compiled_rom.py"], text=True
    )
    exec(compile(source, args.ref, "exec"), old.__dict__)  # noqa: S102 - pinned local reference reader.
    circuit = build(resident_memory512=True)
    heads = [node for node in circuit.nodes if isinstance(node, Head)]
    ratio = circuit.target.value_heads // circuit.target.key_heads
    layers = max((node.level - 1) // 2 for node in heads) + 2
    states = [torch.zeros(1, circuit.target.value_heads, 128, 6) for _ in range(layers)]
    cache = types.SimpleNamespace(
        layers=[types.SimpleNamespace(recurrent_states=[state]) for state in states]
    )
    requests = []
    for i, node in enumerate(heads):
        layer = (node.level - 1) // 2
        index = sum(
            isinstance(previous, Head) and previous.level == node.level
            for previous in circuit.nodes[: circuit.nodes.index(node)]
        )
        for slot in (0, 1, 63, 127):
            value = (i + slot) % 16
            states[layer][0, index * ratio, slot, :2] = A_REF
            states[layer][0, index * ratio, slot, 2:] = torch.tensor(
                [A_BIT if value >> bit & 1 else 0 for bit in range(4)]
            )
            requests.append((node.name, slot, value))
    for name, slot, expected in requests:
        assert (
            old.head_row(cache, circuit, name, slot, 4)
            == head_row(cache, circuit, name, slot, 4)
            == expected
        )
    appended = replace(
        heads[-1],
        name="audit.cache.append",
        level=max(node.level for node in heads) + 2,
    )
    circuit.nodes.append(appended)
    assert (
        old.head_row(cache, circuit, appended.name, 0, 4)
        is head_row(cache, circuit, appended.name, 0, 4)
        is None
    )
    first = heads[0]
    row = states[(first.level - 1) // 2][0, 0, 0]
    for values in (
        [0, 0, 0, 0, 0, 0],
        [0, 0, A_BIT, 0, 0, 0],
        [0, A_REF, 0, 0, 0, 0],
        [A_REF, A_REF, A_BIT / 2, 0, 0, 0],
    ):
        row[:] = torch.tensor(values)
        outcomes = []
        for reader in (old.head_row, head_row):
            try:
                outcomes.append(reader(cache, circuit, first.name, 0, 4))
            except ValueError as error:
                outcomes.append(str(error))
        assert outcomes[0] == outcomes[1]
    report = {
        "reference": args.ref,
        "heads": len(heads),
        "row_comparisons": len(requests),
        "append_invalidates_cache": True,
        "unwritten_and_invalid_bit_checks_identical": True,
        "weights_allocated": False,
        "all_exact": True,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
