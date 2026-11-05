"""Run three uploaded programs with one stock Qwen3.5 ROM model on MPS."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import transformers

from chipformer.q35.ir import A_BIT, A_REF, Head

from qwengpu.rom_spike import (ADVANCE, PAD, PAYLOAD_BITS, ROM_SLOTS,
                               SAMPLE_PROGRAMS, TICK,
                               build_standalone, instruction_token, load_token)


def head_row(cache, circuit, name: str, slot: int, bits: int) -> int:
    node = next(node for node in circuit.nodes
                if isinstance(node, Head) and node.name == name)
    layer = (node.level - 1) // 2
    index = sum(isinstance(previous, Head) and previous.level == node.level
                for previous in circuit.nodes[:circuit.nodes.index(node)])
    ratio = circuit.target.value_heads // circuit.target.key_heads
    row = cache.layers[layer].recurrent_states[0][0, index * ratio, slot].float()
    if tuple(row[:2].tolist()) != (A_REF, A_REF):
        raise ValueError(f"{name}[{slot}] has wrong recurrent ballast")
    value = 0
    for bit, column in enumerate(row[2:2 + bits]):
        if float(column) not in (0.0, A_BIT):
            raise ValueError(f"{name}[{slot}] has an invalid bit {bit}")
        value |= int(float(column) == A_BIT) << bit
    return value


@torch.no_grad()
def evaluate_case(model, circuit, program: tuple[int, ...], device: str) -> dict:
    start = time.monotonic()
    prefix = [PAD, *(load_token(i, payload)
                     for i, payload in enumerate(program)), 0]
    output = model(input_ids=torch.tensor([prefix], device=device), use_cache=True)
    cache = output.past_key_values
    generated = []
    expected = []

    def feed(token: int) -> int:
        nonlocal cache
        result = model(input_ids=torch.tensor([[token]], device=device),
                       past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        return int(result.logits[0, -1].argmax())

    for pc, payload in enumerate(program):
        token = feed(TICK)
        generated.append(token)
        expected.append(instruction_token(payload))
        feed(token)
        if pc < ROM_SLOTS - 1:
            next_pc = feed(ADVANCE)
            generated.append(next_pc)
            expected.append(pc + 1)
            feed(next_pc)
    observed_rom = [head_row(cache, circuit, "rom", slot, PAYLOAD_BITS)
                    for slot in range(ROM_SLOTS)]
    observed_pc = head_row(cache, circuit, "pc", 0, 4)
    return {"program": program, "generated": generated, "expected": expected,
            "observed_rom": observed_rom, "observed_pc": observed_pc,
            "exact": generated == expected and observed_rom == list(program)
            and observed_pc == ROM_SLOTS - 1,
            "seconds": round(time.monotonic() - start, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                        else "cpu")
    args = parser.parse_args()
    model, circuit = build_standalone()
    model.to(args.device)
    programs = SAMPLE_PROGRAMS
    report = {"device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__, "circuit": circuit.stats(),
              "programs": []}
    for program in programs:
        result = evaluate_case(model, circuit, program, args.device)
        report["programs"].append(result)
        print(json.dumps({"program": program, "exact": result["exact"],
                          "seconds": result["seconds"]}), flush=True)
    report["all_exact"] = all(item["exact"] for item in report["programs"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_exact": report["all_exact"], "out": str(args.out)}), flush=True)
    if not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
