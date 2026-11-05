"""Check autonomous ROM fetch and SIMT execution in stock Qwen3.5 layers."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import transformers

from chipformer.q35.ir import A_BIT, A_REF, Head

from qwengpu.compiled_rom import (ROM_LOAD_BASE, ROM_SLOTS, SCRATCH_SLOTS,
                                  STATE_BASE, TICK, build_standalone, load_a,
                                  out_token, program_tokens, read_lane,
                                  state_token)
from qwengpu.reference import LANES, WarpState, step
from qwengpu.run_kernel import parse_kernel


def head_row(cache, circuit, name: str, slot: int, bits: int) -> int | None:
    node = next(node for node in circuit.nodes
                if isinstance(node, Head) and node.name == name)
    layer = (node.level - 1) // 2
    index = sum(isinstance(previous, Head) and previous.level == node.level
                for previous in circuit.nodes[:circuit.nodes.index(node)])
    ratio = circuit.target.value_heads // circuit.target.key_heads
    row = cache.layers[layer].recurrent_states[0][0, index * ratio, slot].float()
    if tuple(row[:2].tolist()) == (0.0, 0.0):
        if any(float(value) != 0.0 for value in row[2:2 + bits]):
            raise ValueError(f"unwritten {name}[{slot}] contains data")
        return None
    if tuple(row[:2].tolist()) != (A_REF, A_REF):
        raise ValueError(f"{name}[{slot}] has wrong recurrent ballast")
    value = 0
    for bit, column in enumerate(row[2:2 + bits]):
        if float(column) not in (0.0, A_BIT):
            raise ValueError(f"{name}[{slot}] has an invalid bit {bit}")
        value |= int(float(column) == A_BIT) << bit
    return value


@torch.no_grad()
def evaluate_case(model, circuit, initial, program, device: str) -> dict:
    start = time.monotonic()
    rom_tokens = program_tokens(program)
    prefix = [load_a(lane, value) for lane, value in enumerate(initial)]
    prefix.extend(rom_tokens)
    prefix.append(state_token(0))
    output = model(input_ids=torch.tensor([prefix], device=device), use_cache=True)
    cache = output.past_key_values
    reference = WarpState(lanes=initial)
    generated = []
    expected = []
    tick_count = 0

    def feed(token: int) -> int:
        nonlocal cache
        result = model(input_ids=torch.tensor([[token]], device=device),
                       past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        return int(result.logits[0, -1].argmax())

    for _ in range(ROM_SLOTS + 1):
        if reference.halted:
            break
        reference = step(reference, program)
        state = feed(TICK)
        generated.append(state)
        expected.append(state_token(reference.pc, reference.halted))
        feed(state)
        tick_count += 1
        if not STATE_BASE <= state < TICK:
            break
        if (state - STATE_BASE) & 16:
            break
        for lane, value in enumerate(reference.lanes):
            token = feed(read_lane(lane))
            generated.append(token)
            expected.append(out_token(lane, value))
            feed(token)

    observed_rom = [head_row(cache, circuit, "rom", slot, 11)
                    for slot in range(ROM_SLOTS)]
    expected_rom = [(token - ROM_LOAD_BASE) & ((1 << 11) - 1)
                    for token in rom_tokens]
    observed_pc = head_row(cache, circuit, "pc", 0, 5)
    expected_pc = reference.pc + (16 if reference.halted else 0)
    observed_lanes = [head_row(cache, circuit, f"a.{lane}", 0, 8)
                      for lane in range(LANES)]
    written_slots = {ins.arg // LANES for ins in program if ins.op == "STORE"}
    observed_scratch = [[head_row(cache, circuit, f"scratch.{lane}", slot, 8)
                         for lane in range(LANES)] for slot in range(SCRATCH_SLOTS)]
    expected_scratch = [
        [reference.scratch[slot * LANES + lane] if slot in written_slots else None
         for lane in range(LANES)] for slot in range(SCRATCH_SLOTS)]
    exact = (reference.halted and generated == expected and
             observed_rom == expected_rom and observed_pc == expected_pc and
             observed_lanes == list(reference.lanes) and
             observed_scratch == expected_scratch)
    return {"initial": initial,
            "program": [{"op": ins.op, "arg": ins.arg} for ins in program],
            "generated": generated, "expected": expected,
            "observed_rom": observed_rom, "expected_rom": expected_rom,
            "observed_pc": observed_pc, "expected_pc": expected_pc,
            "observed_lanes": observed_lanes,
            "expected_lanes": reference.lanes,
            "observed_scratch": observed_scratch,
            "expected_scratch": expected_scratch,
            "tick_count": tick_count, "exact": exact,
            "seconds": round(time.monotonic() - start, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "eval/frozen/rom_gate_v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                        else "cpu")
    args = parser.parse_args()
    paths = sorted(args.cases_dir.glob("*.json"))
    if len(paths) != 4:
        raise ValueError("expected exactly four frozen ROM-gate kernels")
    model, circuit = build_standalone()
    model.to(args.device).eval()
    report = {"device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__, "circuit": circuit.stats(),
              "cases": []}
    for path in paths:
        payload = path.read_bytes()
        initial, program = parse_kernel(json.loads(payload))
        result = evaluate_case(model, circuit, initial, program, args.device)
        result["fixture"] = path.name
        result["fixture_sha256"] = hashlib.sha256(payload).hexdigest()
        report["cases"].append(result)
        print(json.dumps({"fixture": path.name, "exact": result["exact"],
                          "ticks": result["tick_count"],
                          "seconds": result["seconds"]}), flush=True)
    report["all_exact"] = all(case["exact"] for case in report["cases"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_exact": report["all_exact"], "out": str(args.out)}), flush=True)
    if not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
