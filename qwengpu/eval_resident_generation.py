"""Run the interleaved integer GPU through stock generation and audit its cache."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from chipformer.q35.ir import Target
from transformers import AutoModelForCausalLM

from qwengpu import resident_generation as gen
from qwengpu.alu32 import RUNTIME_EXTENT_BASE
from qwengpu.cache_register_edge import register_head_names
from qwengpu.compiled_tile8 import (
    DEADLOCK, DONE, GLOBAL_GRID, GRID, MASK_REGISTER, ROM_BANK, ROW_PAIRS, WARP_BANKS,
    WARP_DIRTY_OFFSET, WIDE_WARP_TAGS, build, build_standalone, program_tokens,
)
from qwengpu.eval_compiled_rom import head_row
from qwengpu.eval_tile8 import _digits, _words, nibbles, oracle, parse_launch
from qwengpu.memory_geometry import WIDE_MEMORY
from qwengpu.resident_scheduler import EXTRA_WARP_BANKS, RESIDENT_ENABLE, RESIDENT_TICKET, resident_control
from qwengpu.scoped_memory import SCOPED_BASE


def launch_prompt(case):
    if not case.get("interleaved"):
        raise ValueError("stock generation requires two interleaved workgroups")
    tokens = [RESIDENT_ENABLE, *(resident_control(warp) for warp in range(4)),
              RESIDENT_TICKET + 3, GRID + GLOBAL_GRID + ROW_PAIRS]
    tokens.extend(nibbles(WIDE_MEMORY.base, _digits(case["global_words"], 16)))
    if "scoped_words" in case:
        tokens.extend(nibbles(SCOPED_BASE, _digits(case["scoped_words"], 16)))
    if "extent" in case:
        tokens.append(RUNTIME_EXTENT_BASE + case["extent"])
    for warp, program in enumerate(case["programs"]):
        tokens.append(ROM_BANK + warp)
        tokens.extend(program_tokens(program, resident_rom32=True, resident_alu32=True,
                                     resident_scoped="scoped_words" in case, full_integer=True))
    for bank in (*WARP_BANKS, *EXTRA_WARP_BANKS):
        tokens.extend(nibbles(bank, initial_registers()))
    return [*tokens, gen.SCHEDULE]


def initial_registers():
    return [15 if token in (MASK_REGISTER, MASK_REGISTER + 16) else 0
            for token in WIDE_WARP_TAGS]


def expected_tokens(reference):
    """Encode the independent interpreter's trace for comparison after generation."""
    records = []
    controls = [resident_control(warp) for warp in range(4)]
    registers = [initial_registers() for _ in range(4)]
    for index, schedule in enumerate(reference["schedules"]):
        if controls != schedule["controls"]:
            records.extend(gen.RELEASE_CONTROL + (token - resident_control(0))
                           for token in schedule["controls"])
        controls = schedule["controls"][:]
        ticket = schedule["ticket"]
        if ticket in (DONE, DEADLOCK):
            records.append(ticket)
            break
        warp = ticket - RESIDENT_TICKET
        records.append(gen.TICKET + warp)
        if index == len(reference["trace"]):
            break
        trace = reference["trace"][index]
        controls[warp] = trace["control"]
        records.append(gen.STEP_CONTROL + (trace["control"] - resident_control(0)))
        packed = int(trace["snapshot"], 16)
        digits = [token & 15 for field in ("registers", "indices", "mask", "branch", "loop")
                  for token in trace[field]]
        digits.extend((packed >> (WARP_DIRTY_OFFSET + 4 * nib)) & 15 for nib in range(32))
        for nib, (before, after) in enumerate(zip(registers[warp], digits, strict=True)):
            if before != after:
                records.append(gen.REGISTER + warp * gen.REGISTER_STRIDE + nib * 16 + after)
        registers[warp] = digits
        records.extend(gen.COMMIT + lane for lane in range(8) if trace["patch"][4 * lane])
        records.append(gen.SCHEDULE)
    return records, controls, registers


def cache_words(cache, circuit, *, scoped=False):
    words = 128 if scoped else 256
    result = []
    mirrored = True
    for word in range(words):
        digits = []
        for nib in range(4):
            rows = ([head_row(cache, circuit, f"shared.{lane}.{nib}", word, 4)
                     for lane in range(8)] if scoped else
                    [head_row(cache, circuit, f"ram.{lane}.{nib}.{word // 128}", word % 128, 4)
                     for lane in range(8)])
            mirrored &= rows[0] is not None and all(value == rows[0] for value in rows)
            digits.append(rows[0] or 0)
        result.extend(digits)
    return _words(result), mirrored


@torch.no_grad()
def evaluate(model, circuit, case, device, offset=0, *, max_new_tokens=32768):
    start = time.monotonic()
    prompt = launch_prompt(case)
    inputs = torch.tensor([[token + offset for token in prompt]], device=device)
    output = model.generate(
        input_ids=inputs, attention_mask=torch.ones_like(inputs),
        do_sample=False, eos_token_id=[offset + DONE, offset + DEADLOCK],
        max_new_tokens=max_new_tokens, use_cache=True, return_dict_in_generate=True, pad_token_id=0,
    )
    generation_seconds = time.monotonic() - start
    tokens = [t - offset for t in output.sequences[0, inputs.size(1):].tolist()]
    retained = output.past_key_values
    reference = oracle(case, 256, looping=True)
    expected, controls, registers = expected_tokens(reference)
    words, mirrors = cache_words(retained, circuit)
    scoped = cache_words(retained, circuit, scoped=True) if "scoped_words" in case else None
    register_exact = all(
        head_row(retained, circuit, name, warp, 4) == value
        for warp, bank in enumerate(registers)
        for name, value in zip(register_head_names(), bank, strict=True)
    )
    control_exact = all(
        head_row(retained, circuit, f"resident.control.{warp}", 0, 7) == (token & 127)
        for warp, token in enumerate(controls)
    )
    outcome = {DONE: "halted", DEADLOCK: "deadlock"}.get(
        tokens[-1] if tokens else None, "token_limit",
    )
    first_difference = next(
        (index for index, (actual, wanted) in enumerate(zip(tokens, expected))
         if actual != wanted),
        None if len(tokens) == len(expected) else min(len(tokens), len(expected)),
    )
    return {
        "outcome": outcome,
        "expected_outcome": reference["outcome"],
        "steps": len(reference["trace"]),
        "prompt_tokens": len(prompt),
        "generation_calls": 1,
        "generation_seconds": round(generation_seconds, 3),
        "host_execution_tokens": 0,
        "generated_tokens": tokens,
        "expected_tokens": expected,
        "global_words": words,
        "scoped_words": scoped[0] if scoped else None,
        "tokens_exact": tokens == expected,
        "first_token_difference": first_difference,
        "outcome_exact": outcome == reference["outcome"],
        "registers_exact": register_exact,
        "control_exact": control_exact,
        "global_exact": mirrors and words == reference["shared_words"],
        "scoped_exact": scoped is None or (scoped[1] and scoped[0] == reference["scoped_words"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("fixtures", type=Path, nargs="+")
    args = parser.parse_args()
    if args.model:
        meta = json.loads((args.model / "qwengpu9b.json").read_text())
        if meta["layout"] != gen.LAYOUT:
            raise ValueError("expected a stock resident generation checkpoint")
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        cfg = model.config
        circuit = build(Target(cfg.linear_num_key_heads, cfg.linear_num_value_heads,
                               cfg.intermediate_size), generate=True)
        offset = meta["chip_token_offset"]
    else:
        model, circuit = build_standalone(generate=True)
        offset = 0
    model.to(args.device).eval()
    report = {"device": args.device, "torch": torch.__version__,
              "circuit": circuit.stats(), "cases": []}
    for fixture in args.fixtures:
        payload = fixture.read_bytes()
        case = parse_launch(
            json.loads(payload), resident_globalgrid=True, resident_compute=True, full_integer=True,
        )
        result = evaluate(
            model, circuit, case, args.device, offset, max_new_tokens=args.max_new_tokens,
        )
        result.update(fixture=fixture.name, sha256=hashlib.sha256(payload).hexdigest())
        report["cases"].append(result)
        print(f"{fixture.name}: {result['steps']} instructions, "
              f"{len(result['generated_tokens'])} generated tokens", flush=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        if not all(value for key, value in result.items() if key.endswith("_exact")):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
