"""Run resident integer GPU workgroups through stock generation and audit the cache."""

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
    DEADLOCK, DONE, GLOBAL_GRID, GRID, MASK_REGISTER, ROM32_PC_HIGH, ROM_BANK, ROW_PAIRS,
    WARP_BANKS, WARP_DIRTY_OFFSET, WIDE_WARP_TAGS, WORKGROUP_TICKETS,
    build, build_standalone, program_tokens,
)
from qwengpu.eval_compiled_rom import head_row
from qwengpu.eval_tile8 import _digits, _words, nibbles, oracle, parse_launch
from qwengpu.memory_geometry import WIDE_MEMORY
from qwengpu.resident_scheduler import (
    EXTRA_WARP_BANKS, RESIDENT_ENABLE, RESIDENT_TICKET, resident_control,
)
from qwengpu.scoped_memory import SCOPED_BASE, SCOPED_WORDS


def launch_prompt(case, refresh_words=None):
    groups = case.get("workgroups")
    if groups not in (1, 2, 3, 4):
        raise ValueError("stock generation requires one through four workgroups")
    interleaved = case.get("interleaved", False)
    active_warps = 4 if interleaved else 2
    tokens = [RESIDENT_ENABLE,
              *(resident_control(warp, status=int(warp >= active_warps)) for warp in range(4)),
              RESIDENT_TICKET + 3, GRID + GLOBAL_GRID + ROW_PAIRS * (groups - 1)]
    if groups > 1 and not interleaved:
        tokens.append(gen.GROUP)
    words = case["global_words"]
    if refresh_words is not None:
        words = words[:refresh_words]
    tokens.extend(nibbles(WIDE_MEMORY.base, _digits(words, 16)))
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


def resident_reference(reference):
    """Translate the sequential reference trace into the four-bank wire format."""

    def control(warp, token):
        pc = (token & 15) + (16 if token & ROM32_PC_HIGH else 0)
        return resident_control(warp, pc, (token >> 4) & 3)

    inactive = [resident_control(warp, status=1) for warp in (2, 3)]
    return {
        **reference,
        "schedules": [
            {
                **({"advance_group": WORKGROUP_TICKETS.index(schedule["ticket"]) + 1}
                   if schedule["ticket"] in WORKGROUP_TICKETS else {}),
                "ticket": (
                    schedule["ticket"] if schedule["ticket"] in (DONE, DEADLOCK)
                    else RESIDENT_TICKET + (schedule["ticket"] & 1)
                ),
                "controls": [control(warp, token) for warp, token in enumerate(schedule["controls"])]
                + inactive,
            }
            for schedule in reference["schedules"]
        ],
        "trace": [
            {**record, "control": control(record["warp"], record["control"])}
            for record in reference["trace"]
        ],
    }


def expected_tokens(reference, active_warps=4):
    """Encode the independent interpreter's trace for comparison after generation."""
    records = []
    controls = [resident_control(warp, status=int(warp >= active_warps)) for warp in range(4)]
    registers = [initial_registers() for _ in range(4)]
    for index, schedule in enumerate(reference["schedules"]):
        if "advance_group" in schedule:
            records.extend((gen.GROUP + schedule["advance_group"], *gen.RESET,
                            gen.SCHEDULE))
            registers[:2] = [initial_registers(), initial_registers()]
        elif controls != schedule["controls"]:
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
    words = SCOPED_WORDS if scoped else WIDE_MEMORY.words
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
def evaluate(
    model, circuit, case, device, offset=0, *, max_new_tokens=32768,
    cache=None, refresh_words=None,
):
    start = time.monotonic()
    if refresh_words is not None and (
        cache is None
        or type(refresh_words) is not int
        or not 0 <= refresh_words <= WIDE_MEMORY.words
    ):
        raise ValueError("global refresh requires a retained cache and a valid prefix length")
    prompt = launch_prompt(case, refresh_words)
    cache_length = 0 if cache is None else cache.get_seq_length()
    inputs = torch.tensor([[token + offset for token in prompt]], device=device)
    if cache_length:
        # Generation slices this prefix using the existing cache length.
        inputs = torch.cat((inputs.new_zeros((1, cache_length)), inputs), dim=1)
    output = model.generate(
        input_ids=inputs, attention_mask=torch.ones_like(inputs),
        past_key_values=cache,
        do_sample=False, eos_token_id=[offset + DONE, offset + DEADLOCK],
        max_new_tokens=max_new_tokens, use_cache=True, return_dict_in_generate=True, pad_token_id=0,
    )
    generation_seconds = time.monotonic() - start
    tokens = [t - offset for t in output.sequences[0, inputs.size(1):].tolist()]
    retained = output.past_key_values
    reference = oracle(case, 256, looping=True)
    sequential = not case.get("interleaved")
    if sequential:
        reference = resident_reference(reference)
    expected, controls, registers = expected_tokens(reference, 2 if sequential else 4)
    words, mirrors = cache_words(retained, circuit)
    scoped = cache_words(retained, circuit, scoped=True) if "scoped_words" in case else None
    observed_registers = [
        [head_row(retained, circuit, name, warp, 4) for name in register_head_names()]
        for warp in range(4)
    ]
    control_exact = all(
        head_row(retained, circuit, f"resident.control.{warp}", 0, 7) == (token & 127)
        for warp, token in enumerate(controls)
    )
    group_exact = True
    if sequential:
        cursor = head_row(retained, circuit, "workgroup", 0, 3)
        transitions = sum(gen.GROUP < token < gen.GROUP + 4 for token in expected)
        group_exact = cursor == (4 | transitions) if case["workgroups"] > 1 else cursor == 0
    outcome = {DONE: "halted", DEADLOCK: "deadlock"}.get(
        tokens[-1] if tokens else None, "token_limit",
    )
    first_difference = next(
        (index for index, (actual, wanted) in enumerate(zip(tokens, expected))
         if actual != wanted),
        None if len(tokens) == len(expected) else min(len(tokens), len(expected)),
    )
    report = {
        "outcome": outcome,
        "expected_outcome": reference["outcome"],
        "steps": len(reference["trace"]),
        "prompt_tokens": len(prompt),
        "generation_calls": 1,
        "generation_seconds": round(generation_seconds, 3),
        "host_execution_tokens": 0,
        "global_upload_words": WIDE_MEMORY.words if refresh_words is None else refresh_words,
        "cache_length_exact": (
            retained.get_seq_length() == cache_length + len(prompt) + len(tokens) - 1
        ),
        "cache_retained_exact": cache is None or retained is cache,
        "generated_tokens": tokens,
        "expected_tokens": expected,
        "global_words": words,
        "scoped_words": scoped[0] if scoped else None,
        "tokens_exact": tokens == expected,
        "first_token_difference": first_difference,
        "outcome_exact": outcome == reference["outcome"],
        "registers_exact": observed_registers == registers,
        "control_exact": control_exact,
        "workgroup_cursor_exact": group_exact,
        "global_exact": mirrors and words == reference["shared_words"],
        "scoped_exact": scoped is None or (scoped[1] and scoped[0] == reference["scoped_words"]),
    }
    return retained, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("fixtures", type=Path, nargs="+")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
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
        _, result = evaluate(
            model, circuit, case, args.device, offset, max_new_tokens=args.max_new_tokens,
        )
        result.update(fixture=fixture.name, sha256=hashlib.sha256(payload).hexdigest())
        report["cases"].append(result)
        failed = [key for key, value in result.items() if key.endswith("_exact") and not value]
        print(f"{fixture.name}: {result['steps']} instructions, "
              f"{len(result['generated_tokens'])} generated tokens"
              f"{', failed: ' + ', '.join(failed) if failed else ', exact'}", flush=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
