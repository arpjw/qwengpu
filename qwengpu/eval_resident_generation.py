"""Run resident integer GPU workgroups through stock generation and audit the cache."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from chipformer.q35.ir import Target
from transformers import AutoModelForCausalLM, Qwen3_5TextConfig, StoppingCriteria

from qwengpu import resident_generation as gen
from qwengpu.alu32 import ALU_VOCAB, RUNTIME_EXTENT_BASE
from qwengpu.cache_register_edge import register_head_names
from qwengpu.compiled_tile8 import (
    ADVANCE, C_MEMORY, DEADLOCK, DONE, GLOBAL_GRID, GRID, INDEX_OPERAND, MASK_REGISTER, ROM32_PC_HIGH,
    ROM_BANK, ROM_PAYLOAD_BITS, ROW_PAIRS, TICKET,
    WARP_BANKS, WARP_DIRTY_OFFSET, WIDE_WARP_TAGS, WORKGROUP_TICKETS,
    build, build_standalone, program_tokens, rom_payload,
)
from qwengpu.eval_compiled_rom import head_row, head_rows
from qwengpu.eval_tile8 import _check_resident_memory, _digits, _words, nibbles, oracle, parse_launch
from qwengpu.memory_geometry import WIDE_MEMORY
from qwengpu.resident_scheduler import (
    EXTRA_WARP_BANKS, RESIDENT_ENABLE, RESIDENT_TICKET, resident_control,
)
from qwengpu.scoped_memory import SCOPED_BASE, SCOPED_WORDS


def launch_rom(case):
    return [
        program_tokens(program, resident_rom32=True, resident_alu32=True,
                       resident_scoped="scoped_words" in case, full_integer=True)
        for program in case["programs"]
    ]


def launch_prompt(case, refresh_words=None, *, retain_tile_output=False):
    groups = case.get("workgroups")
    if groups is not None and groups not in (1, 2, 3, 4):
        raise ValueError("stock generation requires one through four workgroups")
    if groups is None and refresh_words is not None:
        raise ValueError("global refresh requires a workgroup launch")
    interleaved = case.get("interleaved", False)
    active_warps = 4 if interleaved else 2
    descriptor = (
        GRID + GLOBAL_GRID + ROW_PAIRS * (groups - 1) if groups is not None else
        GRID + case["first_row_pair"] + case["row_pairs"] - 1
    )
    tokens = [RESIDENT_ENABLE,
              *(resident_control(warp, status=int(warp >= active_warps)) for warp in range(4)),
              RESIDENT_TICKET + 3, descriptor]
    if groups is None:
        tokens.extend(nibbles(INDEX_OPERAND, _digits(case["a"] + case["b"], 8)))
        if not retain_tile_output:
            tokens.extend(nibbles(C_MEMORY, _digits(case["c"], 16)))
        tokens.append(gen.ROW + case["first_row_pair"])
    elif groups > 1 and not interleaved:
        tokens.append(gen.GROUP)
    if groups is not None:
        words = case["global_words"]
        if refresh_words is not None:
            words = words[:refresh_words]
        tokens.extend(nibbles(WIDE_MEMORY.base, _digits(words, 16)))
    if "scoped_words" in case:
        tokens.extend(nibbles(SCOPED_BASE, _digits(case["scoped_words"], 16)))
    if "extent" in case:
        tokens.append(RUNTIME_EXTENT_BASE + case["extent"])
    for warp, program in enumerate(launch_rom(case)):
        tokens.append(ROM_BANK + warp)
        tokens.extend(program)
    for bank, digits in zip((*WARP_BANKS, *EXTRA_WARP_BANKS), launch_registers(case), strict=True):
        tokens.extend(nibbles(bank, digits))
    return [*tokens, gen.SCHEDULE]


def initial_registers():
    return [15 if token in (MASK_REGISTER, MASK_REGISTER + 16) else 0
            for token in WIDE_WARP_TAGS]


def launch_registers(case):
    registers = [initial_registers() for _ in range(4)]
    if "workgroups" not in case:
        for bank, words in zip(registers[:2], case["initial_registers"], strict=True):
            bank[:32] = _digits([word & 65535 for word in words], 16)
            bank[-32:] = _digits([word >> 16 for word in words], 16)
    return registers


def resident_reference(reference, *, tiled=False):
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
                **({"advance_row": ((schedule["ticket"] - TICKET) >> 3) & 3}
                   if tiled and schedule["ticket"] & ADVANCE and schedule["ticket"] not in (DONE, DEADLOCK)
                   else {}),
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


def expected_tokens(reference, active_warps=4, *, registers=None):
    """Encode the independent interpreter's trace for comparison after generation."""
    records = []
    controls = [resident_control(warp, status=int(warp >= active_warps)) for warp in range(4)]
    if registers is None:
        registers = [initial_registers() for _ in range(4)]
    for index, schedule in enumerate(reference["schedules"]):
        if "advance_group" in schedule:
            records.extend((gen.GROUP + schedule["advance_group"], *gen.RESET,
                            gen.SCHEDULE))
            registers[:2] = [initial_registers(), initial_registers()]
        elif "advance_row" in schedule:
            records.extend((gen.ROW + schedule["advance_row"], gen.SCHEDULE))
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


class InstructionLimit(StoppingCriteria):
    """Stop before executing an issued ticket once the instruction budget is spent."""

    def __init__(self, maximum, offset, device):
        self.issued = torch.zeros(1, dtype=torch.int32, device=device)
        self.maximum = maximum
        self.offset = offset

    def __call__(self, input_ids, scores, **kwargs):
        token = input_ids[:, -1] - self.offset
        self.issued += (token >= gen.TICKET) & (token < gen.TICKET + 4)
        return self.issued > self.maximum


def cache_words(cache, circuit, *, scoped=False):
    result = []
    mirrored = True
    words_per_bank = SCOPED_WORDS if scoped else 128
    for bank in range(1 if scoped else WIDE_MEMORY.words // words_per_bank):
        columns = []
        for nib in range(4):
            mirrors = [
                head_rows(
                    cache, circuit, f"shared.{lane}.{nib}" if scoped else f"ram.{lane}.{nib}.{bank}",
                    0, words_per_bank, 4,
                )
                for lane in range(8)
            ]
            canonical, *copies = mirrors
            mirrored &= None not in canonical and all(copy == canonical for copy in copies)
            columns.append(canonical)
        result.extend(value or 0 for word in zip(*columns, strict=True) for value in word)
    return _words(result), mirrored


def cache_tile_words(cache, circuit):
    digits = [
        head_row(cache, circuit, f"shared.0.{nib}", word, 4)
        for word in range(128)
        for nib in range(4)
    ]
    words = _words([value or 0 for value in digits])
    operands = [byte for word in words[:64] for byte in (word & 255, word >> 8)]
    return words, None not in digits and _check_resident_memory(
        cache, circuit, nibbles(INDEX_OPERAND, _digits(operands, 8)),
        nibbles(C_MEMORY, _digits(words[64:], 16)), indexed=True,
    )


def load_generation_model(checkpoint, device):
    standalone = checkpoint is None
    if standalone:
        weights = os.environ.get("QWENGPU_STANDALONE_WEIGHTS")
        if not weights:
            model, circuit = build_standalone(generate=True)
            return model.to(device).eval(), circuit, 0
        checkpoint = Path(weights)
        if not (checkpoint / "config.json").is_file() or (checkpoint / "qwengpu9b.json").exists():
            raise ValueError("QWENGPU_STANDALONE_WEIGHTS requires the standalone circuit")
        offset = 0
    else:
        meta = json.loads((checkpoint / "qwengpu9b.json").read_text())
        if meta.get("layout") not in gen.LAYOUTS:
            raise ValueError(f"expected a {gen.LAYOUT} checkpoint")
        offset = meta["chip_token_offset"]
    if standalone:
        # Match build_standalone's circuit schedule. The model pads its MLP
        # beyond the circuit's scheduling capacity for the emitter.
        config = Qwen3_5TextConfig.from_pretrained(checkpoint)
        layout = getattr(config, "qwengpu_layout", gen.LAYOUT)
        if layout not in gen.LAYOUTS:
            raise ValueError(f"unknown standalone generation layout {layout!r}")
        circuit = build(generate=True, generated_tilegrid=layout == gen.TILEGRID_LAYOUT)
        expected_layers = ["linear_attention"] * circuit.n_layers() + ["full_attention"]
        if (config.layer_types != expected_layers or config.vocab_size != ALU_VOCAB
                or config.linear_num_key_heads != circuit.target.key_heads
                or config.linear_num_value_heads != circuit.target.value_heads
                or config.intermediate_size != 5120 or not config.tie_word_embeddings):
            raise ValueError("expected a standalone resident generation model")
        model = AutoModelForCausalLM.from_pretrained(checkpoint, config=config, dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16)
        config = model.config
        circuit = build(Target(config.linear_num_key_heads, config.linear_num_value_heads,
                               config.intermediate_size), generate=True,
                        generated_tilegrid=meta["layout"] == gen.TILEGRID_LAYOUT)
    return model.to(device).eval(), circuit, offset


@torch.no_grad()
def evaluate(
    model, circuit, case, device, offset=0, *, max_new_tokens=32768,
    cache=None, refresh_words=None, max_steps=256, retain_tile_output=False,
):
    start = time.monotonic()
    tiled = "workgroups" not in case
    if retain_tile_output and (not tiled or cache is None):
        raise ValueError("retaining tile output requires a tiled launch with an existing cache")
    if refresh_words is not None and (
        cache is None
        or type(refresh_words) is not int
        or not 0 <= refresh_words <= WIDE_MEMORY.words
    ):
        raise ValueError("global refresh requires a retained cache and a valid prefix length")
    prompt = launch_prompt(case, refresh_words, retain_tile_output=retain_tile_output)
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
        stopping_criteria=[InstructionLimit(max_steps, offset, device)],
    )
    generation_seconds = time.monotonic() - start
    tokens = [t - offset for t in output.sequences[0, inputs.size(1):].tolist()]
    retained = output.past_key_values
    reference = oracle(case, max_steps, looping=True)
    sequential = not case.get("interleaved")
    if sequential:
        reference = resident_reference(reference, tiled=tiled)
    expected, controls, registers = expected_tokens(
        reference, 2 if sequential else 4, registers=launch_registers(case),
    )
    if tiled:
        words, mirrors = cache_tile_words(retained, circuit)
    else:
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
    rom_exact = all(
        head_row(retained, circuit, "rom", len(program) * warp + address, ROM_PAYLOAD_BITS + 3)
        == rom_payload(
            token, indexed=True, masked=True, divergent=True, looping=True,
            resident_add32=True, resident_shared32=True, resident_addmem32=True,
            resident_branch32=True, resident_loop32=True, resident_globalgrid=True,
            resident_compute=True, resident_memory512=True, resident_rom32=True,
            resident_alu32=True, resident_scoped=True, full_integer=True,
        )
        for warp, program in enumerate(launch_rom(case))
        for address, token in enumerate(program)
    )
    group_exact = True
    if sequential:
        cursor = head_row(retained, circuit, "workgroup", 0, 3)
        transitions = sum(gen.GROUP < token < gen.GROUP + 4 for token in expected)
        if tiled:
            rows = [token - gen.ROW for token in expected if gen.ROW <= token < gen.ROW + ROW_PAIRS]
            group_exact = cursor == (rows[-1] if rows else case["first_row_pair"])
        else:
            group_exact = cursor == (4 | transitions) if case["workgroups"] > 1 else cursor == 0
    outcome = {DONE: "halted", DEADLOCK: "deadlock"}.get(
        tokens[-1] if tokens else None, "token_limit",
    )
    if outcome == "token_limit" and sum(
        gen.TICKET <= token < gen.TICKET + 4 for token in tokens
    ) > max_steps:
        outcome = "step_limit"
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
        "global_upload_words": 0 if tiled else WIDE_MEMORY.words if refresh_words is None else refresh_words,
        **({"tile_output_upload_words": 0 if retain_tile_output else len(case["c"])}
           if tiled else {}),
        "cache_length_exact": (
            retained.get_seq_length() == cache_length + len(prompt) + len(tokens) - 1
        ),
        "cache_retained_exact": cache is None or retained is cache,
        "generated_tokens": tokens,
        "expected_tokens": expected,
        "tile_words" if tiled else "global_words": words,
        "scoped_words": scoped[0] if scoped else None,
        "tokens_exact": tokens == expected,
        "first_token_difference": first_difference,
        "outcome_exact": outcome == reference["outcome"],
        "registers_exact": observed_registers == registers,
        "control_exact": control_exact,
        "rom_exact": rom_exact,
        "workgroup_cursor_exact": group_exact,
        "memory_exact" if tiled else "global_exact": mirrors and words == reference["shared_words"],
        "scoped_exact": scoped is None or (scoped[1] and scoped[0] == reference["scoped_words"]),
    }
    return retained, report


def launch_compute(model, circuit, raw, device, offset, cache, *, transport_mode,
                   refresh_global_prefix=0):
    if transport_mode != "stock_generation":
        raise ValueError("expected stock generation transport")
    case = parse_launch(raw, full_integer=True)
    retained, report = evaluate(
        model, circuit, case, device, offset, cache=cache,
        refresh_words=None if cache is None else refresh_global_prefix,
    )
    failures = [key for key, value in report.items() if key.endswith("_exact") and not value]
    if failures:
        raise ValueError(f"generation differs at token {report['first_token_difference']}: {failures}")
    if report["outcome"] != "halted":
        raise ValueError(f"compute launch ended with {report['outcome']}")
    return retained, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("fixtures", type=Path, nargs="+")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, circuit, offset = load_generation_model(args.model, args.device)
    report = {"device": args.device, "torch": torch.__version__,
              "circuit": circuit.stats(),
              "merge": json.loads((args.model / "qwengpu9b.json").read_text()) if args.model else None,
              "cases": []}
    for fixture in args.fixtures:
        payload = fixture.read_bytes()
        raw = json.loads(payload)
        case = parse_launch(
            raw, resident_globalgrid=True, resident_compute=True, full_integer=True,
        )
        _, result = evaluate(
            model, circuit, case, args.device, offset, max_new_tokens=args.max_new_tokens,
            max_steps=args.max_steps,
        )
        result.update(fixture=fixture.name, sha256=hashlib.sha256(payload).hexdigest())
        if "expected_outcome" in raw:
            result["fixture_outcome_exact"] = result["outcome"] == raw["expected_outcome"]
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
