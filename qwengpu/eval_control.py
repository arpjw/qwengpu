"""Verify ROM-driven loops/divergence with model-generated control feedback."""

from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
import transformers
from chipformer.q35.ir import Target
from qwengpu.compiled_control import (
    build,
    build_standalone,
    program_tokens,
    state_token,
    STATE_BASE,
    STATE_COUNT,
    STOP_STATES,
    GENERATION_LAYOUT,
    TICK,
    CHUNK_BITS,
    ROM_BASE,
    STRIDE,
    LINK_BASE,
)
from qwengpu.compiled_rom import load_a
from qwengpu.control_reference import (
    MAX_STEPS,
    ControlState,
    parse_kernel,
    snapshot_bits,
    step,
)
from qwengpu.reference import WarpState
from qwengpu.eval_compiled_rom import head_row


def snapshot(cache, circuit, stage: int) -> list[int] | None:
    packed = 0
    for chunk, start in enumerate(range(0, circuit.snapshot_bits, CHUNK_BITS)):
        value = head_row(
            cache,
            circuit,
            f"stage.{stage}.{chunk}",
            0,
            min(CHUNK_BITS, circuit.snapshot_bits - start),
        )
        if value is None:
            return None
        packed |= value << start
    return [(packed >> b) & 1 for b in range(circuit.snapshot_bits)]


def hex_bits(bits: list[int] | None) -> str | None:
    if bits is None:
        return None
    width = (len(bits) + 3) // 4
    return f"{sum(v << b for b, v in enumerate(bits)):0{width}x}"


def describe(bits: list[int] | None) -> dict | None:
    if bits is None:
        return None

    def field(start, width):
        return sum(bits[start + b] << b for b in range(width))

    return {
        "lanes": [field(i * 8, 8) for i in range(8)],
        "scratch": [field(64 + i * 8, 8) for i in range(32)],
        "mask": field(320, 8),
        "branch_depth": field(328, 2),
        "branch_frames": [
            {"parent": field(330 + i * 16, 8), "then": field(338 + i * 16, 8)}
            for i in range(2)
        ],
        "loop_depth": field(362, 2),
        "loop_parents": [field(364 + i * 8, 8) for i in range(2)],
    }


@torch.no_grad()
def evaluate_case(
    model, circuit, initial, program, device: str, offset: int = 0
) -> dict:
    start = time.monotonic()
    uploads = program_tokens(program)
    prefix = (
        [load_a(i, v) for i, v in enumerate(initial)] + uploads + [state_token(0, 0)]
    )
    output = model(
        input_ids=torch.tensor([[t + offset for t in prefix]], device=device),
        use_cache=True,
    )
    cache = output.past_key_values
    reference = ControlState(WarpState(lanes=initial))
    records = []
    token_calls = 0
    outcome = "invalid"

    def feed(token: int) -> int:
        nonlocal cache, token_calls
        output = model(
            input_ids=torch.tensor([[token + offset]], device=device),
            past_key_values=cache,
            use_cache=True,
        )
        cache = output.past_key_values
        token_calls += 1
        return int(output.logits[0, -1].argmax()) - offset

    for stage in range(MAX_STEPS):
        executed_pc = reference.warp.pc
        reference = step(reference, program)
        halted = reference.warp.halted
        exhausted = not halted and stage == MAX_STEPS - 1
        expected_token = state_token(
            reference.warp.pc,
            stage if halted or exhausted else stage + 1,
            halted,
            exhausted,
        )
        generated = feed(TICK)
        observed = snapshot(cache, circuit, stage)
        expected = snapshot_bits(reference)
        valid = STATE_BASE <= generated < STATE_BASE + STATE_COUNT
        if valid:
            feed(generated)
        observed_control = head_row(cache, circuit, "control", 0, 10)
        exact = (
            generated == expected_token
            and observed == expected
            and observed_control == expected_token - STATE_BASE
        )
        records.append(
            {
                "stage": stage,
                "executed_pc": executed_pc,
                "generated": generated,
                "expected_token": expected_token,
                "observed_control": observed_control,
                "observed_snapshot_hex": hex_bits(observed),
                "expected_snapshot_hex": hex_bits(expected),
                "observed": describe(observed),
                "expected": describe(expected),
                "exact": exact,
            }
        )
        # Execution schedule and termination depend only on generated status.
        if not valid:
            break
        status = generated - STATE_BASE
        if status & 256:
            outcome = "halted"
            break
        if status & 512:
            outcome = "exhausted"
            break
    observed_rom = [head_row(cache, circuit, "rom", s, 12) for s in range(16)]
    expected_rom = [(t - ROM_BASE) % STRIDE for t in uploads[:16]]
    observed_links = [head_row(cache, circuit, "link", s, 4) for s in range(16)]
    expected_links = [(t - LINK_BASE) % 16 for t in uploads[16:]]
    exact = (
        all(r["exact"] for r in records)
        and outcome in ("halted", "exhausted")
        and observed_rom == expected_rom
        and observed_links == expected_links
    )
    return {
        "initial": initial,
        "program": [{"op": i.op, "arg": i.arg} for i in program],
        "trace": records,
        "outcome": outcome,
        "exact": exact,
        "observed_rom": observed_rom,
        "expected_rom": expected_rom,
        "observed_links": observed_links,
        "expected_links": expected_links,
        "single_token_calls": token_calls,
        "lane_feedback_tokens": 0,
        "seconds": round(time.monotonic() - start, 3),
    }


@torch.no_grad()
def evaluate_generation(model, circuit, initial, program, device, offset=0) -> dict:
    """Launch once via stock generation and audit the retained state after it stops."""
    start = time.monotonic()
    uploads = program_tokens(program)
    prefix = [load_a(i, v) for i, v in enumerate(initial)] + uploads + [state_token(0, 0)]
    inputs = torch.tensor([[token + offset for token in prefix]], device=device)
    result = model.generate(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        do_sample=False,
        use_cache=True,
        eos_token_id=[offset + token for token in STOP_STATES],
        pad_token_id=0,
        max_new_tokens=MAX_STEPS,
        return_dict_in_generate=True,
    )
    generated = [token - offset for token in result.sequences[0, len(prefix):].tolist()]
    cache = result.past_key_values
    reference = ControlState(WarpState(lanes=initial))
    records = []
    expected_outcome = "exhausted"
    for stage in range(MAX_STEPS):
        executed_pc = reference.warp.pc
        reference = step(reference, program)
        halt = reference.warp.halted
        exhausted = stage == MAX_STEPS - 1 and not halt
        token = state_token(reference.warp.pc, stage if halt or exhausted else stage + 1,
                            halt, exhausted)
        actual = generated[stage] if stage < len(generated) else None
        observed = snapshot(cache, circuit, stage)
        expected = snapshot_bits(reference)
        records.append({
            "stage": stage,
            "executed_pc": executed_pc,
            "generated": actual,
            "expected_token": token,
            "observed_snapshot_hex": hex_bits(observed),
            "expected_snapshot_hex": hex_bits(expected),
            "exact": actual == token and observed == expected,
        })
        if halt or exhausted:
            expected_outcome = "halted" if halt else "exhausted"
            break
    terminal = generated[-1] if generated else None
    outcome = (
        "invalid" if terminal not in STOP_STATES else
        "halted" if (terminal - STATE_BASE) & 256 else "exhausted"
    )
    observed_rom = [head_row(cache, circuit, "rom", slot, 12) for slot in range(16)]
    expected_rom = [(token - ROM_BASE) % STRIDE for token in uploads[:16]]
    observed_links = [head_row(cache, circuit, "link", slot, 4) for slot in range(16)]
    expected_links = [(token - LINK_BASE) % 16 for token in uploads[16:]]
    return {
        "trace": records,
        "outcome": outcome,
        "exact": (
            outcome == expected_outcome
            and len(generated) == len(records)
            and all(record["exact"] for record in records)
            and observed_rom == expected_rom
            and observed_links == expected_links
        ),
        "generated_tokens": len(generated),
        "generation_calls": 1,
        "host_tick_tokens": 0,
        "host_control_feedback_tokens": 0,
        "lane_feedback_tokens": 0,
        "seconds": round(time.monotonic() - start, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path)
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--kernel", type=Path)
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "eval/frozen/control_gate_v1",
    )
    args = parser.parse_args()
    paths = (
        [args.kernel]
        if args.kernel
        else sorted(
            p for p in args.cases_dir.glob("*.json") if not p.name.startswith("._")
        )
    )
    if not args.kernel and len(paths) != 14:
        raise ValueError("expected 14 frozen control kernels")
    # Validate all inputs before constructing or calling a model.
    cases = []
    for path in paths:
        payload = path.read_bytes()
        data = json.loads(payload)
        initial, program = parse_kernel(data)
        outcome = data.get("expected_outcome", "halted")
        if outcome not in ("halted", "exhausted"):
            raise ValueError("invalid expected_outcome")
        cases.append((path, payload, initial, program, outcome))
    meta = None
    offset = 0
    if args.model:
        meta = json.loads((args.model / "qwengpu9b.json").read_text())
        layout = GENERATION_LAYOUT if args.generate else "qwengpu-control9b-v1"
        if meta["layout"] != layout:
            raise ValueError("not a control checkpoint")
        offset = meta["chip_token_offset"]
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16
        )
        cfg = model.config
        circuit = build(
            Target(
                cfg.linear_num_key_heads,
                cfg.linear_num_value_heads,
                cfg.intermediate_size,
            ),
            generate=args.generate,
        )
    else:
        model, circuit = build_standalone(generate=args.generate)
    model.to(args.device).eval()
    report = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "circuit": circuit.stats(),
        "merge": meta,
        "cases": [],
    }
    for path, payload, initial, program, outcome in cases:
        evaluate = evaluate_generation if args.generate else evaluate_case
        record = evaluate(model, circuit, initial, program, args.device, offset)
        record.update(
            fixture=path.name,
            fixture_sha256=hashlib.sha256(payload).hexdigest(),
            expected_outcome=outcome,
            outcome_matched=outcome == record["outcome"],
        )
        report["cases"].append(record)
        print(
            json.dumps(
                {
                    "fixture": path.name,
                    "exact": record["exact"],
                    "outcome": record["outcome"],
                    "seconds": record["seconds"],
                }
            ),
            flush=True,
        )
    report["all_exact"] = all(c["exact"] for c in report["cases"])
    report["all_expected_outcomes"] = all(c["outcome_matched"] for c in report["cases"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_exact"] or not report["all_expected_outcomes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
