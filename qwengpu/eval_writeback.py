"""Check bounded internal writeback with TICK/state feedback only."""

from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
import transformers
from chipformer.q35.ir import Target
from qwengpu.compiled_rom import (
    ROM_SLOTS,
    STATE_BASE,
    TICK,
    load_a,
    program_tokens,
    state_token,
    ROM_LOAD_BASE,
)
from qwengpu.compiled_writeback import (
    build,
    build_standalone,
    CHUNK_BITS,
    SNAPSHOT_BITS,
)
from qwengpu.eval_compiled_rom import head_row
from qwengpu.reference import Instruction, WarpState, step
from qwengpu.run_kernel import parse_kernel


def snapshot(cache, circuit, stage: int) -> list[int] | None:
    packed = 0
    for chunk, start in enumerate(range(0, SNAPSHOT_BITS, CHUNK_BITS)):
        width = min(CHUNK_BITS, SNAPSHOT_BITS - start)
        value = head_row(cache, circuit, f"stage.{stage}.{chunk}", 0, width)
        if value is None:
            return None
        packed |= value << start
    return [(packed >> bit) & 255 for bit in range(0, SNAPSHOT_BITS, 8)]


@torch.no_grad()
def evaluate_case(
    model,
    circuit,
    initial: tuple[int, ...],
    program: tuple[Instruction, ...],
    device: str,
    offset: int = 0,
) -> dict:
    start = time.monotonic()
    uploads = program_tokens(program)
    prefix = [load_a(lane, value) for lane, value in enumerate(initial)]
    prefix += uploads + [state_token(0)]
    output = model(
        input_ids=torch.tensor([[t + offset for t in prefix]], device=device),
        use_cache=True,
    )
    cache = output.past_key_values
    reference = WarpState(lanes=initial)
    records = []
    token_calls = 0

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

    for stage in range(ROM_SLOTS):
        reference = step(reference, program)
        generated = feed(TICK)
        observed = snapshot(cache, circuit, stage)
        expected = list(reference.lanes) + list(reference.scratch[:32])
        expected_token = state_token(reference.pc, reference.halted)
        # Never substitute an oracle value for a model-generated token.
        valid = STATE_BASE <= generated < TICK
        if valid:
            feed(generated)
        observed_pc = head_row(cache, circuit, "pc", 0, 5)
        exact = (
            generated == expected_token
            and observed == expected
            and observed_pc == reference.pc + (16 if reference.halted else 0)
        )
        records.append(
            {
                "stage": stage,
                "generated": generated,
                "expected_token": expected_token,
                "observed": observed,
                "expected": expected,
                "observed_pc": observed_pc,
                "exact": exact,
            }
        )
        if not valid or (generated - STATE_BASE) & 16:
            break
    observed_rom = [head_row(cache, circuit, "rom", s, 11) for s in range(ROM_SLOTS)]
    expected_rom = [(t - ROM_LOAD_BASE) & 2047 for t in uploads]
    exact = (
        reference.halted
        and all(r["exact"] for r in records)
        and observed_rom == expected_rom
    )
    return {
        "initial": initial,
        "program": [{"op": i.op, "arg": i.arg} for i in program],
        "trace": records,
        "observed_rom": observed_rom,
        "expected_rom": expected_rom,
        "single_token_calls": token_calls,
        "lane_feedback_tokens": 0,
        "exact": exact,
        "seconds": round(time.monotonic() - start, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path)
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "eval/frozen/writeback_gate_v1",
    )
    args = parser.parse_args()
    paths = sorted(
        p for p in args.cases_dir.glob("*.json") if not p.name.startswith("._")
    )
    if len(paths) != 7:
        raise ValueError("expected seven frozen writeback kernels")
    meta = None
    offset = 0
    if args.model:
        meta = json.loads((args.model / "qwengpu9b.json").read_text())
        if meta["layout"] != "qwengpu-writeback9b-v1":
            raise ValueError("not a writeback checkpoint")
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
            )
        )
    else:
        model, circuit = build_standalone()
    model.to(args.device).eval()
    report = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "circuit": circuit.stats(),
        "merge": meta,
        "cases": [],
    }
    for path in paths:
        payload = path.read_bytes()
        initial, program = parse_kernel(json.loads(payload))
        record = evaluate_case(model, circuit, initial, program, args.device, offset)
        record.update(
            fixture=path.name, fixture_sha256=hashlib.sha256(payload).hexdigest()
        )
        report["cases"].append(record)
        print(
            json.dumps(
                {
                    "fixture": path.name,
                    "exact": record["exact"],
                    "seconds": record["seconds"],
                }
            ),
            flush=True,
        )
    report["all_exact"] = all(c["exact"] for c in report["cases"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
