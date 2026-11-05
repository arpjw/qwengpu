"""Compare a mixed one-level SIMT program with the independent interpreter."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import transformers

from chipformer.q35.ir import Target

from qwengpu.compiled_add import (ELSE_TOKEN, ENDIF_TOKEN, EXEC_ADD, EXEC_XOR,
                                  SCRATCH_SLOTS, build, build_standalone,
                                  compare_token, load_a, load_b, load_token,
                                  mask_token, out_token, read_lane, store_token)
from qwengpu.eval_memory import scratch_snapshot
from qwengpu.reference import Instruction, LANES, WarpState, step


def program_for(threshold: int, then_add: int, else_xor: int,
                join_add: int) -> tuple[Instruction, ...]:
    return (Instruction("STORE", 16), Instruction("IF_LT", threshold),
            Instruction("ADDI", then_add), Instruction("STORE", 0),
            Instruction("ELSE"), Instruction("XORI", else_xor),
            Instruction("STORE", 8), Instruction("ENDIF"),
            Instruction("LOAD", 16), Instruction("ADDI", join_add),
            Instruction("HALT"))


@torch.no_grad()
def evaluate_case(model, circuit, initial: tuple[int, ...],
                  program: tuple[Instruction, ...], device: str, offset: int = 0,
                  probe_final: bool = False) -> dict:
    start = time.monotonic()
    prefix = [load_a(lane, value) for lane, value in enumerate(initial)]
    prefix.append(mask_token(255))
    output = model(input_ids=torch.tensor([[token + offset for token in prefix]],
                                            device=device), use_cache=True)
    cache = output.past_key_values
    generated = []
    expected = []
    state = WarpState(lanes=initial)
    written = [[False] * LANES for _ in range(SCRATCH_SLOTS)]

    def feed(token: int) -> int:
        nonlocal cache
        result = model(input_ids=torch.tensor([[token + offset]], device=device),
                       past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        return int(result.logits[0, -1].argmax()) - offset

    def expect_output(token: int, wanted: int) -> None:
        observed = feed(token)
        generated.append(observed)
        expected.append(wanted)
        feed(observed)

    while not state.halted:
        ins = program[state.pc]
        next_state = step(state, program)
        if ins.op in ("ADDI", "XORI"):
            for lane in range(LANES):
                feed(load_b(lane, ins.arg))
            feed(EXEC_ADD if ins.op == "ADDI" else EXEC_XOR)
        elif ins.op == "STORE":
            slot = ins.arg // LANES
            feed(store_token(slot))
            for lane in range(LANES):
                if state.active_mask >> lane & 1:
                    written[slot][lane] = True
        elif ins.op == "LOAD":
            feed(load_token(ins.arg // LANES))
        elif ins.op == "IF_LT":
            expect_output(compare_token(ins.arg), mask_token(next_state.active_mask))
        elif ins.op == "ELSE":
            expect_output(ELSE_TOKEN, mask_token(next_state.active_mask))
        elif ins.op == "ENDIF":
            expect_output(ENDIF_TOKEN, mask_token(next_state.active_mask))
        elif ins.op != "HALT":
            raise ValueError(f"unsupported opcode {ins.op}")
        if ins.op in ("ADDI", "XORI", "LOAD"):
            for lane, value in enumerate(next_state.lanes):
                expect_output(read_lane(lane), out_token(lane, value))
        state = next_state

    final_probe = None
    if probe_final:
        feed(mask_token(255))
        for lane in range(LANES):
            feed(load_b(lane, 0))
        feed(EXEC_XOR)
        final_probe = []
        for lane, value in enumerate(state.lanes):
            observed = feed(read_lane(lane))
            final_probe.append(observed)
            generated.append(observed)
            expected.append(out_token(lane, value))
            feed(observed)

    observed_scratch = [scratch_snapshot(cache, circuit, slot)
                        for slot in range(SCRATCH_SLOTS)]
    expected_scratch = [
        [state.scratch[slot * LANES + lane] if written[slot][lane] else None
         for lane in range(LANES)] for slot in range(SCRATCH_SLOTS)]
    return {"initial": initial,
            "program": [{"op": ins.op, "arg": ins.arg} for ins in program],
            "generated": generated, "expected": expected,
            "observed_scratch": observed_scratch,
            "expected_scratch": expected_scratch,
            "tokens_exact": generated == expected,
            "scratch_exact": observed_scratch == expected_scratch,
            "exact": generated == expected and observed_scratch == expected_scratch,
            "final_probe": final_probe,
            "final_lanes": state.lanes,
            "seconds": round(time.monotonic() - start, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extra-cases", type=int,
                        help="seeded additional cases (default: 8 for a merge, 0 standalone)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    if args.extra_cases is not None and args.extra_cases < 0:
        parser.error("--extra-cases must be nonnegative")
    extra_cases = args.extra_cases if args.extra_cases is not None else (8 if args.model else 0)
    offset = 0
    if args.model:
        meta = json.loads((args.model / "qwengpu9b.json").read_text(encoding="utf-8"))
        offset = int(meta["chip_token_offset"])
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16).to(args.device).eval()
        cfg = model.config
        circuit = build(Target(cfg.linear_num_key_heads,
                               cfg.linear_num_value_heads,
                               cfg.intermediate_size))
    else:
        model, circuit = build_standalone()
        model.to(args.device)
        meta = None
    cases = [
        (tuple(range(8)), 4, 10, 85, 1),
        ((255, 0, 128, 127, 0, 1, 200, 199), 128, 250, 17, 7),
        ((100,) * 8, 0, 9, 13, 5),
        ((100,) * 8, 255, 9, 13, 5),
    ]
    if extra_cases:
        rng = random.Random(20261105)
        cases.extend((tuple(rng.randrange(256) for _ in range(LANES)),
                      rng.randrange(256), rng.randrange(256),
                      rng.randrange(256), rng.randrange(256))
                     for _ in range(extra_cases))
    report = {"device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__, "circuit": circuit.stats(),
              "merge": meta, "cases": []}
    for initial, threshold, then_add, else_xor, join_add in cases:
        program = program_for(threshold, then_add, else_xor, join_add)
        result = evaluate_case(model, circuit, initial, program, args.device, offset)
        report["cases"].append(result)
        print(json.dumps({"threshold": threshold, "exact": result["exact"],
                          "seconds": result["seconds"]}), flush=True)
    report["all_exact"] = all(case["exact"] for case in report["cases"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_exact": report["all_exact"], "out": str(args.out)}), flush=True)
    if not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
