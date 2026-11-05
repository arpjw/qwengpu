"""Free-running one-level SIMT divergence and reconvergence on Qwen3.5."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import transformers

from qwengpu.compiled_add import (ELSE_TOKEN, ENDIF_TOKEN, EXEC_ADD,
                                  build_standalone, compare_token, load_a,
                                  load_b, mask_token, out_token, read_lane)
from qwengpu.reference import LANES, Instruction, WarpState, run


@torch.no_grad()
def evaluate_case(model, a: tuple[int, ...], threshold: int, then_add: int,
                  else_add: int, join_add: int, device: str, offset: int = 0) -> dict:
    start = time.monotonic()
    prefix = [token for lane in range(LANES)
              for token in (load_a(lane, a[lane]), load_b(lane, then_add))]
    prefix.append(mask_token(255))
    output = model(input_ids=torch.tensor([[token + offset for token in prefix]],
                                            device=device), use_cache=True)
    cache = output.past_key_values
    generated = []
    expected = []

    def feed(token: int) -> int:
        nonlocal cache
        result = model(input_ids=torch.tensor([[token + offset]], device=device),
                       past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        return int(result.logits[0, -1].argmax()) - offset

    def check_output(command: int, expected_token: int) -> None:
        observed = feed(command)
        generated.append(observed)
        expected.append(expected_token)
        feed(observed)

    def add_and_read(values: tuple[int, ...]) -> None:
        feed(EXEC_ADD)
        for lane, value in enumerate(values):
            check_output(read_lane(lane), out_token(lane, value))

    then_mask = sum(1 << lane for lane, value in enumerate(a) if value < threshold)
    check_output(compare_token(threshold), mask_token(then_mask))
    after_then = tuple((value + then_add) & 255 if then_mask >> lane & 1 else value
                       for lane, value in enumerate(a))
    add_and_read(after_then)

    else_mask = 255 & ~then_mask
    check_output(ELSE_TOKEN, mask_token(else_mask))
    for lane in range(LANES):
        feed(load_b(lane, else_add))
    after_else = tuple((value + else_add) & 255 if else_mask >> lane & 1 else value
                       for lane, value in enumerate(after_then))
    add_and_read(after_else)

    check_output(ENDIF_TOKEN, mask_token(255))
    for lane in range(LANES):
        feed(load_b(lane, join_add))
    after_join = tuple((value + join_add) & 255 for value in after_else)
    add_and_read(after_join)
    reference = run((Instruction("IF_LT", threshold), Instruction("ADDI", then_add),
                     Instruction("ELSE"), Instruction("ADDI", else_add),
                     Instruction("ENDIF"), Instruction("ADDI", join_add),
                     Instruction("HALT")), state=WarpState(lanes=a))
    assert after_join == reference.lanes
    return {"a": a, "threshold": threshold, "then_add": then_add,
            "else_add": else_add, "join_add": join_add,
            "generated": generated, "expected": expected,
            "exact": generated == expected,
            "final_lanes": after_join,
            "seconds": round(time.monotonic() - start, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        (tuple(range(LANES)), 4, 10, 20, 1),
        ((255, 0, 128, 127, 0, 1, 200, 199), 128, 1, 250, 7),
        ((100,) * LANES, 0, 9, 13, 5),
        ((100,) * LANES, 255, 9, 13, 5),
    ]
    model, circuit = build_standalone()
    model.to(args.device)
    report = {"device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__, "circuit": circuit.stats(),
              "hidden": model.config.hidden_size,
              "layers": model.config.num_hidden_layers, "cases": []}
    for a, threshold, then_add, else_add, join_add in cases:
        result = evaluate_case(model, a, threshold, then_add, else_add, join_add,
                               args.device)
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
