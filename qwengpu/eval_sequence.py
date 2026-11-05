"""Check two model-executed vector instructions with result-token writeback."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import transformers

from qwengpu.compiled_add import (EXEC_ADD, build_standalone, load_a, load_b,
                                  mask_token, out_token, read_lane)
from qwengpu.reference import LANES


@torch.no_grad()
def evaluate_case(model, a: tuple[int, ...], b: tuple[int, ...], mask: int,
                  device: str) -> dict:
    start = time.monotonic()
    prefix = [token for lane in range(LANES)
              for token in (load_a(lane, a[lane]), load_b(lane, b[lane]))]
    prefix.append(mask_token(mask))
    output = model(input_ids=torch.tensor([prefix], device=device), use_cache=True)
    cache = output.past_key_values
    expected = []
    generated = []
    current = a
    for _ in range(2):
        output = model(input_ids=torch.tensor([[EXEC_ADD]], device=device),
                       past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        current = tuple((current[lane] + b[lane]) & 255 if mask >> lane & 1
                        else current[lane] for lane in range(LANES))
        for lane, value in enumerate(current):
            output = model(input_ids=torch.tensor([[read_lane(lane)]], device=device),
                           past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            token = int(output.logits[0, -1].argmax())
            generated.append(token)
            expected.append(out_token(lane, value))
            output = model(input_ids=torch.tensor([[token]], device=device),
                           past_key_values=cache, use_cache=True)
            cache = output.past_key_values
    return {"a": a, "b": b, "mask": mask, "generated": generated,
            "expected": expected, "exact": generated == expected,
            "seconds": round(time.monotonic() - start, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        (tuple(range(8)), (17,) * 8, 0b10101010),
        ((255, 0, 128, 127, 0, 1, 200, 199), (1,) * 8, 255),
        ((100,) * 8, (50,) * 8, 0),
        ((250,) * 8, (20,) * 8, 0b01010101),
    ]
    model, circuit = build_standalone()
    model.to(args.device)
    report = {"device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__, "circuit": circuit.stats(),
              "hidden": model.config.hidden_size,
              "layers": model.config.num_hidden_layers, "cases": []}
    for a, b, mask in cases:
        result = evaluate_case(model, a, b, mask, args.device)
        report["cases"].append(result)
        print(json.dumps({"mask": mask, "exact": result["exact"],
                          "seconds": result["seconds"]}), flush=True)
    report["all_exact"] = all(case["exact"] for case in report["cases"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_exact": report["all_exact"], "out": str(args.out)}), flush=True)
    if not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
