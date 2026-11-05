"""Free-running exactness check of the standalone Qwen3.5 masked-add model."""

from __future__ import annotations

import argparse
import json
import random
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
    prefix.extend((mask_token(mask), EXEC_ADD))
    inputs = torch.tensor([prefix], device=device)
    output = model(input_ids=inputs, use_cache=True)
    cache = output.past_key_values
    observed = []
    expected = []
    for lane in range(LANES):
        query = torch.tensor([[read_lane(lane)]], device=device)
        output = model(input_ids=query, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        token = int(output.logits[0, -1].argmax())
        observed.append(token)
        expected.append(out_token(lane, (a[lane] + b[lane]) & 255
                                  if mask >> lane & 1 else a[lane]))
        response = torch.tensor([[token]], device=device)
        output = model(input_ids=response, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
    return {"a": a, "b": b, "mask": mask, "observed": observed,
            "expected": expected, "exact": observed == expected,
            "seconds": round(time.monotonic() - start, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    random_source = random.Random(4169)
    cases = [
        (tuple(range(LANES)), (10,) * LANES, 255),
        ((255, 0, 1, 2, 3, 4, 5, 6), (1, 255, 254, 200, 1, 2, 3, 4), 85),
        ((100,) * LANES, (50,) * LANES, 0),
    ]
    cases.extend((tuple(random_source.randrange(256) for _ in range(LANES)),
                  tuple(random_source.randrange(256) for _ in range(LANES)),
                  random_source.randrange(256)) for _ in range(5))
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
