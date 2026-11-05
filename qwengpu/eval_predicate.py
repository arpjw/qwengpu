"""Check model-generated lane masks followed by masked vector addition."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import transformers

from qwengpu.compiled_add import (EXEC_ADD, build_standalone, compare_token,
                                  load_a, load_b, mask_token, out_token,
                                  read_lane)
from qwengpu.reference import LANES


@torch.no_grad()
def evaluate_case(model, a: tuple[int, ...], b: tuple[int, ...], threshold: int,
                  device: str) -> dict:
    start = time.monotonic()
    prefix = [token for lane in range(LANES)
              for token in (load_a(lane, a[lane]), load_b(lane, b[lane]))]
    prefix.append(compare_token(threshold))
    output = model(input_ids=torch.tensor([prefix], device=device), use_cache=True)
    cache = output.past_key_values
    generated_mask_token = int(output.logits[0, -1].argmax())
    expected_mask = sum(1 << lane for lane, value in enumerate(a) if value < threshold)
    mask_exact = generated_mask_token == mask_token(expected_mask)

    generated = []
    expected = []
    if mask_exact:
        for token in (generated_mask_token, EXEC_ADD):
            output = model(input_ids=torch.tensor([[token]], device=device),
                           past_key_values=cache, use_cache=True)
            cache = output.past_key_values
        for lane in range(LANES):
            output = model(input_ids=torch.tensor([[read_lane(lane)]], device=device),
                           past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            token = int(output.logits[0, -1].argmax())
            generated.append(token)
            expected.append(out_token(lane, (a[lane] + b[lane]) & 255
                                      if expected_mask >> lane & 1 else a[lane]))
            output = model(input_ids=torch.tensor([[token]], device=device),
                           past_key_values=cache, use_cache=True)
            cache = output.past_key_values
    return {"a": a, "b": b, "threshold": threshold,
            "generated_mask_token": generated_mask_token,
            "expected_mask_token": mask_token(expected_mask),
            "generated": generated, "expected": expected,
            "mask_exact": mask_exact, "exact": mask_exact and generated == expected,
            "seconds": round(time.monotonic() - start, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    random_source = random.Random(20261104)
    cases = [
        (tuple(range(LANES)), (10,) * LANES, 4),
        ((255, 0, 128, 127, 0, 1, 200, 199), (1,) * LANES, 128),
        ((100,) * LANES, (50,) * LANES, 0),
        ((100,) * LANES, (50,) * LANES, 255),
    ]
    cases.extend((tuple(random_source.randrange(256) for _ in range(LANES)),
                  tuple(random_source.randrange(256) for _ in range(LANES)),
                  random_source.randrange(256)) for _ in range(4))
    model, circuit = build_standalone()
    model.to(args.device)
    report = {"device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__, "circuit": circuit.stats(),
              "hidden": model.config.hidden_size,
              "layers": model.config.num_hidden_layers, "cases": []}
    for a, b, threshold in cases:
        result = evaluate_case(model, a, b, threshold, args.device)
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
