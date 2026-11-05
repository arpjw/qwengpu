"""Free-running lane-banked scratch-memory checks on a Qwen3.5 model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import transformers

from qwengpu.compiled_add import (EXEC_ADD, build_standalone, load_a, load_b,
                                  load_token, mask_token, out_token, read_lane,
                                  store_token)
from qwengpu.reference import LANES


@torch.no_grad()
def evaluate_case(model, a: tuple[int, ...], slot: int, store_mask: int,
                  load_mask: int, device: str, offset: int = 0) -> dict:
    start = time.monotonic()
    prefix = [token for lane in range(LANES)
              for token in (load_a(lane, a[lane]), load_b(lane, 1))]
    prefix.append(mask_token(store_mask))
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

    def read(values: tuple[int, ...]) -> None:
        for lane, value in enumerate(values):
            token = feed(read_lane(lane))
            generated.append(token)
            expected.append(out_token(lane, value))
            feed(token)

    feed(store_token(slot))
    feed(mask_token(255))
    feed(EXEC_ADD)
    incremented = tuple((value + 1) & 255 for value in a)
    read(incremented)

    feed(mask_token(load_mask))
    feed(load_token(slot))
    restored = tuple((a[lane] if store_mask >> lane & 1 else 0)
                     if load_mask >> lane & 1 else incremented[lane]
                     for lane in range(LANES))
    read(restored)

    feed(mask_token(255))
    feed(EXEC_ADD)
    final = tuple((value + 1) & 255 for value in restored)
    read(final)
    return {"a": a, "slot": slot, "store_mask": store_mask,
            "load_mask": load_mask, "generated": generated,
            "expected": expected, "exact": generated == expected,
            "final_lanes": final, "seconds": round(time.monotonic() - start, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        (tuple(range(5, 13)), 2, 255, 85),
        ((255, 0, 128, 127, 0, 1, 200, 199), 0, 170, 255),
        ((100,) * 8, 3, 0, 255),
        ((250,) * 8, 1, 255, 0),
    ]
    model, circuit = build_standalone()
    model.to(args.device)
    report = {"device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__, "circuit": circuit.stats(),
              "hidden": model.config.hidden_size,
              "layers": model.config.num_hidden_layers, "cases": []}
    for a, slot, store_mask, load_mask in cases:
        result = evaluate_case(model, a, slot, store_mask, load_mask, args.device)
        report["cases"].append(result)
        print(json.dumps({"slot": slot, "store_mask": store_mask,
                          "load_mask": load_mask, "exact": result["exact"],
                          "seconds": result["seconds"]}), flush=True)
    report["all_exact"] = all(case["exact"] for case in report["cases"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_exact": report["all_exact"], "out": str(args.out)}), flush=True)
    if not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
