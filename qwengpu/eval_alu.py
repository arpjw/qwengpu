"""Check masked XOR followed by ADD with generated-token writeback."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import transformers

from chipformer.q35.ir import Target

from qwengpu.compiled_add import (EXEC_ADD, EXEC_XOR, build, build_standalone,
                                  load_a, load_b, mask_token, out_token, read_lane)
from qwengpu.reference import LANES


@torch.no_grad()
def evaluate_case(model, a: tuple[int, ...], b: tuple[int, ...], mask: int,
                  device: str, offset: int = 0) -> dict:
    start = time.monotonic()
    prefix = [token for lane in range(LANES)
              for token in (load_a(lane, a[lane]), load_b(lane, b[lane]))]
    prefix.append(mask_token(mask))
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

    feed(EXEC_XOR)
    xored = tuple(a[lane] ^ b[lane] if mask >> lane & 1 else a[lane]
                  for lane in range(LANES))
    read(xored)
    feed(EXEC_ADD)
    added = tuple((xored[lane] + b[lane]) & 255 if mask >> lane & 1
                  else xored[lane] for lane in range(LANES))
    read(added)
    return {"a": a, "b": b, "mask": mask, "generated": generated,
            "expected": expected, "exact": generated == expected,
            "final_lanes": added, "seconds": round(time.monotonic() - start, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    offset = 0
    if args.model:
        meta = json.loads((args.model / "qwengpu9b.json").read_text(encoding="utf-8"))
        offset = int(meta["chip_token_offset"])
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16).to(args.device).eval()
        config = model.config
        circuit = build(Target(config.linear_num_key_heads,
                               config.linear_num_value_heads,
                               config.intermediate_size))
    else:
        model, circuit = build_standalone()
        model.to(args.device)
        meta = None
    cases = [
        ((255, 0, 128, 127, 42, 213, 5, 250),
         (1, 255, 128, 127, 85, 85, 17, 11), 0b10110101),
        (tuple(range(8)), (17,) * 8, 255),
        ((100,) * 8, (50,) * 8, 0),
        ((250,) * 8, (20,) * 8, 0b01010101),
    ]
    if args.model:
        rng = random.Random(20261104)
        cases.extend((tuple(rng.randrange(256) for _ in range(LANES)),
                      tuple(rng.randrange(256) for _ in range(LANES)),
                      rng.randrange(256)) for _ in range(8))
    report = {"device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__, "circuit": circuit.stats(),
              "merge": meta, "cases": []}
    for a, b, mask in cases:
        result = evaluate_case(model, a, b, mask, args.device, offset)
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
