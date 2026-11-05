"""Check lane-banked scratch memory in a pretrained Qwen3.5-9B merge."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import transformers

from chipformer.q35.ir import Target

from qwengpu.compiled_add import build
from qwengpu.eval_memory import evaluate_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    meta = json.loads((args.model / "qwengpu9b.json").read_text(encoding="utf-8"))
    offset = int(meta["chip_token_offset"])
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to(args.device).eval()
    config = model.config
    circuit = build(Target(config.linear_num_key_heads,
                           config.linear_num_value_heads,
                           config.intermediate_size))
    cases = [
        (tuple(range(5, 13)), 2, 255, 85),
        ((255, 0, 128, 127, 0, 1, 200, 199), 0, 170, 255),
        ((100,) * 8, 3, 0, 255),
        ((250,) * 8, 1, 255, 0),
    ]
    random_source = random.Random(20261104)
    cases.extend((tuple(random_source.randrange(256) for _ in range(8)),
                  random_source.randrange(4), random_source.randrange(256),
                  random_source.randrange(256)) for _ in range(8))
    report = {"device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__, "merge": meta,
              "cases": []}
    for a, slot, store_mask, load_mask in cases:
        result = evaluate_case(model, a, slot, store_mask, load_mask,
                               args.device, offset=offset, circuit=circuit)
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
