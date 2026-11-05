"""Check one-level SIMT divergence in a pretrained Qwen3.5-9B merge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers

from qwengpu.eval_branch import evaluate_case


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
    cases = [
        (tuple(range(8)), 4, 10, 20, 1),
        ((255, 0, 128, 127, 0, 1, 200, 199), 128, 1, 250, 7),
    ]
    report = {"device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__, "merge": meta,
              "cases": []}
    for a, threshold, then_add, else_add, join_add in cases:
        result = evaluate_case(model, a, threshold, then_add, else_add,
                               join_add, args.device, offset=offset)
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
