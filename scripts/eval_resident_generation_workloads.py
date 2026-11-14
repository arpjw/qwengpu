"""Check compute workloads with stock resident generation and retained RAM."""

import argparse
import json
from pathlib import Path

import torch
from chipformer.q35.ir import Target
from transformers import AutoModelForCausalLM

from qwengpu.compiled_tile8 import build
from qwengpu.eval_resident_generation import evaluate
from qwengpu.eval_tile8 import parse_launch
from qwengpu.resident_generation import LAYOUT
from qwengpu.run_compute import execute_request, parse_request


def launch(model, circuit, raw, device, offset, cache, *, transport_mode,
           refresh_global_prefix=0):
    if transport_mode != "stock_generation":
        raise ValueError("expected stock generation transport")
    case = parse_launch(
        {**raw, "interleaved": raw["workgroups"] == 2},
        resident_globalgrid=True, resident_compute=True, full_integer=True,
    )
    retained, report = evaluate(
        model, circuit, case, device, offset, cache=cache,
        refresh_words=None if cache is None else refresh_global_prefix,
    )
    failures = [key for key, value in report.items() if key.endswith("_exact") and not value]
    if failures:
        raise ValueError(f"generation replay differs at token {report['first_token_difference']}: {failures}")
    return retained, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suite", type=Path, help="JSON map from case names to compute requests")
    parser.add_argument("--case", action="append", help="run a named request from the suite")
    parser.add_argument("matrix", type=Path, nargs="*")
    args = parser.parse_intermixed_args()
    meta = json.loads((args.model / "qwengpu9b.json").read_text())
    if meta["layout"] != LAYOUT:
        raise ValueError("expected a stock resident generation checkpoint")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(args.device).eval()
    cfg = model.config
    circuit = build(Target(cfg.linear_num_key_heads, cfg.linear_num_value_heads,
                           cfg.intermediate_size), generate=True)
    offset = meta["chip_token_offset"]
    requests = (
        json.loads(args.suite.read_text()) if args.suite else
        {"reduction": {"kernel": "reduce",
                       "values": [0xFFFF0000 + 7919 * (i % 8) for i in range(32)]}}
    )
    requests.update({path.stem: json.loads(path.read_text()) for path in args.matrix})
    if args.case:
        missing = set(args.case) - requests.keys()
        if missing:
            parser.error(f"unknown cases: {', '.join(sorted(missing))}")
        requests = {name: requests[name] for name in args.case}
    report = {"device": args.device, "cases": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, data in requests.items():
        request = parse_request(data, memory512=True)
        report["cases"][name] = execute_request(
            model, circuit, request, args.device, offset,
            transport_mode="stock_generation", launcher=launch,
        )
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{name}: exact", flush=True)


if __name__ == "__main__":
    main()
