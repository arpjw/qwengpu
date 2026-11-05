"""Baseline the frozen ROM-gate kernels on the host-dispatched circuit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers

from qwengpu.compiled_add import build_standalone
from qwengpu.eval_program import evaluate_case
from qwengpu.run_kernel import parse_kernel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "eval/frozen/rom_gate_v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                        else "cpu")
    args = parser.parse_args()
    paths = sorted(args.cases_dir.glob("*.json"))
    if len(paths) != 4:
        raise ValueError("expected exactly four frozen ROM-gate kernels")
    model, circuit = build_standalone()
    model.to(args.device)
    report = {"device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__, "circuit": circuit.stats(),
              "cases": []}
    for path in paths:
        payload = path.read_bytes()
        initial, program = parse_kernel(json.loads(payload))
        result = evaluate_case(model, circuit, initial, program,
                               args.device, probe_final=True)
        result["fixture"] = path.name
        result["fixture_sha256"] = hashlib.sha256(payload).hexdigest()
        report["cases"].append(result)
        print(json.dumps({"fixture": path.name, "exact": result["exact"],
                          "seconds": result["seconds"]}), flush=True)
    report["all_exact"] = all(case["exact"] for case in report["cases"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_exact": report["all_exact"], "out": str(args.out)}), flush=True)
    if not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
