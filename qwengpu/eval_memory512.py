"""Qualify banked memory and larger kernels on one loaded checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers

from qwengpu.eval_globalgrid import launch
from qwengpu.eval_tile8 import load_tile_model
from qwengpu.memory_geometry import MEMORY512_LAYOUT, WIDE_MEMORY
from qwengpu.run_compute import execute_request, parse_request

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--raw", action="append")
    parser.add_argument("--case", action="append")
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = (
        json.loads((args.model / "qwengpu9b.json").read_text()) if args.model else None
    )
    if meta and (
        meta["layout"] != MEMORY512_LAYOUT or meta["chip_vocab"] != WIDE_MEMORY.vocab
    ):
        parser.error("expected a memory512 checkpoint")
    paths = sorted((ROOT / "eval/frozen/memory512_v1").glob("*.json"))
    if args.raw:
        paths = [path for path in paths if path.stem in args.raw]
        if len(paths) != len(set(args.raw)):
            parser.error("unknown or duplicate raw memory case")
    manifest = ROOT / "eval/frozen/memory512_compute_v1/cases.json"
    payload = manifest.read_bytes()
    inputs = json.loads(payload)
    names = [] if args.raw_only else (args.case or list(inputs))
    if len(set(names)) != len(names) or any(name not in inputs for name in names):
        parser.error("unknown or duplicate compute case")
    model, circuit = load_tile_model(args.model, args.device, resident_memory512=True)
    offset = meta["chip_token_offset"] if meta else 0
    raw = []
    for path in paths:
        data = path.read_bytes()
        _, result = launch(model, circuit, json.loads(data), args.device, offset)
        raw.append(
            {
                "name": path.stem,
                "fixture_sha256": hashlib.sha256(data).hexdigest(),
                **result,
            }
        )
        print(json.dumps({"raw": path.stem, "exact": True}), flush=True)
    cases = []
    for name in names:
        request = parse_request(inputs[name], memory512=True)
        result = execute_request(model, circuit, request, args.device, offset)
        cases.append({"name": name, "request": request, **result})
        print(json.dumps({"case": name, "exact": True}), flush=True)
    report = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "merge": meta,
        "circuit": circuit.stats(),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "raw": raw,
        "cases": cases,
        "all_exact": True,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
