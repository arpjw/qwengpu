"""Qualify variable-size compute requests on one fixed checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers

from qwengpu.compiled_tile8 import RESIDENT_COMPUTE_LAYOUT, TILEGRID_VOCAB
from qwengpu.compiled_tile8 import ROM32_VOCAB
from qwengpu.eval_tile8 import load_tile_model
from qwengpu.memory_geometry import MEMORY512_LAYOUTS, ROM32_LAYOUT, WIDE_MEMORY
from qwengpu.run_compute import execute_request, parse_request

FIXTURE = Path(__file__).resolve().parents[1] / "eval/frozen/compute_v1/cases.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=FIXTURE)
    parser.add_argument("--case", action="append")
    parser.add_argument("--model", type=Path)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--resident-memory512", action="store_true")
    variants.add_argument("--resident-rom32", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = args.cases.read_bytes()
    raw = json.loads(payload)
    selected = args.case or list(raw)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(name not in raw for name in selected)
    ):
        parser.error("expected unique case names from the input manifest")
    meta = (
        json.loads((args.model / "qwengpu9b.json").read_text()) if args.model else None
    )
    memory512 = (
        args.resident_memory512
        or args.resident_rom32
        or (meta is not None and meta.get("layout") in MEMORY512_LAYOUTS)
    )
    requests = [
        (name, parse_request(raw[name], memory512=memory512)) for name in selected
    ]
    if meta is not None and (
        meta.get("layout")
        not in (MEMORY512_LAYOUTS if memory512 else (RESIDENT_COMPUTE_LAYOUT,))
        or meta.get("chip_vocab")
        != (
            ROM32_VOCAB
            if meta.get("layout") == ROM32_LAYOUT
            else WIDE_MEMORY.vocab
            if memory512
            else TILEGRID_VOCAB
        )
    ):
        parser.error("expected a merged compute checkpoint")
    model, circuit = load_tile_model(
        args.model,
        args.device,
        resident_compute=True,
        resident_memory512=memory512,
        resident_rom32=args.resident_rom32,
    )
    records = []
    for name, request in requests:
        result = execute_request(
            model,
            circuit,
            request,
            args.device,
            meta["chip_token_offset"] if meta else 0,
        )
        records.append({"name": name, "request": request, **result})
        print(json.dumps({"name": name, "exact": result["exact"]}), flush=True)
    report = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "merge": meta,
        "circuit": circuit.stats(),
        "fixture_sha256": hashlib.sha256(payload).hexdigest(),
        "cases": records,
        "all_exact": all(record["exact"] for record in records),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
