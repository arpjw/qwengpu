"""Check retained-cache reduction and tiled matmul with stock generation."""

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
from qwengpu.run_compute import (
    MASK32, OPERAND_WORDS, TILE_K, TILE_M, TILE_N, compile_request, parse_request, words32,
)


def launch(raw):
    return parse_launch(
        {**raw, "workgroups": 2, "interleaved": True},
        resident_globalgrid=True, resident_compute=True, full_integer=True,
    )


def check_report(report):
    if not all(value for key, value in report.items() if key.endswith("_exact")):
        raise ValueError(f"generation replay differs at token {report['first_token_difference']}")


def run_reduction(model, circuit, device, offset):
    values = [0xFFFF0000 + 7919 * (i % 8) for i in range(32)]
    first, second = compile_request({"kernel": "reduce", "values": values}, memory512=True)
    cache, partial = evaluate(model, circuit, launch(first), device, offset)
    check_report(partial)
    second["global_words"] = partial["global_words"]
    _, result = evaluate(
        model, circuit, launch(second), device, offset, cache=cache, refresh_words=0,
    )
    check_report(result)
    assert result["global_upload_words"] == 0
    totals = words32(result["global_words"])
    assert totals[0] == sum(values) & MASK32
    return {"total": totals[0], "launches": [partial, result]}


def run_matrix(model, circuit, device, offset, request):
    a, b, initial = (request[key] for key in ("a", "b", "c"))
    m, k, n = len(a), len(b), len(b[0])
    actual = [row[:] for row in initial]
    records = []
    for row in range(0, m, TILE_M):
        for col in range(0, n, TILE_N):
            rows, cols = min(TILE_M, m - row), min(TILE_N, n - col)
            cache = None
            prior = None
            for begin in range(0, k, TILE_K):
                end = min(begin + TILE_K, k)
                tile = {
                    "kernel": "matmul",
                    "a": [line[begin:end] for line in a[row:row + rows]],
                    "b": [line[col:col + cols] for line in b[begin:end]],
                    "c": [line[col:col + cols] for line in initial[row:row + rows]],
                }
                (raw,) = compile_request(tile, memory512=True)
                if prior is not None:
                    raw["global_words"][OPERAND_WORDS:] = prior[OPERAND_WORDS:]
                cache, report = evaluate(
                    model, circuit, launch(raw), device, offset, cache=cache,
                    refresh_words=None if prior is None else OPERAND_WORDS,
                )
                check_report(report)
                if prior is not None:
                    assert report["global_upload_words"] == OPERAND_WORDS
                prior = report["global_words"]
                flat = words32(prior[64:64 + 2 * rows * cols])
                scalar = [
                    (initial[i][j] + sum(a[i][t] * b[t][j] for t in range(end))) & MASK32
                    for i in range(row, row + rows) for j in range(col, col + cols)
                ]
                assert flat == scalar
                records.append({"tile": [row, col], "panel": [begin, end], **report})
            for i in range(rows):
                actual[row + i][col:col + cols] = flat[i * cols:(i + 1) * cols]
    return {"output": actual, "launches": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("matrix", type=Path, nargs="*")
    args = parser.parse_args()
    meta = json.loads((args.model / "qwengpu9b.json").read_text())
    if meta["layout"] != LAYOUT:
        raise ValueError("expected a stock resident generation checkpoint")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(args.device).eval()
    cfg = model.config
    circuit = build(Target(cfg.linear_num_key_heads, cfg.linear_num_value_heads,
                           cfg.intermediate_size), generate=True)
    offset = meta["chip_token_offset"]
    report = {"device": args.device, "reduction": run_reduction(model, circuit, args.device, offset)}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for path in args.matrix:
        request = parse_request(json.loads(path.read_text()), memory512=True)
        if request["kernel"] != "matmul_tiled":
            raise ValueError("expected a tiled matrix fixture")
        report[path.stem] = run_matrix(model, circuit, args.device, offset, request)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
