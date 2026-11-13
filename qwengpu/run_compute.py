"""Run bounded vector, reduction, and uint8 matrix kernels in one global bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from qwengpu.assemble import assemble
from qwengpu.alu32 import ALU_VOCAB
from qwengpu.compiled_tile8 import RESIDENT_COMPUTE_LAYOUT, TILEGRID_VOCAB
from qwengpu.eval_globalgrid import launch
from qwengpu.eval_tile import _unsigned
from qwengpu.compiled_tile8 import ROM32_VOCAB
from qwengpu.eval_tile8 import load_tile_model
from qwengpu.memory_geometry import (
    ALU32_LAYOUTS,
    MEMORY512_LAYOUTS,
    ROM32_LAYOUTS,
    WIDE_MEMORY,
)

MASK32 = 0xFFFFFFFF
GUARD = 0xD00D0000
TILE_M, TILE_K, TILE_N = 8, 4, 4
OPERAND_WORDS = TILE_M * TILE_K + TILE_K * TILE_N
TILED_LIMIT = 16


def words16(values: list[int]) -> list[int]:
    return [part for value in values for part in (value & 65535, value >> 16)]


def words32(words: list[int]) -> list[int]:
    return [lo | hi << 16 for lo, hi in zip(words[::2], words[1::2], strict=True)]


def vector(values: object, name: str, limit: int = 32) -> list[int]:
    if not isinstance(values, list) or len(values) > limit:
        raise ValueError(f"{name} must contain 0 through {limit} words")
    return _unsigned(values, len(values), MASK32, name)


def matrix(
    values: object, rows: int | None, columns: int | None, maximum: int, name: str
) -> list[list[int]]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty matrix")
    if rows is not None and len(values) != rows:
        raise ValueError(f"{name} has the wrong row count")
    width = columns
    if width is None:
        if not isinstance(values[0], list) or not values[0]:
            raise ValueError(f"{name} must have at least one column")
        width = len(values[0])
    return [
        _unsigned(row, width, maximum, f"{name}[{i}]") for i, row in enumerate(values)
    ]


def parse_request(data: object, *, memory512: bool = False) -> dict:
    if not isinstance(data, dict):
        raise TypeError("compute request must be an object")
    kernel = data.get("kernel")
    if kernel == "vector_add":
        limit = 64 if memory512 else 32
        a, b = vector(data.get("a"), "a", limit), vector(data.get("b"), "b", limit)
        if len(a) != len(b):
            raise ValueError("vector operands must have equal length")
        return {"kernel": kernel, "a": a, "b": b}
    if kernel == "reduce":
        return {
            "kernel": kernel,
            "values": vector(data.get("values"), "values", 128 if memory512 else 64),
        }
    if kernel in ("matmul", "matmul_tiled"):
        a = matrix(data.get("a"), None, None, 255, "a")
        m, k = len(a), len(a[0])
        b = matrix(data.get("b"), k, None, 255, "b")
        n = len(b[0])
        if kernel == "matmul_tiled" and not memory512:
            raise ValueError("matmul_tiled requires the 512-byte layout")
        limits = (TILED_LIMIT,) * 3 if kernel == "matmul_tiled" else (TILE_M, TILE_K, TILE_N)
        if any(size > limit for size, limit in zip((m, k, n), limits, strict=True)):
            raise ValueError(f"matrix limits are M <= {limits[0]}, K <= {limits[1]}, and N <= {limits[2]}")
        c = matrix(data.get("c", [[0] * n for _ in range(m)]), m, n, MASK32, "c")
        return {"kernel": kernel, "a": a, "b": b, "c": c}
    raise ValueError("kernel must be vector_add, reduce, matmul, or matmul_tiled")


def assemble_launch(
    body: list[str], words: list[int], groups: int, second: list[str] | None = None
) -> dict:
    source = (
        ".warp 0\n"
        + "\n".join(body)
        + "\n.warp 1\n"
        + "\n".join(second if second is not None else body)
    )
    return assemble(source, {"workgroups": groups, "global_words": words})


def compile_request(request: dict, *, memory512: bool = False) -> list[dict]:
    kind = request["kernel"]
    if kind == "matmul":
        a, b, c = request["a"], request["b"], request["c"]
        m, k, n = len(a), len(b), len(b[0])
        left = [value for row in a for value in row]
        right = [value for row in b for value in row]
        initial = [value for row in c for value in row]
        words = left + [0xAB00 + i for i in range(32 - len(left))]
        words += right + [0xBC00 + i for i in range(16 - len(right))]
        words += [0xCD00 + i for i in range(16)]
        words += words16(initial + [GUARD + i for i in range(32 - len(initial))])
        if memory512:
            words += [0xBEEF] * 128
        body = [
            "GLOBAL_ID",
            "SHL32I 1",
            "ADD32I 64",
            "SAVE_INDEX",
            f"MASK_GLOBAL_LT {m * n}",
            "LOAD_INDEX",
            "LOAD_INDEX_HI",
        ]
        body += [f"MATRIX_MAC8 {(k - 1) * 16 + (n - 1) * 4 + pos}" for pos in range(k)]
        body += ["STORE_INDEX", "STORE_INDEX_HI", "HALT"]
        return [assemble_launch(body, words, (m * n + 15) // 16)]
    if kind != "vector_add" and kind != "reduce":
        raise ValueError(f"{kind} is not a single compute program")
    values = request["a"] if kind == "vector_add" else request["values"]
    count = len(values)
    threads = min(count, 64)
    groups = max(1, (threads + 15) // 16)
    prefix = [
        "GLOBAL_ID",
        "SHL32I 1",
        "SAVE_INDEX",
        f"MASK_GLOBAL_LT {threads}",
        "LOAD_INDEX",
        "LOAD_INDEX_HI",
    ]
    if kind == "vector_add":
        capacity = 64 if memory512 else 32
        left = values + [GUARD + i for i in range(capacity - count)]
        right = request["b"] + [GUARD + capacity + i for i in range(capacity - count)]
        words = words16(left + right)
        body = prefix + [
            f"ADD_INDEX_LO {2 * capacity}",
            f"ADD_INDEX_HI {2 * capacity}",
            "STORE_INDEX",
            "STORE_INDEX_HI",
            "HALT",
        ]
        return [assemble_launch(body, words, groups)]
    padded = max(32, count)
    words = words16(
        values
        + [0] * (padded - count)
        + [GUARD + i for i in range((128 if memory512 else 64) - padded)]
    )
    if count > 64:
        prefix += [
            f"MASK_GLOBAL_LT {count - 64}",
            "ADD_INDEX_LO 128",
            "ADD_INDEX_HI 128",
            "MASK_ALL",
        ]
    producer = prefix + [
        "WARP_SUM",
        "MASK_LANES 1",
        "STORE_INDEX",
        "STORE_INDEX_HI",
        "HALT",
    ]
    consumer = [
        "LANE_ID",
        "SHL32I 4",
        "SAVE_INDEX",
        f"MASK_LANES {(threads + 7) // 8}",
        "LOAD_INDEX",
        "LOAD_INDEX_HI",
        "WARP_SUM",
        "MASK_LANES 1",
        "STORE_INDEX",
        "STORE_INDEX_HI",
        "HALT",
    ]
    return [
        assemble_launch(producer, words, groups),
        assemble_launch(consumer, words, 1, ["HALT"]),
    ]


def execute_tiled_matmul(model, circuit, request, device, offset, transport_mode):
    """Keep each output tile on one cache and refresh only A/B between panels."""
    a, b, c = (request[name] for name in ("a", "b", "c"))
    m, k, n = len(a), len(b), len(b[0])
    output = [row[:] for row in c]
    records = []
    final_tiles = []
    for row in range(0, m, TILE_M):
        for col in range(0, n, TILE_N):
            rows, cols = min(TILE_M, m - row), min(TILE_N, n - col)
            initial = [values[col : col + cols] for values in c[row : row + rows]]
            cache = None
            words = None
            for start in range(0, k, TILE_K):
                end = min(start + TILE_K, k)
                tile = {
                    "kernel": "matmul",
                    "a": [values[start:end] for values in a[row : row + rows]],
                    "b": [values[col : col + cols] for values in b[start:end]],
                    "c": initial,
                }
                (program,) = compile_request(tile, memory512=True)
                refresh = 0 if words is None else OPERAND_WORDS
                if refresh:
                    program["global_words"][refresh:] = words[refresh:]
                cache, observed = launch(
                    model,
                    circuit,
                    program,
                    device,
                    offset,
                    cache,
                    transport_mode=transport_mode,
                    refresh_global_prefix=refresh,
                )
                words = observed["global_words"]
                expected = program["global_words"][:]
                scalar = [
                    (c[i][j] + sum(a[i][t] * b[t][j] for t in range(end))) & MASK32
                    for i in range(row, row + rows)
                    for j in range(col, col + cols)
                ]
                expected[64 : 64 + 2 * len(scalar)] = words16(scalar)
                if words != expected:
                    raise ValueError(f"matrix tile ({row}, {col}) panel {start} differs from scalar")
                records.append({"tile": [row, col], "k_panel": [start, end], **observed})
            flat = words32(words[64 : 64 + 2 * rows * cols])
            for i in range(rows):
                output[row + i][col : col + cols] = flat[i * cols : (i + 1) * cols]
            final_tiles.append({"tile": [row, col], "words": words})
    return {
        "kernel": "matmul_tiled",
        "output": output,
        "launches": records,
        "global_words": final_tiles,
        "exact": True,
    }


def execute_request(
    model,
    circuit,
    request: dict,
    device: str,
    offset: int = 0,
    *,
    transport_mode: str = "state_store_plan",
) -> dict:
    if request["kernel"] == "matmul_tiled":
        if not circuit.resident_memory512:
            raise ValueError("matmul_tiled requires the 512-byte layout")
        return execute_tiled_matmul(model, circuit, request, device, offset, transport_mode)
    launches = compile_request(request, memory512=circuit.resident_memory512)
    cache = None
    records = []
    words = launches[0]["global_words"][:]
    for data in launches:
        cache, observed = launch(
            model,
            circuit,
            {**data, "global_words": words},
            device,
            offset,
            cache,
            transport_mode=transport_mode,
        )
        words = observed["global_words"]
        records.append(observed)
    expected = launches[0]["global_words"][:]
    kind = request["kernel"]
    if kind == "vector_add":
        result = words32(words)[: len(request["a"])]
        scalar = [
            (a + b) & MASK32 for a, b in zip(request["a"], request["b"], strict=True)
        ]
        expected[: 2 * len(scalar)] = words16(scalar)
    elif kind == "reduce":
        result = words32(words[:2])[0]
        scalar = sum(request["values"]) & MASK32
        for start in range(0, min(len(request["values"]), 64), 8):
            subtotal = (
                sum(request["values"][start : start + 8])
                + sum(request["values"][64 + start : 72 + start])
            ) & MASK32
            expected[2 * start : 2 * start + 2] = words16([subtotal])
        expected[:2] = words16([scalar])
    else:
        a, b, c = request["a"], request["b"], request["c"]
        m, k, n = len(a), len(b), len(b[0])
        flat = words32(words[64 : 64 + 2 * m * n])
        result = [flat[i * n : (i + 1) * n] for i in range(m)]
        scalar = [
            [
                (c[i][j] + sum(a[i][t] * b[t][j] for t in range(k))) & MASK32
                for j in range(n)
            ]
            for i in range(m)
        ]
        expected[64 : 64 + 2 * m * n] = words16([v for row in scalar for v in row])
    if result != scalar or words != expected:
        raise ValueError(
            "kernel result or untouched global memory differs from scalar reference"
        )
    return {
        "kernel": kind,
        "output": result,
        "launches": records,
        "global_words": words,
        "exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--resident-memory512", action="store_true")
    variants.add_argument("--resident-rom32", action="store_true")
    variants.add_argument("--resident-alu32", action="store_true")
    variants.add_argument("--resident-scoped", action="store_true")
    variants.add_argument("--resident-interleaved", action="store_true")
    variants.add_argument("--full-integer", action="store_true")
    parser.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"))
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--transport", choices=("fixed", "state-store-plan"), default="state-store-plan"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    meta = (
        json.loads((args.model / "qwengpu9b.json").read_text()) if args.model else None
    )
    memory512 = (
        args.resident_memory512
        or args.resident_rom32
        or args.resident_alu32
        or args.resident_scoped
        or args.resident_interleaved
        or args.full_integer
        or (meta is not None and meta.get("layout") in MEMORY512_LAYOUTS)
    )
    try:
        payload = args.input.read_bytes()
        request = parse_request(json.loads(payload), memory512=memory512)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    meta = (
        json.loads((args.model / "qwengpu9b.json").read_text()) if args.model else None
    )
    if meta is not None and (
        meta.get("layout")
        not in (MEMORY512_LAYOUTS if memory512 else (RESIDENT_COMPUTE_LAYOUT,))
        or meta.get("chip_vocab")
        != (
            ALU_VOCAB
            if meta.get("layout") in ALU32_LAYOUTS
            else ROM32_VOCAB
            if meta.get("layout") in ROM32_LAYOUTS
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
        resident_alu32=args.resident_alu32,
        resident_scoped=args.resident_scoped,
        resident_interleaved=args.resident_interleaved,
        full_integer=args.full_integer,
    )
    result = execute_request(
        model,
        circuit,
        request,
        args.device,
        meta["chip_token_offset"] if meta else 0,
        transport_mode=args.transport.replace("-", "_"),
    )
    if not args.trace:
        result.pop("global_words")
        result["launches"] = [
            {
                **({"tile": run["tile"], "k_panel": run["k_panel"]} if "tile" in run else {}),
                "instructions": len(run["trace"]),
                "decisions": len(run["schedules"]),
                "transport": run["transport"],
            }
            for run in result["launches"]
        ]
    report = {
        "format": "qwengpu-compute-v1",
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "device": args.device,
        "merge": meta,
        "torch": torch.__version__,
        "transport_mode": args.transport,
        **result,
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
