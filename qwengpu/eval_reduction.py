"""Run a two-launch, 16-value modular reduction in one tilegrid cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers

from qwengpu.assemble import assemble
from qwengpu.compiled_tile8 import RESIDENT_TILEGRID_LAYOUT, TILEGRID_VOCAB
from qwengpu.eval_tile import _unsigned
from qwengpu.eval_tile8 import launch_cached_tile, load_tile_model, oracle, parse_launch

FIXTURE = (
    Path(__file__).resolve().parents[1] / "eval/frozen/tilegrid_reduction_v1/words.json"
)
WORD_MASK = 0xFFFFFFFF
GUARDS = [0xDEA0 + index for index in range(32)]


def launch_data(words: list[int], masks: tuple[int, int]) -> dict:
    instructions = ["SAVE_INDEX", "LOAD_INDEX", "LOAD_INDEX_HI"]
    for mask in masks:
        instructions.extend(
            [
                f"ADD_INDEX_LO {mask}",
                f"ADD_INDEX_HI {mask}",
                "BARRIER",
                "STORE_INDEX",
                "STORE_INDEX_HI",
                "BARRIER",
            ]
        )
    body = "\n".join([*instructions, "HALT"])
    return assemble(
        f".warp 0\n{body}\n.warp 1\n{body}\n",
        {
            "c": words,
            "row_pairs": 1,
            "initial_registers": [
                list(range(start, start + 16, 2)) for start in (64, 80)
            ],
        },
    )


def reduce(model, circuit, values: list[int], device: str, offset: int = 0) -> dict:
    words = [part for value in values for part in (value & 65535, value >> 16)] + GUARDS
    cache = None
    passes = []
    for masks, width in (((2, 4), 4), ((8, 16), 16)):
        data = launch_data(words, masks)
        launch = parse_launch(data, resident_loop32=True)
        expected = oracle(launch, 64, looping=True)
        previous = cache
        cache, actual, audit = launch_cached_tile(
            model, circuit, launch, device, offset, cache=cache, retain_operands=True
        )
        transport = actual.pop("transport")
        if (
            actual != expected
            or actual["outcome"] != "halted"
            or not all(
                transport[f"resident_{name}_exact"]
                for name in ("grid", "memory", "rom", "state")
            )
        ):
            raise ValueError(f"reduction pass {masks} differs from the reference")
        if previous is not None and (
            cache is not previous or transport["matrix_initial_tokens"]
        ):
            raise ValueError("the second pass reloaded memory or replaced the cache")
        words = actual["shared_words"][64:]
        partial = [words[i] | words[i + 1] << 16 for i in range(0, 32, 2)]
        scalar = [
            (sum(values[i - i % width : i - i % width + width]) & WORD_MASK)
            for i in range(len(values))
        ]
        if partial != scalar or words[32:] != GUARDS:
            raise ValueError(
                f"reduction pass {masks} differs from scalar sums or guards"
            )
        passes.append(
            {
                "masks": masks,
                "programs": data["programs"],
                "partial_sums": partial,
                "instructions": len(actual["trace"]),
                "decisions": len(actual["schedules"]),
                "transport": transport,
                "memory_tokens_by_call": audit,
                "trace": actual["trace"],
                "schedules": actual["schedules"],
            }
        )
    return {"sum": partial[0], "passes": passes, "guards": words[32:], "exact": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=FIXTURE)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = args.input.read_bytes()
    values = _unsigned(json.loads(payload), 16, WORD_MASK, "input")
    meta = (
        json.loads((args.model / "qwengpu9b.json").read_text()) if args.model else None
    )
    if meta is not None and (
        meta.get("layout") != RESIDENT_TILEGRID_LAYOUT
        or meta.get("chip_vocab") != TILEGRID_VOCAB
    ):
        parser.error("expected a merged tilegrid checkpoint")
    model, circuit = load_tile_model(
        args.model, args.device, resident_add32=True, resident_tilegrid=True
    )
    result = reduce(
        model, circuit, values, args.device, meta["chip_token_offset"] if meta else 0
    )
    report = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "merge": meta,
        "input": values,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        **result,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Two passes exact on {args.device}, sum = 0x{result['sum']:08x}")


if __name__ == "__main__":
    main()
