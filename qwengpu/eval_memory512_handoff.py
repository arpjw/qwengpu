"""Keep the expanded bank across native text between reduction launches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers

from qwengpu.eval_compiled_rom import head_row
from qwengpu.eval_globalgrid import launch
from qwengpu.eval_text_handoff import CORPUS, clear_text_history, compare, measure_text
from qwengpu.eval_tile8 import load_tile_model
from qwengpu.memory_geometry import MEMORY512_LAYOUT
from qwengpu.run_compute import MASK32, compile_request, parse_request, words32
from qwengpu.textcheck_9b import _ids, load_prompts

ROOT = Path(__file__).resolve().parents[1]


def ram_rows(cache, circuit) -> list[int]:
    return [
        head_row(cache, circuit, f"ram.{lane}.{nib}.{word // 128}", word % 128, 4)
        for word in range(256)
        for nib in range(4)
        for lane in range(8)
    ]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.model / "qwengpu9b.json").read_text())
    if meta["layout"] != MEMORY512_LAYOUT:
        parser.error("expected a memory512 checkpoint")
    manifest = ROOT / "eval/frozen/memory512_compute_v1/cases.json"
    payload = manifest.read_bytes()
    request = parse_request(json.loads(payload)["reduce_127"], memory512=True)
    producer_data, consumer_data = compile_request(request, memory512=True)
    model, circuit = load_tile_model(args.model, args.device, resident_memory512=True)
    offset = meta["chip_token_offset"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    ids = _ids(tokenizer, load_prompts(CORPUS)[0])
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    fresh = measure_text(model, ids, args.device, offset, eos)
    cache, producer = launch(model, circuit, producer_data, args.device, offset)
    before = ram_rows(cache, circuit)
    expected_rows = [
        word >> (4 * nib) & 15
        for word in producer["global_words"]
        for nib in range(4)
        for _lane in range(8)
    ]
    if before != expected_rows:
        raise ValueError("producer RAM rows differ from its decoded memory")
    clear_text_history(cache, model.config, meta["chip_layers"])
    after_gpu = measure_text(model, ids, args.device, offset, eos, cache=cache)
    text_comparison = compare(fresh, after_gpu)
    after = ram_rows(cache, circuit)
    if (
        before != after
        or text_comparison["next_max_abs_logit_difference"] != 0
        or not text_comparison["greedy_identical"]
    ):
        raise ValueError("native text changed RAM or failed its fresh-cache control")
    clear_text_history(cache, model.config, meta["chip_layers"])
    data = {**consumer_data, "global_words": producer["global_words"]}
    _, retained = launch(model, circuit, data, args.device, offset, cache=cache)
    _, cold = launch(model, circuit, data, args.device, offset)
    fields = retained.keys() - {"transport", "audit"}
    if any(retained[field] != cold[field] for field in fields):
        raise ValueError("retained consumer differs from the cold control")
    scalar = sum(request["values"]) & MASK32
    if (
        words32(retained["global_words"][:2])[0] != scalar
        or retained["transport"]["matrix_initial_tokens"] != 0
        or retained["transport"]["fresh_caches"] != 0
        or cold["transport"]["matrix_initial_tokens"] != 1024
    ):
        raise ValueError("reduction result or retained-memory transport failed")
    report = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "merge": meta,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "case": "reduce_127",
        "request": request,
        "prompt_index": 0,
        "prompt_ids": ids,
        "producer": producer,
        "ram_before": before,
        "ram_after_text": after,
        "text": {
            "fresh": fresh[1],
            "after_gpu": after_gpu[1],
            "comparison": text_comparison,
        },
        "consumer": retained,
        "cold_consumer": cold,
        "sum": scalar,
        "exact": True,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Expanded RAM: 8192 port nibbles preserved across native text; retained reduction exact"
    )


if __name__ == "__main__":
    main()
