"""Verify GPU-to-native-text-to-GPU handoff with stock resident generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from chipformer.q35.ir import Target

from qwengpu.cache_register_edge import register_head_names
from qwengpu.compiled_tile8 import ROM_PAYLOAD_BITS, build
from qwengpu.eval_compiled_rom import head_row
from qwengpu.eval_memory512_handoff import ram_rows
from qwengpu.eval_resident_generation import evaluate
from qwengpu.eval_text_handoff import CORPUS, PROMPT_INDICES, clear_text_history, compare, measure_text
from qwengpu.eval_tile8 import parse_launch
from qwengpu.resident_generation import LAYOUTS, TILEGRID_LAYOUT
from qwengpu.run_compute import MASK32, compile_request, parse_request, words32
from qwengpu.textcheck_9b import _ids, load_prompts


def parse_interleaved(launch: dict) -> dict:
    return parse_launch(
        {**launch, "workgroups": 2, "interleaved": True},
        resident_globalgrid=True, resident_compute=True, full_integer=True,
    )


def gpu_rows(cache, circuit) -> dict:
    return {
        "global": ram_rows(cache, circuit),
        "registers": [
            head_row(cache, circuit, name, warp, 4)
            for warp in range(4)
            for name in register_head_names()
        ],
        "control": [
            head_row(cache, circuit, f"resident.control.{warp}", 0, 7)
            for warp in range(4)
        ],
        "rom": [
            head_row(cache, circuit, "rom", address, ROM_PAYLOAD_BITS + 3)
            for address in range(64)
        ],
    }


def digest(rows: dict) -> str:
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.model / "qwengpu9b.json").read_text())
    if meta["layout"] not in LAYOUTS:
        parser.error("expected a stock resident generation checkpoint")
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(args.device).eval()
    config = model.config
    circuit = build(Target(config.linear_num_key_heads, config.linear_num_value_heads,
                           config.intermediate_size), generate=True,
                    generated_tilegrid=meta["layout"] == TILEGRID_LAYOUT)
    offset = meta["chip_token_offset"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])

    manifest = CORPUS.parents[1] / "memory512_compute_v1" / "cases.json"
    payload = manifest.read_bytes()
    request = parse_request({"kernel": "reduce", "values": json.loads(payload)["reduce_127"]["values"][:32]},
                            memory512=True)
    producer_data, consumer_data = compile_request(request, memory512=True)
    cache, producer = evaluate(model, circuit, parse_interleaved(producer_data), args.device, offset)
    before = gpu_rows(cache, circuit)
    expected_global = [
        word >> (4 * nibble) & 15
        for word in producer["global_words"]
        for nibble in range(4)
        for _ in range(8)
    ]
    if before["global"] != expected_global or any(
        value is None for values in before.values() for value in values
    ):
        raise ValueError("producer GPU rows are incomplete or differ from decoded memory")

    records = []
    corpus = load_prompts(CORPUS)
    for index in PROMPT_INDICES:
        ids = _ids(tokenizer, corpus[index])
        fresh = measure_text(model, ids, args.device, offset, eos)
        clear_text_history(cache, config, meta["chip_layers"])
        resumed = measure_text(model, ids, args.device, offset, eos, cache=cache)
        after = gpu_rows(cache, circuit)
        comparison = compare(fresh, resumed)
        records.append({
            "prompt_index": index,
            "prompt_ids": ids,
            "fresh": fresh[1],
            "after_gpu": resumed[1],
            "comparison": comparison,
            "gpu_rows_exact": after == before,
            "gpu_rows_sha256": digest(after),
        })
        print(f"Text prompt {index}: state preserved={after == before}, "
              f"greedy exact={comparison['greedy_identical']}", flush=True)

    clear_text_history(cache, config, meta["chip_layers"])
    consumer_case = parse_interleaved({**consumer_data, "global_words": producer["global_words"]})
    _, consumer = evaluate(model, circuit, consumer_case, args.device, offset, cache=cache, refresh_words=0)
    _, cold = evaluate(model, circuit, consumer_case, args.device, offset)
    scalar = sum(request["values"]) & MASK32
    exact = (
        all(value for report in (producer, consumer, cold)
            for key, value in report.items() if key.endswith("_exact"))
        and all(record["gpu_rows_exact"] and record["comparison"]["greedy_identical"]
                and record["comparison"]["next_max_abs_logit_difference"] == 0
                and not record["after_gpu"]["chip_tokens"] for record in records)
        and consumer["generated_tokens"] == cold["generated_tokens"]
        and consumer["global_words"] == cold["global_words"]
        and words32(consumer["global_words"][:2]) == [scalar]
        and consumer["global_upload_words"] == 0
    )
    report = {
        "device": args.device, "torch": torch.__version__, "transformers": transformers.__version__,
        "checkpoint": meta, "source_cases_sha256": hashlib.sha256(payload).hexdigest(),
        "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "request": request, "gpu_rows_before": before, "gpu_rows_sha256": digest(before),
        "native_text": records, "producer": producer, "consumer": consumer, "cold_consumer": cold,
        "sum": scalar, "exact": exact,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Native text across {len(records)} prompts and retained reduction: exact={exact}", flush=True)
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
