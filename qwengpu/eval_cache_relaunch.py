"""Verify a GPU-to-text-to-GPU round trip with resident shared memory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import transformers

from qwengpu.compiled_tile8 import C_MEMORY, INDEX_OPERAND
from qwengpu.eval_text_handoff import (
    CORPUS,
    KERNEL,
    ROOT,
    clear_text_history,
    compare,
    launch_gpu,
    measure_text,
    persistent_gpu_rows,
)
from qwengpu.eval_tile8 import (
    RESIDENT_ADD32_LAYOUT,
    execute,
    load_tile_model,
    oracle,
    parse_launch,
)
from qwengpu.textcheck_9b import _ids, load_prompts

CONSUMER = ROOT / "eval/frozen/cache_relaunch_gate_v1/consumer.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.model / "qwengpu9b.json").read_text())
    if meta["layout"] != RESIDENT_ADD32_LAYOUT:
        parser.error("expected a resident ADD32 checkpoint")
    offset = meta["chip_token_offset"]
    model, circuit = load_tile_model(args.model, args.device, resident_add32=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    ids = _ids(tokenizer, load_prompts(CORPUS)[0])
    fresh_text = measure_text(model, ids, args.device, offset, eos)

    producer_payload = KERNEL.read_bytes()
    producer = json.loads(producer_payload)
    producer_case = parse_launch(producer, looping=True, resident_add32=True)
    expected_producer = oracle(producer_case, 256, looping=True)
    cache, gpu = launch_gpu(model, circuit, producer, args.device, offset)
    rows = persistent_gpu_rows(cache, circuit)
    clear_text_history(cache, model.config, meta["chip_layers"])
    after_text = measure_text(model, ids, args.device, offset, eos, cache=cache)
    text_comparison = compare(fresh_text, after_text)
    if (
        text_comparison["next_max_abs_logit_difference"] != 0
        or not text_comparison["greedy_identical"]
    ):
        raise ValueError("text boundary did not reproduce the fresh-cache output")
    if persistent_gpu_rows(cache, circuit) != rows:
        raise ValueError("text changed persistent GPU rows")
    clear_text_history(cache, model.config, meta["chip_layers"])

    consumer_payload = CONSUMER.read_bytes()
    consumer = json.loads(consumer_payload)
    shared = expected_producer["shared_words"]
    operands = [byte for word in shared[:64] for byte in (word & 255, word >> 8)]
    # These arrays are the interpreter's expected memory image. The retained launch
    # uses them for trace and cache checks without converting them to input tokens.
    launch = parse_launch(
        {**consumer, "a": operands[:64], "b": operands[64:], "c": shared[64:]},
        looping=True,
        resident_add32=True,
    )
    expected = oracle(launch, 256, looping=True)

    shared_tokens_by_call = []

    def audit(_model, _args, kwargs) -> None:
        if kwargs["past_key_values"] is not cache:
            raise ValueError("consumer did not retain the producer's cache")
        tokens = kwargs["input_ids"][0].tolist()
        shared_tokens_by_call.append(
            sum(
                any(
                    base <= token - offset < base + 4096
                    for base in (INDEX_OPERAND, C_MEMORY)
                )
                for token in tokens
            )
        )

    handle = model.register_forward_pre_hook(audit, with_kwargs=True)
    try:
        retained = execute(
            model,
            circuit,
            launch,
            args.device,
            256,
            offset,
            resident_add32=True,
            resident_io=True,
            resident_cache=cache,
        )
    finally:
        handle.remove()
    cold = execute(
        model,
        circuit,
        launch,
        args.device,
        256,
        offset,
        resident_add32=True,
        resident_io=True,
    )
    retained_transport = retained.pop("transport")
    cold_transport = cold.pop("transport")
    if retained != expected or cold != expected or retained["outcome"] != "halted":
        raise ValueError("consumer trace differs from the reference")
    if not all(
        retained_transport[f"resident_{bank}_exact"]
        for bank in ("grid", "memory", "rom", "state")
    ):
        raise ValueError("consumer cache rows differ from the reference")
    if (
        retained_transport["fresh_caches"] != 0
        or retained_transport["matrix_initial_tokens"] != 0
        or shared_tokens_by_call[0] != 0
        or sum(shared_tokens_by_call) != retained_transport["matrix_patch_tokens"]
        or cold_transport["matrix_initial_tokens"] != 512
        or cold_transport["input_tokens"] - retained_transport["input_tokens"] != 512
    ):
        raise ValueError("consumer shared-memory transport audit failed")
    if retained["matrix"][6:8] != expected_producer["matrix"][:2]:
        raise ValueError("consumer did not copy the first two produced rows")

    report = {
        "device": args.device,
        "merge": meta,
        "producer_sha256": hashlib.sha256(producer_payload).hexdigest(),
        "consumer_sha256": hashlib.sha256(consumer_payload).hexdigest(),
        "producer": gpu,
        "text": {
            "prompt_index": 0,
            "fresh": fresh_text[1],
            "after_gpu": after_text[1],
            "comparison": text_comparison,
        },
        "gpu_rows_preserved_after_text": True,
        "retained_transport": retained_transport,
        "cold_transport": cold_transport,
        "shared_tokens_by_call": shared_tokens_by_call,
        "consumer": retained,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Exact round trip, {len(retained['trace'])} consumer instructions, 512 upload tokens saved"
    )


if __name__ == "__main__":
    main()
