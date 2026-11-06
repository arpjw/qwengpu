"""Measure text after a verified GPU launch on the same inference cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from transformers.cache_utils import Cache, DynamicCache

from qwengpu.eval_compiled_rom import head_row
from qwengpu.eval_tile8 import (
    RESIDENT_ADD32_LAYOUT,
    execute,
    load_tile_model,
    oracle,
    parse_launch,
)
from qwengpu.textcheck_9b import _ids, load_prompts

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "eval/frozen/resident_add32_gate_v1/cooperative_add32.json"
CORPUS = ROOT / "eval/frozen/text_add32_gate_v1/prompts.json"
PROMPT_INDICES = (0, 1, 21, 23)
NEW_TOKENS = 16


def launch_gpu(
    model, circuit, manifest: dict, device: str, offset: int
) -> tuple[Cache, dict]:
    case = parse_launch(manifest, looping=True, resident_add32=True)
    cache = None

    def capture(_model, _args, output) -> None:
        nonlocal cache
        cache = output.past_key_values

    handle = model.register_forward_hook(capture)
    try:
        observed = execute(
            model,
            circuit,
            case,
            device,
            256,
            offset,
            resident_add32=True,
            resident_io=True,
        )
    finally:
        handle.remove()
    transport = observed.pop("transport")
    expected = oracle(case, 256, looping=True)
    if (
        observed != expected
        or observed["outcome"] != "halted"
        or observed["matrix"] != manifest["expected_matrix"]
        or transport["fresh_caches"] != 1
        or not all(
            transport[f"resident_{bank}_exact"]
            for bank in ("grid", "memory", "rom", "state")
        )
    ):
        raise ValueError("GPU launch did not match the reference")
    if cache is None or cache.get_seq_length() != transport["input_tokens"]:
        raise ValueError("GPU cache length does not match the launch transcript")
    return cache, {
        "instructions": len(observed["trace"]),
        "scheduler_decisions": len(observed["schedules"]),
        "input_tokens": transport["input_tokens"],
        "trace_sha256": hashlib.sha256(
            json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "transport": transport,
    }


def clear_text_history(cache: Cache, config, chip_layers: int) -> None:
    """Replace pretrained layers' cache entries while preserving the GPU prefix."""
    if (
        not isinstance(cache, DynamicCache)
        or not 0 < chip_layers < len(cache.layers)
        or any(kind != "linear_attention" for kind in config.layer_types[:chip_layers])
        or "full_attention" not in config.layer_types[chip_layers:]
    ):
        raise ValueError("expected a dynamic resident cache with a linear GPU prefix")
    fresh = DynamicCache(config=config)
    if len(fresh.layers) != len(cache.layers):
        raise ValueError("cache layer count does not match the model")
    cache.layers[chip_layers:] = fresh.layers[chip_layers:]
    if cache.get_seq_length() != 0:
        raise ValueError("text cache did not reset to position zero")


def persistent_gpu_rows(cache: Cache, circuit) -> dict:
    rows = {
        "registers": [
            head_row(cache, circuit, f"{bank}.{nib}", warp, 4)
            for warp in range(2)
            for bank in ("register", "high")
            for nib in range(32)
        ],
        "shared_memory": [
            head_row(cache, circuit, f"shared.{lane}.{nib}", word, 4)
            for lane in range(8)
            for nib in range(4)
            for word in range(128)
        ],
        "tile_memory": [
            *(
                head_row(cache, circuit, f"A.{nib}", byte, 4)
                for nib in range(2)
                for byte in range(64)
            ),
            *(
                head_row(cache, circuit, f"B.{col}.{nib}", row, 4)
                for col in range(8)
                for nib in range(2)
                for row in range(8)
            ),
            *(
                head_row(cache, circuit, f"C.{nib}", row, 4)
                for nib in range(32)
                for row in range(8)
            ),
        ],
    }
    if any(value is None for bank in rows.values() for value in bank):
        raise ValueError("persistent GPU row was not initialized")
    return rows


@torch.no_grad()
def measure_text(
    model,
    ids: list[int],
    device: str,
    vocab: int,
    eos: set[int],
    *,
    cache: Cache | None = None,
    position_start: int | None = None,
) -> tuple[torch.Tensor, dict]:
    before = cache.get_seq_length() if cache is not None else 0
    position = position_start
    tokens = ids
    greedy = []
    for step in range(NEW_TOKENS):
        positions = None
        if position is not None:
            positions = torch.arange(position, position + len(tokens), device=device)[
                None
            ]
            position += len(tokens)
        output = model(
            input_ids=torch.tensor([tokens], device=device),
            past_key_values=cache,
            position_ids=positions,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        if step == 0:
            after_prompt = cache.get_seq_length()
            native_logits = output.logits[0, -1, :vocab].float().cpu()
        token = int(output.logits[0, -1].argmax())
        greedy.append(token)
        if token in eos:
            break
        tokens = [token]
    if (
        after_prompt != before + len(ids)
        or cache.get_seq_length() != after_prompt + len(greedy) - 1
    ):
        raise ValueError(
            "text cache length did not advance by the number of input tokens"
        )
    if not torch.isfinite(native_logits).all():
        raise ValueError("nonfinite text logits")
    return native_logits, {
        "cache_before_prompt": before,
        "cache_after_prompt": after_prompt,
        "cache_after_greedy": cache.get_seq_length(),
        "first_position": before if position_start is None else position_start,
        "next_native_top1": int(native_logits.argmax()),
        "greedy_ids": greedy,
        "chip_tokens": sum(token >= vocab for token in greedy),
    }


def compare(left: tuple[torch.Tensor, dict], right: tuple[torch.Tensor, dict]) -> dict:
    left_logits, left_trace = left
    right_logits, right_trace = right
    diff = (left_logits - right_logits).abs()
    a, b = left_trace["greedy_ids"], right_trace["greedy_ids"]
    prefix = next(
        (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
    )
    return {
        "native_next_top1_same": left_trace["next_native_top1"]
        == right_trace["next_native_top1"],
        "greedy_identical": a == b,
        "greedy_prefix_tokens": prefix,
        "next_max_abs_logit_difference": float(diff.max()),
        "next_mean_abs_logit_difference": float(diff.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--reset-text",
        action="store_true",
        help="replace pretrained cache entries before appending each text prompt",
    )
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
    corpus = load_prompts(CORPUS)
    payload = KERNEL.read_bytes()
    manifest = json.loads(payload)
    records = []
    for index in PROMPT_INDICES:
        ids = _ids(tokenizer, corpus[index])
        fresh = measure_text(model, ids, args.device, offset, eos)
        cache, gpu = launch_gpu(model, circuit, manifest, args.device, offset)
        record = {
            "prompt_index": index,
            "prompt": corpus[index],
            "prompt_ids": ids,
            "gpu": gpu,
            "fresh": fresh[1],
        }
        if args.reset_text:
            before = persistent_gpu_rows(cache, circuit)
            clear_text_history(cache, model.config, meta["chip_layers"])
            reset = measure_text(model, ids, args.device, offset, eos, cache=cache)
            after = persistent_gpu_rows(cache, circuit)
            record.update(
                reset=reset[1],
                reset_vs_fresh=compare(fresh, reset),
                gpu_rows={
                    name: {
                        "count": len(values),
                        "before_sha256": hashlib.sha256(
                            json.dumps(values).encode()
                        ).hexdigest(),
                        "after_sha256": hashlib.sha256(
                            json.dumps(after[name]).encode()
                        ).hexdigest(),
                        "exact": values == after[name],
                    }
                    for name, values in before.items()
                },
            )
            comparison = record["reset_vs_fresh"]
        else:
            reused = measure_text(model, ids, args.device, offset, eos, cache=cache)
            del cache
            positioned = measure_text(
                model, ids, args.device, offset, eos, position_start=gpu["input_tokens"]
            )
            record.update(
                gpu_cache=reused[1],
                position_only=positioned[1],
                gpu_vs_fresh=compare(fresh, reused),
                position_vs_fresh=compare(fresh, positioned),
                gpu_vs_position=compare(positioned, reused),
            )
            comparison = record["gpu_vs_fresh"]
        records.append(record)
        print(json.dumps({"prompt_index": index, **comparison}), flush=True)
    report = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "merge": meta,
        "kernel_sha256": hashlib.sha256(payload).hexdigest(),
        "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "max_new_tokens": NEW_TOKENS,
        "eos_token_ids": sorted(eos),
        "records": records,
        **({"cache_policy": "reset_text_layers"} if args.reset_text else {}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
