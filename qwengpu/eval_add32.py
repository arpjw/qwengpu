"""Verify emitted bytes and recurrent state for the masked ADD32 prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from chipformer.q35.ir import Target

from qwengpu.compiled_add32 import (
    BYTE_SLOTS,
    EXEC,
    LANES,
    LAYOUT,
    MASK,
    OUT,
    READ,
    WORD_BITS,
    WORD_BYTES,
    A,
    B,
    build,
    build_standalone,
)
from qwengpu.eval_compiled_rom import head_row
from qwengpu.eval_tile import _unsigned

FIXTURE = Path(__file__).resolve().parents[1] / "eval/frozen/add32_gate_v1/lanes.json"
WORD_MASK = (1 << WORD_BITS) - 1


def bytes_of(words: list[int]) -> list[int]:
    return [(word >> shift) & 255 for word in words for shift in range(0, WORD_BITS, 8)]


@torch.no_grad()
def evaluate_case(model, circuit, case: dict, device: str, offset: int) -> dict:
    a = _unsigned(case["a"], LANES, WORD_MASK, "a")
    b = _unsigned(case["b"], LANES, WORD_MASK, "b")
    mask = _unsigned([case["mask"]], 1, 255, "mask")[0]
    uploads = [
        base + 256 * slot + byte
        for base, words in ((A, a), (B, b))
        for slot, byte in enumerate(bytes_of(words))
    ]
    prefix = [*uploads, MASK + mask, EXEC]
    prefill = model(
        input_ids=torch.tensor([[token + offset for token in prefix]], device=device),
        use_cache=True,
        logits_to_keep=1,
    )
    readout = model(
        input_ids=torch.tensor(
            [[READ + slot + offset for slot in range(BYTE_SLOTS)]], device=device
        ),
        past_key_values=prefill.past_key_values,
        use_cache=True,
        logits_to_keep=BYTE_SLOTS,
    )
    cache = readout.past_key_values
    observed = [token - offset for token in readout.logits[0].argmax(dim=-1).tolist()]
    expected_words = [
        (left + right) & WORD_MASK if (mask >> lane) & 1 else left
        for lane, (left, right) in enumerate(zip(a, b, strict=True))
    ]
    expected = [
        OUT + 256 * slot + byte for slot, byte in enumerate(bytes_of(expected_words))
    ]
    cache_words = [
        head_row(cache, circuit, f"result.{lane}", 0, WORD_BITS)
        for lane in range(LANES)
    ]
    cache_mask = head_row(cache, circuit, "mask", 0, LANES)
    cache_operands = [
        [
            head_row(cache, circuit, f"{name}.{lane}.{byte}", 0, 8)
            for lane in range(LANES)
            for byte in range(WORD_BYTES)
        ]
        for name in ("a", "b")
    ]
    return {
        **case,
        "expected_words": expected_words,
        "expected_tokens": expected,
        "observed_tokens": observed,
        "cache_words": cache_words,
        "cache_mask": cache_mask,
        "cache_operands_exact": cache_operands == [bytes_of(a), bytes_of(b)],
        "exact": (
            observed == expected
            and cache_words == expected_words
            and cache_mask == mask
            and cache_operands == [bytes_of(a), bytes_of(b)]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path)
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    payload = FIXTURE.read_bytes()
    cases = json.loads(payload)
    meta = None
    offset = 0
    if args.model:
        meta = json.loads((args.model / "qwengpu9b.json").read_text())
        if meta["layout"] != LAYOUT:
            parser.error("expected a merged ADD32 checkpoint")
        offset = meta["chip_token_offset"]
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16
        )
        cfg = model.config
        circuit = build(
            Target(
                cfg.linear_num_key_heads,
                cfg.linear_num_value_heads,
                cfg.intermediate_size,
            )
        )
    else:
        model, circuit = build_standalone()
    model = model.to(args.device).eval()
    results = [
        evaluate_case(model, circuit, case, args.device, offset) for case in cases
    ]
    report = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "circuit": circuit.stats(),
        "merge": meta,
        "fixture_sha256": hashlib.sha256(payload).hexdigest(),
        "cases": results,
        "all_exact": all(case["exact"] for case in results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{sum(case['exact'] for case in results)}/{len(results)} exact cases")
    if not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
