"""Verify two model-selected workgroups and direct resident global writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import transformers

from qwengpu.compiled_add32 import LANES, WORD_BITS, WORD_BYTES
from qwengpu.compiled_vector_grid import (
    A, B, BYTE_SLOTS, C, DONE, GROUP, GROUPS, LENGTH, OUT, READ, START, STEP,
    WORDS, build_standalone,
)
from qwengpu.eval_add32 import bytes_of
from qwengpu.eval_compiled_rom import head_row
from qwengpu.eval_tile import _unsigned

FIXTURE = Path(__file__).resolve().parents[1] / "eval/frozen/vector_grid_gate_v1/guarded.json"
WORD_MASK = (1 << WORD_BITS) - 1


@torch.no_grad()
def evaluate_case(model, circuit, a, b, initial, length: int, device: str) -> dict:
    start = time.monotonic()
    prefix = [
        base + 256 * slot + byte
        for base, words in ((A, a), (B, b), (C, initial))
        for slot, byte in enumerate(bytes_of(words))
    ] + [LENGTH + length, START]
    cache = None
    inputs = []

    def feed(tokens: list[int], keep: int = 1) -> list[int]:
        nonlocal cache
        inputs.append(tokens)
        result = model(
            input_ids=torch.tensor([tokens], device=device), past_key_values=cache,
            use_cache=True, logits_to_keep=keep,
        )
        cache = result.past_key_values
        return result.logits[0].argmax(dim=-1).tolist()

    (token,) = feed(prefix)
    trace = [token]
    for _ in range(GROUPS):
        if not GROUP <= token < DONE:
            break
        (token,) = feed([token, STEP])
        trace.append(token)
    after_done = feed([DONE, STEP]) if token == DONE else []
    readout = feed(list(range(READ, READ + BYTE_SLOTS)), BYTE_SLOTS)
    cache_bytes = {
        name: [
            head_row(cache, circuit, f"{name}.{lane}.{byte}", group, 8)
            for group in range(GROUPS)
            for lane in range(LANES)
            for byte in range(WORD_BYTES)
        ]
        for name in ("a", "b", "c")
    }
    expected = initial[:]
    expected[:length] = [(a[i] + b[i]) & WORD_MASK for i in range(length)]
    expected_trace = [GROUP + i for i in range((length + LANES - 1) // LANES)] + [DONE]
    exact = (
        trace == expected_trace
        and after_done == [DONE]
        and readout == [OUT + byte for byte in bytes_of(expected)]
        and cache_bytes == {"a": bytes_of(a), "b": bytes_of(b), "c": bytes_of(expected)}
    )
    return {
        "length": length, "trace": trace, "expected_trace": expected_trace,
        "after_done": after_done, "expected_words": expected,
        "observed_bytes": [token - OUT for token in readout], "cache_bytes": cache_bytes,
        "prefix_tokens": len(prefix), "post_upload_inputs": inputs[1:-1],
        "input_tokens": sum(map(len, inputs)), "forward_calls": len(inputs),
        "seconds": round(time.monotonic() - start, 3), "exact": exact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = FIXTURE.read_bytes()
    case = json.loads(payload)
    a, b, initial = (_unsigned(case[name], WORDS, WORD_MASK, name)
                     for name in ("a", "b", "initial"))
    lengths = _unsigned(case["lengths"], len(case["lengths"]), WORDS, "lengths")
    model, circuit = build_standalone()
    model.to(args.device).eval()
    cases = [evaluate_case(model, circuit, a, b, initial, length, args.device)
             for length in lengths]
    report = {
        "device": args.device, "torch": torch.__version__,
        "transformers": transformers.__version__,
        "fixture_sha256": hashlib.sha256(payload).hexdigest(),
        "circuit": circuit.stats(), "cases": cases,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{sum(case['exact'] for case in cases)}/{len(cases)} exact on {args.device}")
    if not all(case["exact"] for case in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
