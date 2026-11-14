"""Check the v4 matrix launch and, for merged CUDA, native text."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results"
LAYOUT = "qwengpu-resident-generation-9b-v4"
TOKEN_SHA256 = "a4e76b2f495757e7a35dc0f880d3265e15d1c1aeabfba2fc67c182072d169ac2"


def check(reports, *, mps=False, standalone_cuda=False):
    matrix = json.loads((reports / "matmul16-single-launch.json").read_text())
    prior_matrix = json.loads((EVIDENCE / "model_stores_merged_cuda/matmul16.json").read_text())
    fixture = ROOT / "eval/frozen/matmul16_gate_v1/dense_seeded.json"
    assert matrix["input_sha256"] == prior_matrix["input_sha256"] == hashlib.sha256(
        fixture.read_bytes()
    ).hexdigest()
    assert matrix["device"] == ("mps" if mps else "cuda")
    assert matrix["checkpoint"] == LAYOUT
    assert matrix["transport_mode"] == "stock_generation" and matrix["exact"]
    assert matrix["generation_calls"] == matrix["fresh_caches"] == 1
    assert matrix["host_execution_tokens"] == 0
    assert matrix["prompt_tokens"] == 2507
    assert matrix["matrix"] == prior_matrix["matrix"]
    tokens = matrix["generated_tokens"]
    assert tokens == matrix["expected_tokens"] and len(tokens) == 20167
    assert hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest() == TOKEN_SHA256
    assert [token - 2784 for token in tokens if 2784 <= token < 2792] == list(range(1, 8))
    assert matrix["panel_transitions"] == list(range(1, 8))
    assert sum(2708 <= token < 2712 for token in tokens) == matrix["instructions"] == 928
    print("Single launch: 928 instructions, 20,167 tokens, 7 panel transitions, 256 exact outputs")
    if mps or standalone_cuda:
        assert matrix["merge"] is None
        return
    prior_text = json.loads((EVIDENCE / "resident_generation_v3_cuda/text.json").read_text())
    text = json.loads((reports / "text.json").read_text())
    assert text["device"] == "cuda"
    assert text["merge"] == matrix["merge"] and matrix["merge"]["layout"] == LAYOUT
    for key in ("prompts_sha256", "tokenized_prompts_sha256", "prompt_count", "max_new_tokens"):
        assert text[key] == prior_text[key], key
    assert text["next_top1_same"] == 24 and text["greedy_identical"] >= 21
    assert text["merged_chip_tokens"] == 0
    print(f"Native text: 24 first tokens, {text['greedy_identical']} full continuations, 0 chip IDs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--mps", action="store_true", help="check standalone Apple MPS evidence")
    backend.add_argument("--standalone-cuda", action="store_true",
                         help="check standalone CUDA evidence")
    args = parser.parse_args()
    check(args.reports, mps=args.mps, standalone_cuda=args.standalone_cuda)


if __name__ == "__main__":
    main()
