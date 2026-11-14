"""Compare a restored checkpoint's execution with its qualified reports."""

import argparse
import hashlib
import json
from pathlib import Path

QUALIFIED = Path(__file__).resolve().parents[1] / "results" / "resident_generation_v3_cuda"
V4_LAYOUT = "qwengpu-resident-generation-9b-v4"


def read(directory, name):
    return json.loads((directory / name).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    parser.add_argument("--qualified", type=Path, default=QUALIFIED,
                        help="complete qualification report directory")
    parser.add_argument("--v4", action="store_true", help="require the v4 single-launch matrix")
    args = parser.parse_args()
    qualified = args.qualified
    known_checksums = qualified / "package-SHA256SUMS"
    digest = hashlib.sha256(known_checksums.read_bytes()).hexdigest()
    assert (args.reports / "restore-checksum.txt").read_text().strip() == digest
    tile = read(args.reports, "restored-tile.json")
    baseline, = (
        case for case in read(qualified, "inherited-merged.json")["cases"]
        if case["fixture"] == "dense_seeded.json"
    )
    assert tile["device"] == "cuda" and tile["outcome"] == "halted"
    checks = {
        "tokens_exact", "rom_exact", "registers_exact", "control_exact", "memory_exact",
    }
    assert checks <= tile["reference"].keys() and all(tile["reference"].values())
    assert tile["kernel_sha256"] == baseline["sha256"]
    assert tile["instructions"] == baseline["steps"]
    for key in ("generated_tokens", "tile_words"):
        assert tile[key] == baseline[key], key
    text = read(args.reports, "restored-text.json")
    qualified_text = read(qualified, "text.json")
    assert text["merge"] == qualified_text["merge"]
    for key in ("prompts_sha256", "tokenized_prompts_sha256", "prompt_count", "max_new_tokens"):
        assert text[key] == qualified_text[key], key
    assert text["generation_config"] == qualified_text["generation_config"]
    assert len(text["records"]) == len(qualified_text["records"]) == 24
    text_keys = ("prompt", "prompt_tokens", "base_greedy_ids", "merged_greedy_ids",
                 "merged_chip_tokens", "next_top1_same", "greedy_identical")
    for restored, expected in zip(text["records"], qualified_text["records"], strict=True):
        for key in text_keys:
            assert restored[key] == expected[key], key
    assert text["next_top1_same"] == 24 and text["greedy_identical"] >= 21
    assert text["merged_chip_tokens"] == 0
    if args.v4:
        assert read(qualified, "manifest.json")["files"]["qwengpu9b.json"]["layout"] == V4_LAYOUT
        matrix = read(args.reports, "restored-matmul16.json")
        expected = read(qualified, "matmul16-single-launch.json")
        assert matrix["device"] == "cuda" and matrix["checkpoint"] == V4_LAYOUT
        assert matrix["exact"] and matrix["merge"] == qualified_text["merge"]
        for key in ("input_sha256", "generated_tokens", "expected_tokens", "matrix",
                    "panel_transitions", "instructions", "prompt_tokens", "generation_calls",
                    "fresh_caches", "host_execution_tokens"):
            assert matrix[key] == expected[key], key
        assert matrix["generation_calls"] == matrix["fresh_caches"] == 1
        assert matrix["host_execution_tokens"] == 0
        print(f"Verified restored v4 matrix: {matrix['instructions']} instructions in one launch")
    print(f"Verified package {digest}, {tile['instructions']} GPU instructions, and native text")


if __name__ == "__main__":
    main()
