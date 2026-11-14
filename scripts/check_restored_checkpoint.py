"""Compare a restored v3 checkpoint's execution with its qualified reports."""

import argparse
import hashlib
import json
from pathlib import Path

QUALIFIED = Path(__file__).resolve().parents[1] / "results" / "resident_generation_v3_cuda"


def read(directory, name):
    return json.loads((directory / name).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    args = parser.parse_args()
    known_checksums = QUALIFIED / "package-SHA256SUMS"
    digest = hashlib.sha256(known_checksums.read_bytes()).hexdigest()
    assert (args.reports / "restore-checksum.txt").read_text().strip() == digest
    tile = read(args.reports, "restored-tile.json")
    baseline, = (
        case for case in read(QUALIFIED, "inherited-merged.json")["cases"]
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
    qualified_text = read(QUALIFIED, "text.json")
    assert text["merge"] == qualified_text["merge"]
    assert text["next_top1_same"] == 24 and text["greedy_identical"] >= 21
    assert text["merged_chip_tokens"] == 0
    print(f"Verified package {digest}, {tile['instructions']} GPU instructions, and native text")


if __name__ == "__main__":
    main()
