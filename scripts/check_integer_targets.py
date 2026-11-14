"""Compare one-model feature smoke checks with the merged checkpoint records."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/final_candidate_cuda"


def load(path):
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    args = parser.parse_args()
    for name, baseline in (
        ("interleaved_publish", "interleaved_publish"),
        ("barrier_groups", "interleaved_barrier_groups"),
        ("deadlock", "interleaved_deadlock"),
        ("indexed_words", "indexed_words"),
        ("isolation", "isolation"),
        ("tiled_3x5x2", "tiled_3x5x2"),
        ("tiled_9x7x5", "tiled_9x7x5"),
    ):
        report = load(args.reports / f"{name}.json")
        result = report["result"]
        merged = load(BASE / f"{baseline}.json")
        if name.startswith("tiled_"):
            for key in ("output", "global_words", "launches"):
                assert result[key] == merged[key], (name, key)
        else:
            for key in ("trace", "schedules", "global_words", "warp_registers", "outcome"):
                assert result[key] == merged[key], (name, key)
            assert result.get("scoped_words") == merged.get("scoped_words")
        print(f"{report['device']}: {name} matches the merged checkpoint")


if __name__ == "__main__":
    main()
