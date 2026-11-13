"""Compare store traces and prove that actual model inputs contain only fixed commits."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read(path):
    return json.loads(path.read_bytes())


def check(path):
    report = read(path)
    baseline = {
        "address_walk": read(RESULTS / "final_candidate_cuda/address_walk.json"),
        "interleaved_publish": read(RESULTS / "final_candidate_cuda/interleaved_publish.json"),
        "isolation": read(RESULTS / "final_candidate_cuda/isolation.json"),
    }
    results = {}
    for case in report["cases"]:
        name, got = case["name"], case["result"]
        want = baseline[name]
        for key in (
            "trace", "schedules", "global_words", "warp_registers", "outcome",
            "workgroup_cursor", "workgroup_transitions",
        ):
            assert got[key] == want[key], (name, key)
        assert got.get("scoped_words") == want.get("scoped_words"), name
        t = got["transport"]
        count = sum(1 for step in got["trace"] for tag in step["patch"] if tag) // 4
        scoped = sum(
            1 for step in got["trace"] for tag in step["patch"] if 0x18000 <= tag < 0x1A000
        ) // 4
        assert t["internal_store_words"] == t["internal_store_commands"] == count
        assert t["internal_scoped_words"] == scoped
        assert t["internal_store_trace_tags"] == 4 * count
        assert t["matrix_patch_tokens"] == t["store_read_tokens"] == t["internal_store_read_tags"] == 0
        audit = got["audit"]
        assert sum(call["memory"] for call in audit) == audit[0]["memory"] == 1024
        assert sum(len(call["commits"]) for call in audit) == count
        if scoped:
            assert sum(call["scoped"] for call in audit) == audit[0]["scoped"] == 512
            assert t["scoped_patch_tokens"] == 0
        for key in ("resident_state_exact", "resident_memory_exact", "resident_rom_exact"):
            assert t[key]
        print(f"{report['device']}: {name}, {count} fixed commits, zero address/value tokens")
        results[name] = got
    assert results.keys() == baseline.keys()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    first, *other = [check(path) for path in args.reports]
    assert all(item == first for item in other)
