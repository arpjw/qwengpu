"""Check paired 32-bit copies through the retained GPU cache."""

import argparse
import hashlib
import json
from pathlib import Path

from check_addmem32_text import check_reset
from check_resident_add32_full import RESULTS, cases_by_name, load_report
from check_resident_addmem32 import LAYOUT, MANIFEST, check_sum

CONSUMER = RESULTS.parent / "eval/frozen/cache_relaunch_gate_v1/addmem32_consumer.json"


def check_copy(report: dict, producer: dict) -> None:
    assert (
        report["producer_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    )
    assert (
        report["consumer_sha256"] == hashlib.sha256(CONSUMER.read_bytes()).hexdigest()
    )
    fields = (
        "schedules",
        "trace",
        "outcome",
        "matrix",
        "shared_words",
        "warp_registers",
    )
    producer_trace = {name: producer[name] for name in fields}
    assert (
        report["producer"]["trace_sha256"]
        == hashlib.sha256(
            json.dumps(producer_trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert report["producer"]["transport"] == producer["transport"]

    consumer = report["consumer"]
    assert consumer["outcome"] == "halted"
    assert len(consumer["trace"]) == 20 and len(consumer["schedules"]) == 21
    manifest = json.loads(CONSUMER.read_bytes())
    original = producer["shared_words"]
    expected = original[:]
    registers = []
    for bank in manifest["initial_registers"]:
        values = []
        for seed in bank:
            dst = seed & 127
            src = dst ^ 64
            if seed & 128:
                assert original[dst] == original[dst ^ 1] == 0
                low, high = original[src], original[src ^ 1]
                expected[dst], expected[dst ^ 1] = low, high
                values.append(low | high << 16)
            else:
                values.append(seed)
        registers.append(values)
    assert consumer["warp_registers"] == registers
    assert consumer["shared_words"] == expected
    retained = report["retained_transport"]
    assert retained["matrix_patch_tokens"] == 64
    assert retained["matrix_initial_tokens"] == retained["fresh_caches"] == 0
    assert report["shared_tokens_by_call"][0] == 0
    assert sum(report["shared_tokens_by_call"]) == 64
    assert report["cold_transport"]["input_tokens"] - retained["input_tokens"] == 512


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.reports or [
        RESULTS / f"resident_addmem32_arithmetic_relaunch_{backend}.json"
        for backend in ("mps", "9b_cuda_fla")
    ]
    producer = cases_by_name(load_report("resident_addmem32_full_mps"))[MANIFEST.name]
    check_sum(producer)
    previous = None
    for path in paths:
        report = json.loads(path.read_text())
        check_copy(report, producer)
        if report["merge"] is not None:
            expected = load_report("resident_addmem32_full_9b_cuda_fla")["merge"]
            assert report["merge"] == expected and expected["layout"] == LAYOUT
            check_reset(report["text"]["comparison"])
            assert report["gpu_rows_preserved_after_text"]
        else:
            assert report["text"] is None
        if previous is not None:
            for key in (
                "producer",
                "consumer",
                "retained_transport",
                "cold_transport",
                "shared_tokens_by_call",
            ):
                assert report[key] == previous[key]
        previous = report
        print(f"{path.name}: exact paired copy and zero initial shared-memory uploads")
    if not args.reports:
        legacy = load_report(
            "resident_addmem32_cache_relaunch_legacy_replay_9b_cuda_fla"
        )
        old = load_report("resident_addmem32_cache_relaunch_9b_cuda_fla")
        assert legacy["merge"] == old["merge"]
        check_reset(legacy["text"]["comparison"])
        assert {key: value for key, value in legacy.items() if key != "text"} == {
            key: value for key, value in old.items() if key != "text"
        }
        print("Legacy relaunch: exact inherited GPU trace and fresh-cache text check")


if __name__ == "__main__":
    main()
