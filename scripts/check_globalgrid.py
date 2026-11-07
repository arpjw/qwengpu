"""Check saved globalgrid ID, memory, and reduction evidence."""

from __future__ import annotations

import argparse
import hashlib
import json

from check_resident_add32_full import RESULTS, load_report

LAYOUT = "qwengpu-resident-globalgrid-9b-v1"
FIXTURE = RESULTS.parent / "eval/frozen/globalgrid_v1/words.json"
MASK32 = 0xFFFFFFFF


def lanes(step: dict) -> list[int]:
    digits = step["registers"]
    return [
        sum(
            (token & 15) << (4 * digit) for digit, token in enumerate(digits[i : i + 4])
        )
        for i in range(0, 32, 4)
    ]


def check(report: dict, layout: str = LAYOUT) -> None:
    payload = FIXTURE.read_bytes()
    values = json.loads(payload)
    assert report["input_sha256"] == hashlib.sha256(payload).hexdigest()
    assert report["input"] == values
    if report["merge"] is not None:
        assert report["merge"]["layout"] == layout
    sentinel = [0xFFF0 + i % 16 for i in range(128)]
    for count, probe in enumerate(report["id_probes"], 1):
        expected = sentinel[:]
        expected[: 32 * count : 2] = range(16 * count)
        assert probe["global_words"] == expected
        assert len(probe["trace"]) == count * 24
        assert probe["workgroup_cursor"] == count - 1
        for group in range(count):
            for warp in range(2):
                trace = [
                    step
                    for step in probe["trace"]
                    if (step["workgroup"], step["warp"]) == (group, warp)
                ]
                ids = [16 * group + 8 * warp + lane for lane in range(8)]
                partners = [i ^ 16 for i in ids]
                reads = partners if group % 2 else [sentinel[2 * i] for i in partners]
                assert lanes(trace[0]) == [0] * 8
                assert lanes(trace[1]) == ids
                assert lanes(trace[6]) == [
                    (i + j) & 65535 for i, j in zip(ids, reads, strict=True)
                ]
                assert lanes(trace[7]) == list(range(8))
                assert lanes(trace[8]) == list(range(8 * warp, 8 * warp + 8))
                assert lanes(trace[9]) == [group] * 8
    assert len(report["id_probes"]) == 4
    if "producer" not in report:
        return
    producer, consumer = report["producer"], report["consumer"]
    expected = [
        part for value in values for part in (value & 65535, value >> 16)
    ] + sentinel[64:]
    partials = [sum(values[i : i + 16]) & MASK32 for i in (0, 16)]
    assert report["partials"] == partials
    for group, partial in enumerate(partials):
        for address in (32 * group, 32 * group + 16):
            expected[address : address + 2] = [partial & 65535, partial >> 16]
    assert producer["global_words"] == expected
    total = sum(values) & MASK32
    expected[:2] = [total & 65535, total >> 16]
    assert consumer["global_words"] == expected and report["sum"] == total
    assert len(producer["trace"]) == 64 and len(consumer["trace"]) == 12
    assert len(producer["workgroup_transitions"]) == 1
    assert producer["transport"]["fresh_caches"] == 1
    assert producer["transport"]["matrix_initial_tokens"] == 512
    assert (
        consumer["transport"]["fresh_caches"]
        == consumer["transport"]["matrix_initial_tokens"]
        == 0
    )
    for run in (producer, consumer):
        assert all(
            run["transport"][f"resident_{kind}_exact"]
            for kind in ("grid", "memory", "rom", "state")
        )


def check_public(probe: dict, backend: str) -> None:
    public = load_report(f"globalgrid_public_ids_{backend}")
    for field in (
        "trace",
        "schedules",
        "global_words",
        "workgroup_cursor",
        "workgroup_transitions",
        "transport",
    ):
        assert public[field] == probe[field], field
    assert all(public["reference"].values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids-only", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    mps = load_report(
        "globalgrid_ids_mps" if args.ids_only else "globalgrid_reduction_mps"
    )
    check(mps)
    print("MPS: exact IDs, cross-workgroup reads, and global memory guards")
    if not args.ids_only:
        check_public(mps["id_probes"][1], "mps")
        print(f"MPS: exact retained-cache reduction, sum = 0x{mps['sum']:08x}")
    if args.cuda:
        cuda = load_report("globalgrid_reduction_9b_cuda_fla")
        check(cuda)
        for key in ("id_probes", "producer", "consumer", "partials", "sum"):
            if key in mps:
                assert mps[key] == cuda[key], key
        check_public(cuda["id_probes"][1], "9b_cuda_fla")
        legacy = load_report("globalgrid_legacy_assembly_9b_cuda_fla")
        assert legacy["checkpoint"] == LAYOUT
        baseline = load_report("assembled_cooperative_pair_mps")
        for field in (
            "trace",
            "schedules",
            "matrix",
            "warp_registers",
            "shared_words",
            "reference",
        ):
            assert legacy[field] == baseline[field], field
        print("Merged 9B CUDA: exact matching traces, state, and transport")


if __name__ == "__main__":
    main()
