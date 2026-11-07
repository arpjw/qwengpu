"""Check frozen 512-byte evidence without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from check_compute_bounds import check as check_bounds
from check_globalgrid import check as check_globalgrid
from check_resident_add32_full import cases_by_name, load_report
from check_tilepipe_matmul16 import check_text

ROOT = Path(__file__).resolve().parents[1]
MASK32 = 0xFFFFFFFF


def split(values: list[int]) -> list[int]:
    return [part for value in values for part in (value & 65535, value >> 16)]


def check_run(run: dict, first: bool) -> None:
    assert run["outcome"] == "halted"
    assert len(run["global_words"]) == 256
    assert run["global_words"] == run["shared_words"]
    transport = run["transport"]
    assert transport["fresh_caches"] == int(first)
    assert transport["matrix_initial_tokens"] == 1024 * int(first)
    assert all(
        transport[f"resident_{kind}_exact"]
        for kind in ("grid", "memory", "rom", "state")
    )
    assert sum(call["memory"] for call in run["audit"]) == (
        1024 * int(first) + transport["matrix_patch_tokens"]
    )
    assert all(t["resident_memory_exact"] for t in run["workgroup_transitions"])


def check(report: dict) -> None:
    manifest = ROOT / "eval/frozen/memory512_compute_v1/cases.json"
    requests = json.loads(manifest.read_bytes())
    assert report["all_exact"]
    assert (
        report["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert len(report["cases"]) == len(requests) == 31
    for case in report["cases"]:
        name = case["name"]
        request = requests[name]
        assert case["request"] == request and case["exact"]
        runs = case["launches"]
        for i, run in enumerate(runs):
            check_run(run, i == 0)
        kind = request["kernel"]
        initial = runs[0]["global_words"][:]
        if kind == "vector_add":
            a, b = request["a"], request["b"]
            expected = [(x + y) & MASK32 for x, y in zip(a, b, strict=True)]
            words = split(a + [0xD00D0000 + i for i in range(64 - len(a))])
            words += split(b + [0xD00D0040 + i for i in range(64 - len(b))])
            words[: 2 * len(a)] = split(expected)
            assert len(runs) == 1
        elif kind == "reduce":
            values = request["values"]
            expected = sum(values) & MASK32
            padded = max(32, len(values))
            words = split(values + [0] * (padded - len(values)))
            words += split([0xD00D0000 + i for i in range(128 - padded)])
            for start in range(0, min(len(values), 64), 8):
                partial = (
                    sum(values[start : start + 8])
                    + sum(values[start + 64 : start + 72])
                ) & MASK32
                words[2 * start : 2 * start + 2] = split([partial])
            assert initial == words, name
            words[:2] = split([expected])
            assert len(runs) == 2
        else:
            a, b, c = (request[key] for key in ("a", "b", "c"))
            m, k, n = len(a), len(b), len(b[0])
            expected = [
                [
                    (c[i][j] + sum(a[i][t] * b[t][j] for t in range(k))) & MASK32
                    for j in range(n)
                ]
                for i in range(m)
            ]
            left = [v for row in a for v in row]
            right = [v for row in b for v in row]
            words = left + [0xAB00 + i for i in range(32 - len(left))]
            words += right + [0xBC00 + i for i in range(16 - len(right))]
            words += [0xCD00 + i for i in range(16)]
            flat = [v for row in expected for v in row]
            words += split(flat + [0xD00D0000 + i for i in range(32 - len(flat))])
            words += [0xBEEF] * 128
            assert len(runs) == 1
        assert case["output"] == expected, name
        assert case["global_words"] == runs[-1]["global_words"] == words, name
    paths = sorted((ROOT / "eval/frozen/memory512_v1").glob("*.json"))
    assert len(report["raw"]) == len(paths) == 7
    by_name = {item["name"]: item for item in report["raw"]}
    assert len(by_name) == 7
    for path in paths:
        data = path.read_bytes()
        fixture = json.loads(data)
        run = by_name[path.stem]
        check_run(run, True)
        assert run["fixture_sha256"] == hashlib.sha256(data).hexdigest()
        expected = fixture["global_words"][:]
        if path.stem.startswith("edge_"):
            for address in map(int, path.stem.split("_")[1:]):
                expected[address] = (expected[address] + 1) & 65535
        elif path.stem == "barrier_pair":
            expected[126:130] = [9, 0, 9, 0]
        else:
            expected[128] = 1
            if path.stem == "collision":
                expected[0] = 0
        assert run["global_words"] == expected, path.stem
    print(
        "31 kernel requests and seven raw probes: independent outputs and all 256 guard words exact"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--inherited", action="store_true")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--global-memory", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--bounds", action="store_true")
    args = parser.parse_args()
    mps = load_report("memory512_mps")
    check(mps)
    cuda = load_report("memory512_9b_cuda_fla") if args.cuda else None
    if cuda:
        assert cuda["merge"]["layout"] == "qwengpu-resident-memory512-9b-v1"
        assert cuda["merge"]["chip_vocab"] == 147456
        check(cuda)
        manifest = load_report("memory512_checkpoint_manifest")
        preflight = load_report("memory512_preflight")
        files = manifest["files"]
        meta = files["qwengpu9b.json"]
        assert meta == cuda["merge"]
        assert manifest["weights_stored_in_git"] is False
        assert manifest["compiler_commit"] == "ca13310e2f9f788d1dd3feeea8213c1b0c23913c"
        assert manifest["base_revision"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        assert (meta["hidden"], meta["slots"], meta["circuit"]) == (
            preflight["merged_hidden"],
            preflight["slots"],
            preflight["circuit"],
        )
        assert (
            preflight["legacy"]["ir_identical"]
            and preflight["legacy"]["rom_domain_identical"]
        )
        assert preflight["maximum_head_row"] == 127
        index = files["model.safetensors.index.json"]
        shards = set(index["weight_map"].values())
        assert len(shards) == 27 and shards <= manifest["sha256"].keys()
        sizes = manifest["dtype_bytes"]
        assert sizes["F32"] == 15360
        assert sum(sizes.values()) == index["metadata"]["total_size"]
        # The meta preflight estimates every parameter at two bytes; the
        # base retains 3840 float32 attention parameters in the actual merge.
        assert (
            index["metadata"]["total_size"]
            == preflight["estimated_weight_bytes_bf16"] + sizes["F32"] // 2
        )
        print(
            "Checkpoint metadata, all 27 shard hashes, and preflight allocation recorded"
        )
        for field in ("raw", "cases", "manifest_sha256"):
            assert cuda[field] == mps[field], field
        print("Merged 9B: all raw and common-kernel records match MPS exactly")
    if args.inherited:
        baseline = cases_by_name(load_report("compute_inherited_mps"))
        assert len(baseline) == 27
        for name in ("memory512_inherited_mps",) + (
            ("memory512_inherited_9b_cuda_fla",) if cuda else ()
        ):
            assert cases_by_name(load_report(name)) == baseline
        print("All 27 inherited case records match the previous checkpoint")
    if args.matrix:
        baseline = load_report("compute_matmul16_mps")
        fields = (
            "matrix",
            "passes",
            "transitions",
            "transport",
            "memory_tokens_by_call",
            "instructions",
            "scheduler_decisions",
            "workgroup_outputs",
            "exact",
        )
        for name in ("memory512_matmul16_mps",) + (
            ("memory512_matmul16_9b_cuda_fla",) if cuda else ()
        ):
            actual = load_report(name)
            for field in fields:
                assert actual[field] == baseline[field], (name, field)
        print(
            "16x16 tiled kernel: both pass matrices, all eight traces, and transport unchanged"
        )
    if args.global_memory:
        layout = "qwengpu-resident-memory512-9b-v1"
        grid = load_report("memory512_globalgrid_mps")
        check_globalgrid(grid, layout, memory_words=256)
        if cuda:
            other = load_report("memory512_globalgrid_9b_cuda_fla")
            check_globalgrid(other, layout, memory_words=256)
            for field in ("id_probes", "producer", "consumer", "partials", "sum"):
                assert other[field] == grid[field], field
        print(
            "Workgroup IDs and retained two-pass reduction: exact with all 256 guard words"
        )
    if args.bounds:
        fixture = ROOT / "eval/frozen/memory512_extent_v1/matrix_extent.json"
        bounds = load_report("memory512_bounds_mps")
        check_bounds(bounds, fixture)
        assert bounds["checkpoint"] == "standalone-resident-memory512"
        if cuda:
            other = load_report("memory512_bounds_9b_cuda_fla")
            check_bounds(other, fixture)
            assert other["checkpoint"] == cuda["merge"]["layout"]
            for field in (
                "trace",
                "schedules",
                "global_words",
                "warp_registers",
                "transport",
                "workgroup_transitions",
            ):
                assert other[field] == bounds[field], field
        print("Raw matrix extent: all 64 lane IDs and 256 untouched words exact")
    if args.public:
        assert cuda is not None
        cases = {case["name"]: case for case in cuda["cases"]}
        for kind, name in (("vector", "vector_64"), ("reduce", "reduce_128")):
            public = load_report(f"memory512_public_{kind}_9b_cuda_fla")
            example = ROOT / "examples" / f"memory512_{name}.json"
            assert (
                public["input_sha256"]
                == hashlib.sha256(example.read_bytes()).hexdigest()
            )
            assert public["merge"] == cuda["merge"]
            assert public["transport_mode"] == "fixed" and public["exact"]
            for field in ("kernel", "output", "global_words"):
                assert public[field] == cases[name][field]
            for i, (run, baseline) in enumerate(
                zip(public["launches"], cases[name]["launches"], strict=True)
            ):
                check_run(run, i == 0)
                assert {
                    k: v for k, v in run.items() if k not in ("audit", "transport")
                } == {
                    k: v for k, v in baseline.items() if k not in ("audit", "transport")
                }
        print(
            "Public fixed transport: vector64 and reduction128 match all planned execution records"
        )
    if args.text:
        assert cuda is not None
        check_text(cuda["merge"], "memory512")


if __name__ == "__main__":
    main()
