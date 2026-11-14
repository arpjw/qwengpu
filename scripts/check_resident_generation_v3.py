"""Check v3 reports against earlier checkpoints and the native-text gate."""

import argparse
import json
from pathlib import Path

from check_resident_generation import check, check_inherited

RESULTS = Path(__file__).resolve().parents[1] / "results"
V2 = RESULTS / "resident_generation_v2_cuda"
BASELINE = RESULTS / "model_stores_merged_cuda"
PINNED_V3 = RESULTS / "resident_generation_v3_cuda" / "manifest.json"


def read(directory, filename):
    return json.loads((directory / filename).read_text())


def check_manifest(manifest):
    index = manifest["files"]["model.safetensors.index.json"]
    shards = set(index["weight_map"].values())
    assert shards and set(manifest["sha256"]) == shards | set(manifest["files"])
    assert sum(manifest["dtype_bytes"].values()) == index["metadata"]["total_size"]
    pinned = json.loads(PINNED_V3.read_text())
    for key in ("checkpoint_source_commit", "compiler_commit", "base_revision",
                "files", "sha256", "dtype_bytes", "non_bf16"):
        assert manifest[key] == pinned[key], f"v3 checkpoint differs from pinned manifest in {key}"
    print(f"Verified the pinned v3 manifest and {len(shards)} weight shard hashes")


def check_launch(run):
    checks = [value for key, value in run.items() if key.endswith("_exact")]
    assert checks and all(checks)
    assert run["generated_tokens"] == run["expected_tokens"]
    assert run["generation_calls"] == 1 and run["host_execution_tokens"] == 0


def check_memory512(root):
    baseline = read(BASELINE, "memory512.json")
    assert baseline["all_exact"]
    raw = {case["fixture"]: case for case in read(root, "memory512-raw.json")["cases"]}
    assert len(raw) == len(baseline["raw"])
    for old in baseline["raw"]:
        case = raw.pop(old["name"] + ".json")
        check_launch(case)
        assert case["sha256"] == old["fixture_sha256"]
        assert case["steps"] == len(old["trace"])
        assert case["outcome"] == old["outcome"]
        assert case["global_words"] == old["global_words"]

    cases = read(root, "memory512-compute.json")["cases"]
    assert cases.keys() == {case["name"] for case in baseline["cases"]}
    for old in baseline["cases"]:
        case = cases[old["name"]]
        assert case["exact"] and old["exact"]
        for key in ("kernel", "output", "global_words"):
            assert case[key] == old[key], (old["name"], key)
        for index, (run, earlier) in enumerate(zip(case["launches"], old["launches"], strict=True)):
            check_launch(run)
            assert run["steps"] == len(earlier["trace"])
            assert run["outcome"] == earlier["outcome"]
            assert run["global_words"] == earlier["global_words"]
            assert run["global_upload_words"] == (0 if index else 256)
    print(f"{len(cases)} common compute requests and {len(baseline['raw'])} raw RAM cases exact")


def check_integer_programs(root):
    cases = {run["fixture"]: run for run in read(root, "integer-programs.json")["cases"]}
    assert set(cases) == {"indexed_words.json", "integer_edges.json", "loops_and_barrier.json"}
    for name, run in cases.items():
        old = read(BASELINE, name)
        check_launch(run)
        assert run["sha256"] == old["kernel_sha256"]
        assert run["steps"] == old["instructions"] and run["outcome"] == old["outcome"]
        memory_key = "global_words" if "global_words" in run else "tile_words"
        assert run[memory_key] == old["shared_words"]
    print("Full-width indexed arithmetic, signed edges and upper-half ROM loop exact")


def check_relaunches(root):
    for name in ("scoped_reset", "interleaved_handoff"):
        got = read(root, name + ".json")
        old = read(BASELINE, name + ".json")
        assert got["exact"] and old["exact"]
        for phase in ("producer", "consumer"):
            new_run, old_run = got[phase], old[phase]
            check_launch(new_run)
            assert new_run["global_words"] == old_run["global_words"]
            assert new_run["scoped_words"] == old_run.get("scoped_words")
            assert new_run["steps"] == len(old_run["trace"])
        assert got["consumer"]["global_upload_words"] == 0
        assert got["consumer"]["cache_retained_exact"]
        print(f"{name}: state and retained global memory exact")


def check_matmul16(root):
    got = read(root, "matmul16.json")
    old = read(BASELINE, "matmul16.json")
    assert got["input_sha256"] == old["input_sha256"]
    assert got["matrix"] == old["matrix"] and got["retain_c"]
    assert got["fresh_caches"] == 1
    assert [panel["k"] for panel in got["passes"]] == [0, 8]
    for panel in got["passes"]:
        bank = panel["k"] // 8
        assert len(panel["launches"]) == 4
        for group, run in enumerate(panel["launches"]):
            check_launch(run)
            earlier = old["passes"][2 * group + bank]
            assert (earlier["workgroup"], earlier["bank"]) == (group, bank)
            assert run["steps"] == earlier["instructions"]
            assert run["tile_words"][64:] == [v for row in earlier["matrix"] for v in row]
            assert run["tile_output_upload_words"] == (0 if bank else 64)
            assert run["cache_retained_exact"]
    print("16x16 modular multiply: eight panels and retained output tiles exact")


def check_compute(root, *, mps):
    tiled = {key: read(BASELINE, f"tiled_{key}.json") for key in ("3x5x2", "9x7x5")}
    suites = [("tiled.json", tiled)]
    if not mps:
        suites.append(("bounds.json", read(V2, "bounds.json")["cases"]))
    for suite, prior in suites:
        got = read(root, suite)["cases"]
        if suite == "tiled.json" and "reduction" in got:
            reduction = got.pop("reduction")
            assert reduction["exact"] and reduction["kernel"] == "reduce"
            assert [run["global_upload_words"] for run in reduction["launches"]] == [256, 0]
            for run in reduction["launches"]:
                check_launch(run)
        assert got.keys() == prior.keys()
        for name, case in got.items():
            old = prior[name]
            for key in ("kernel", "output", "global_words"):
                assert case[key] == old[key], (name, key)
            assert case["exact"]
            for run, earlier in zip(case["launches"], old["launches"], strict=True):
                check_launch(run)
                if "tile" in run:
                    assert (run["tile"], run["k_panel"]) == (earlier["tile"], earlier["k_panel"])
                    assert run["global_upload_words"] == (48 if run["k_panel"][0] else 256)
                else:
                    assert run["global_upload_words"] == earlier["global_upload_words"]
            print(f"{name}: scalar result, final memory and upload counts exact")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="?", default=RESULTS / "resident_generation_v3_cuda")
    parser.add_argument("--mps", action="store_true", help="check standalone Apple MPS reports")
    args = parser.parse_args()
    root = args.reports
    backend = "mps" if args.mps else "merged"
    inherited = root / f"inherited-{backend}.json"
    assert read(root, inherited.name)["device"] == ("mps" if args.mps else "cuda")
    check_inherited(inherited, BASELINE / "inherited.json")
    if args.mps:
        comparison = PINNED_V3.with_name("inherited-merged.json")
    else:
        comparison = root / "inherited-standalone.json"
        check_inherited(comparison, BASELINE / "inherited.json")
    by_fixture = {case["fixture"]: case for case in read(comparison.parent, comparison.name)["cases"]}
    for case in read(root, inherited.name)["cases"]:
        assert case["generated_tokens"] == by_fixture.pop(case["fixture"])["generated_tokens"]
    assert not by_fixture
    check(RESULTS / "cache_register_edge_cuda", [root / f"interleaved-{backend}.json"])
    check(RESULTS / "cache_register_edge_cuda", [root / f"sequential-{backend}.json"],
          sequential_baseline=BASELINE,
          matrix_extent_baseline=RESULTS / "memory512_bounds_9b_cuda_fla.json")
    check_compute(root, mps=args.mps)
    check_memory512(root)
    check_integer_programs(root)
    check_relaunches(root)
    check_matmul16(root)
    if args.mps:
        return

    manifest = read(root, "manifest.json")
    text = read(root, "text.json")
    handoff = read(root, "handoff.json")
    assert text["merge"] == manifest["files"]["qwengpu9b.json"] == handoff["checkpoint"]
    check_manifest(manifest)
    assert text["next_top1_same"] == 24 and text["greedy_identical"] >= 21
    assert text["merged_chip_tokens"] == 0
    assert handoff["exact"] and handoff["consumer"]["global_upload_words"] == 0
    assert handoff["sum"] == read(V2, "handoff.json")["sum"]
    assert len(handoff["native_text"]) == 4 and all(r["gpu_rows_exact"] for r in handoff["native_text"])
    reduction = read(root, "cli-reduction.json")
    assert reduction["output"] == read(V2, "bounds.json")["cases"]["reduce_128"]["output"]
    assert reduction["transport_mode"] == "stock-generation"
    assert [r["global_upload_words"] for r in reduction["launches"]] == [256, 0]
    assert all(r["generation_calls"] == 1 and r["host_execution_tokens"] == 0
               for r in reduction["launches"])
    deadlock = read(root, "cli-deadlock.json")
    assert deadlock["outcome"] == "deadlock" and all(deadlock["reference"].values())
    assert deadlock["generated_tokens"] == deadlock["expected_tokens"]
    print("Native text, retained handoff and public CLI exact")


if __name__ == "__main__":
    main()
