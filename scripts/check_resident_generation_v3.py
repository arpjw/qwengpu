"""Check v3 reports against earlier checkpoints and the native-text gate."""

import argparse
import json
from pathlib import Path

from check_resident_generation import check, check_inherited

RESULTS = Path(__file__).resolve().parents[1] / "results"
V2 = RESULTS / "resident_generation_v2_cuda"
BASELINE = RESULTS / "model_stores_merged_cuda"


def read(directory, filename):
    return json.loads((directory / filename).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="?", default=RESULTS / "resident_generation_v3_cuda")
    root = parser.parse_args().reports
    merged = root / "inherited-merged.json"
    standalone = root / "inherited-standalone.json"
    check_inherited(merged, BASELINE / "inherited.json")
    check_inherited(standalone, BASELINE / "inherited.json")
    for left, right in zip(read(root, merged.name)["cases"],
                           read(root, standalone.name)["cases"], strict=True):
        assert left["fixture"] == right["fixture"]
        assert left["generated_tokens"] == right["generated_tokens"]
    check(RESULTS / "cache_register_edge_cuda", [root / "interleaved-merged.json"])
    check(RESULTS / "cache_register_edge_cuda", [root / "sequential-merged.json"],
          sequential_baseline=BASELINE,
          matrix_extent_baseline=RESULTS / "memory512_bounds_9b_cuda_fla.json")

    for suite, prior in (("bounds.json", read(V2, "bounds.json")["cases"]),
                         ("tiled.json", {key: read(BASELINE, f"tiled_{key}.json")
                                         for key in ("3x5x2", "9x7x5")})):
        got = read(root, suite)["cases"]
        assert got.keys() == prior.keys()
        for name, case in got.items():
            old = prior[name]
            for key in ("kernel", "output", "global_words"):
                assert case[key] == old[key], (name, key)
            assert case["exact"]
            for run, earlier in zip(case["launches"], old["launches"], strict=True):
                assert all(v for k, v in run.items() if k.endswith("_exact"))
                assert run["generation_calls"] == 1 and run["host_execution_tokens"] == 0
                if "tile" in run:
                    assert (run["tile"], run["k_panel"]) == (earlier["tile"], earlier["k_panel"])
                    assert run["global_upload_words"] == (48 if run["k_panel"][0] else 256)
                else:
                    assert run["global_upload_words"] == earlier["global_upload_words"]
            print(f"{name}: scalar result, final memory and upload counts exact")

    manifest = read(root, "manifest.json")
    text = read(root, "text.json")
    handoff = read(root, "handoff.json")
    assert text["merge"] == manifest["files"]["qwengpu9b.json"] == handoff["checkpoint"]
    index = manifest["files"]["model.safetensors.index.json"]
    shards = set(index["weight_map"].values())
    assert shards and set(manifest["sha256"]) == shards | set(manifest["files"])
    assert sum(manifest["dtype_bytes"].values()) == index["metadata"]["total_size"]
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
