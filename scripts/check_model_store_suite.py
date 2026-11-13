"""Check the model-owned store checkpoint against the preceding trace suite."""

import argparse
import json
from pathlib import Path

from check_model_stores import check as check_stores
from check_tiled_compute import check as check_tiles

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "results"
TRANSPORT_FIELDS = {"transport", "audit", "transport_mode"}
METADATA_FIELDS = {"device", "merge", "checkpoint", "torch", "transformers"}


def read(root, name):
    return json.loads((root / f"{name}.json").read_bytes())


def execution(value):
    if isinstance(value, list):
        return [execution(item) for item in value]
    if isinstance(value, dict):
        return {
            key: execution(item)
            for key, item in value.items()
            if key not in TRANSPORT_FIELDS | METADATA_FIELDS
        }
    return value


def check(root, baseline):
    for name in (
        "address_walk", "indexed_words", "integer_edges", "inherited",
        "interleaved_barrier_groups", "interleaved_deadlock", "interleaved_publish",
        "isolation", "loops_and_barrier", "matmul16", "memory512",
        "runtime_extent", "scoped_reset", "interleaved_handoff", "workgroup_reset",
        "fused_interleaved_publish", "fused_barrier_groups", "fused_deadlock",
    ):
        old, new = read(baseline, name), read(root, name)
        assert execution(old) == execution(new), name
    print("All inherited, arithmetic, control, RAM and dependent trace bodies match.")
    for shape in ("3x5x2", "9x7x5"):
        name = f"tiled_{shape}"
        path = root / f"{name}.json"
        check_tiles(path, ROOT / "eval/frozen/tiled_compute_v1" / f"{shape}.json")
        assert execution(read(baseline, name)) == execution(read(root, name)), name
    check_stores(root / "model_stores.json")
    text = read(root, "text")
    assert text["next_top1_same"] == 24 and text["greedy_identical"] >= 21
    assert text["merged_chip_tokens"] == 0
    manifest = read(root, "manifest")
    files = manifest["files"]
    assert files["qwengpu9b.json"] == text["merge"]
    shards = set(files["model.safetensors.index.json"]["weight_map"].values())
    assert set(manifest["sha256"]) == shards | set(files)
    handoff = read(root, "handoff")
    assert handoff["exact"] and handoff["merge"] == text["merge"]
    assert handoff["ram_before"] == handoff["ram_after_text"] and len(handoff["ram_before"]) == 8192
    assert handoff["consumer"]["transport"]["matrix_initial_tokens"] == 0
    print(f"Native text, {len(shards)} weight shard hashes and RAM retention verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=REPORTS / "model_stores_merged_cuda")
    parser.add_argument("--baseline", type=Path, default=REPORTS / "final_candidate_cuda")
    args = parser.parse_args()
    check(args.reports, args.baseline)
