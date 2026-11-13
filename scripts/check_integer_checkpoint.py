"""Compare integrated integer checkpoint records with their qualified baselines."""

import argparse
import json
from pathlib import Path

from check_full_integer import check as check_full_integer
from check_interleaved import check as check_interleaved
from check_memory512 import check as check_memory
from check_rom32 import check_new
from check_tiled_compute import check as check_tiled_compute

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_bytes())


def remove_mode_descriptor(value):
    """The new layout adds exactly one mode descriptor per launch."""
    if isinstance(value, list):
        return [remove_mode_descriptor(item) for item in value]
    if isinstance(value, dict):
        normalized = {k: remove_mode_descriptor(v) for k, v in value.items()}
        if "rom_program_tokens" in normalized:
            assert normalized["rom_program_tokens"] == 64
            normalized["input_tokens"] -= 1
        return normalized
    return value


def compare(left, right, exclude=()):
    assert left.keys() - set(exclude) == right.keys() - set(exclude)
    for key in left.keys() - set(exclude):
        assert left[key] == right[key], key


def check(root, baseline, device):
    cases = {}
    for name, options in (
        ("interleaved_publish", {}),
        ("interleaved_barrier_groups", {"barrier_groups": True}),
        ("interleaved_deadlock", {"deadlock": True}),
    ):
        path = root / f"{name}.json"
        check_interleaved(path, **options)
        cases[name] = read(path)
    cases["indexed_words"] = check_full_integer(root / "indexed_words.json")
    cases.update(check_new(root, device))
    for name in ("integer_edges", "address_walk"):
        cases[name] = read(root / f"{name}.json")
    for name in ("integer_edges", "address_walk", "loops_and_barrier", "workgroup_reset"):
        compare(
            remove_mode_descriptor(cases[name]),
            read(baseline / f"{name}.json"),
            ("device", "checkpoint"),
        )
    cases["memory512"] = memory = read(root / "memory512.json")
    check_memory(memory)
    for name in ("raw", "cases", "manifest_sha256"):
        assert remove_mode_descriptor(memory[name]) == read(baseline / "memory512.json")[name], name
    cases["inherited"] = inherited = read(root / "inherited.json")
    assert inherited["all_exact"]
    assert remove_mode_descriptor(inherited["cases"]) == read(baseline / "inherited.json")["cases"]
    cases["matmul16"] = old_tile = read(root / "matmul16.json")
    prior_tile = read(baseline / "matmul16.json")
    for name in ("matrix", "passes", "transitions", "transport", "memory_tokens_by_call"):
        assert remove_mode_descriptor(old_tile[name]) == prior_tile[name], name
    for name in ("isolation", "runtime_extent", "scoped_reset", "interleaved_handoff"):
        cases[name] = read(root / f"{name}.json")
    assert cases["isolation"]["reference"] and all(cases["isolation"]["reference"].values())
    extents = cases["runtime_extent"]["runs"]
    assert cases["runtime_extent"]["exact"] and [r["extent"] for r in extents] == [19, 0, 64]
    words = [0xA51C] * 256
    for index, run in enumerate(extents):
        count, result = run["extent"], run["result"]
        words[:count] = list(range(1, count + 1))
        assert result["global_words"] == words
        assert result["transport"]["matrix_patch_tokens"] == 4 * count
        assert result["audit"][0]["memory"] == (0 if index else 1024)
        assert result["programs"] == extents[0]["result"]["programs"]
    reset = cases["scoped_reset"]
    assert reset["exact"] and reset["consumer"]["global_words"] == reset["producer"]["global_words"]
    assert reset["consumer"]["scoped_words"] == [0] * 128
    assert reset["consumer"]["transport"]["matrix_initial_tokens"] == 0
    publication = cases["interleaved_handoff"]
    assert publication["exact"] and publication["consumer"]["audit"][0]["memory"] == 0
    assert publication["consumer"]["global_words"][64:96] == [
        value + 7 for value in publication["producer"]["global_words"][:32]
    ]
    for shape in ("3x5x2", "9x7x5"):
        name = f"tiled_{shape}"
        cases[name] = check_tiled_compute(
            root / f"{name}.json", ROOT / "eval/frozen/tiled_compute_v1" / f"{shape}.json"
        )
    for name, baseline_name in (
        ("interleaved_publish", "interleaved_publish"),
        ("barrier_groups", "interleaved_barrier_groups"),
        ("deadlock", "interleaved_deadlock"),
    ):
        key = f"fused_{name}"
        cases[key] = fused = read(root / f"{key}.json")
        regular = cases[baseline_name]
        compare(fused, regular, ("transport", "transport_mode"))
        old, new = regular["transport"], fused["transport"]
        assert old["forward_calls"] > new["forward_calls"]
        assert old["input_tokens"] >= new["input_tokens"]
        print(f"{device}: {key}, {old['forward_calls']} to {new['forward_calls']} calls")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mps", type=Path, default=ROOT / "results/final_candidate_mps")
    parser.add_argument("--cuda", type=Path, default=ROOT / "results/final_candidate_cuda")
    parser.add_argument("--only", choices=("mps", "cuda"))
    args = parser.parse_args()
    standalone = None
    if args.only != "cuda":
        standalone = check(args.mps, ROOT / "results/alu32_mps", "mps")
    if args.only == "mps":
        return
    merged = check(args.cuda, ROOT / "results/alu32_cuda", "cuda")
    if standalone is not None:
        for name in standalone:
            compare(standalone[name], merged[name], ("device", "checkpoint", "merge", "torch", "transformers"))
    text = read(args.cuda / "text.json")
    assert (text["next_top1_same"], text["greedy_identical"], text["merged_chip_tokens"]) == (24, 22, 0)
    manifest = read(args.cuda / "manifest.json")
    meta = manifest["files"]["qwengpu9b.json"]
    assert meta == text["merge"] == merged["memory512"]["merge"]
    shards = set(manifest["files"]["model.safetensors.index.json"]["weight_map"].values())
    assert shards and set(manifest["sha256"]) == shards | set(manifest["files"])
    handoff = read(args.cuda / "handoff.json")
    assert handoff["exact"] and handoff["merge"] == meta
    assert len(handoff["ram_before"]) == 8192
    assert handoff["ram_before"] == handoff["ram_after_text"]
    print(f"Merged 9B text, {len(shards)} weight shard hashes, and native-text RAM handoff verified.")


if __name__ == "__main__":
    main()
