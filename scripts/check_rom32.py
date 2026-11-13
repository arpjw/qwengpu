"""Verify saved ROM32 evidence and compare it with the qualified baseline."""

import argparse
import hashlib
import json

from check_addmem32_text import check_reset
from check_memory512 import check as check_memory
from check_resident_add32_full import RESULTS, cases_by_name, load_report

ROOT = RESULTS.parent


def read(path):
    return json.loads(path.read_bytes())


def project_rom16(value):
    """Remove the 32 additional upload tokens and keep all execution records."""
    if isinstance(value, list):
        return [project_rom16(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: project_rom16(item) for key, item in value.items()}
    if "rom_program_tokens" in result:
        assert result["rom_program_tokens"] == 64
        result["rom_program_tokens"] = 32
        result["input_tokens"] -= 32
    return result


def check_new(root, device):
    reports = {}
    for name in ("loops_and_barrier", "workgroup_reset"):
        path = ROOT / "eval/frozen/resident_rom32_v1" / f"{name}.json"
        report = read(root / f"{name}.json")
        reports[name] = report
        assert report["device"] == device and report["outcome"] == "halted"
        assert report["kernel_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert report["instructions"] == 128 and len(report["schedules"]) == 129
        assert report["reference"] and all(report["reference"].values())
        transport = report["transport"]
        assert transport["rom_program_tokens"] == 64 and transport["fresh_caches"] == 1
        assert all(
            transport[f"resident_{field}_exact"]
            for field in ("grid", "memory", "rom", "state")
        )
    loops = reports["loops_and_barrier"]
    assert loops["warp_registers"] == [[496] * 8, [348] + [347] * 7]
    for warp, before, after in ((0, 15, 16), (0, 15, 13), (0, 18, 16), (1, 18, 16)):
        pcs = [0] + [
            (step["control"] & 15) + (16 if step["control"] & 128 else 0)
            for step in loops["trace"]
            if step["warp"] == warp
        ]
        assert (before, after) in zip(pcs, pcs[1:]), (warp, before, after)
    grid = reports["workgroup_reset"]
    assert grid["global_words"] == [51, 52] + [0] * 254
    assert grid["workgroup_cursor"] == 1
    assert len(grid["workgroup_transitions"]) == 1
    transition = grid["workgroup_transitions"][0]
    assert transition["instructions"] == 64 and transition["resident_memory_exact"]
    assert grid["schedules"][transition["schedule_index"]]["controls"] == [1024, 1088]
    print(
        f"{device}: 256 instruction steps, 64 ROM entries per launch, exact control and data"
    )
    return reports


def check_inherited(root):
    memory = read(root / "memory512.json")
    check_memory(memory)
    old_memory = load_report("memory512_mps")
    for field in ("raw", "cases"):
        assert project_rom16(memory[field]) == old_memory[field], field

    inherited = read(root / "inherited.json")
    old_inherited = load_report("memory512_inherited_mps")
    assert project_rom16(cases_by_name(inherited)) == cases_by_name(old_inherited)

    matrix = read(root / "matmul16.json")
    old_matrix = load_report("memory512_matmul16_mps")
    for field in (
        "matrix",
        "passes",
        "transitions",
        "transport",
        "instructions",
        "scheduler_decisions",
        "workgroup_outputs",
        "input_sha256",
        "exact",
    ):
        assert project_rom16(matrix[field]) == old_matrix[field], field
    # Initial call includes both programs. Subsequent operand and C traffic match.
    assert matrix["memory_tokens_by_call"] == old_matrix["memory_tokens_by_call"]
    print(
        "Inherited 27 traces, 31 common kernels, seven raw probes and all eight matrix traces match"
    )
    return memory


def check_merge(root, memory):
    text = read(root / "text.json")
    meta = memory["merge"]
    assert text["merge"] == meta
    assert (
        text["next_top1_same"],
        text["greedy_identical"],
        text["merged_chip_tokens"],
    ) == (24, 21, 0)

    manifest = read(root / "manifest.json")
    assert manifest["files"]["qwengpu9b.json"] == meta
    preflight = load_report("rom32_preflight")
    assert meta["layout"] == "qwengpu-resident-rom32-9b-v1"
    for key, field in (
        ("hidden", "merged_hidden"),
        ("slots", "slots"),
        ("chip_vocab", "vocab"),
        ("circuit", "circuit"),
    ):
        assert meta[key] == preflight[field], key
    files = manifest["files"]
    shards = set(files["model.safetensors.index.json"]["weight_map"].values())
    assert shards and set(manifest["sha256"]) == shards | set(files)

    handoff = read(root / "handoff.json")
    assert handoff["exact"] and handoff["merge"] == meta
    reduction = next(case for case in memory["cases"] if case["name"] == "reduce_127")
    assert [handoff["producer"], handoff["consumer"]] == reduction["launches"]
    assert handoff["sum"] == reduction["output"]
    ram = [
        word >> (4 * nibble) & 15
        for word in handoff["producer"]["global_words"]
        for nibble in range(4)
        for _lane in range(8)
    ]
    assert (
        len(ram) == 8192 and ram == handoff["ram_before"] == handoff["ram_after_text"]
    )
    assert handoff["consumer"]["transport"]["matrix_initial_tokens"] == 0
    check_reset(handoff["text"]["comparison"])
    print(
        f"CUDA merge: {len(shards)} shard hashes, 24/24 first tokens, 21/24 continuations, 8,192 retained RAM nibbles"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-only", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    mps = RESULTS / "rom32_mps"
    standalone = check_new(mps, "mps")
    if not args.target_only:
        check_inherited(mps)
    if args.cuda:
        cuda = RESULTS / "rom32_cuda"
        merged = check_new(cuda, "cuda")
        for name, report in standalone.items():
            other = merged[name]
            assert {
                k: v for k, v in report.items() if k not in ("device", "checkpoint")
            } == {
                k: v for k, v in other.items() if k not in ("device", "checkpoint")
            }, name
        if not args.target_only:
            check_merge(cuda, check_inherited(cuda))


if __name__ == "__main__":
    main()
