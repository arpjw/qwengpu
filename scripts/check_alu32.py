"""Compare ALU32 traces with independent scalar results and the ROM32 baseline."""

import argparse
import hashlib

from check_addmem32_text import check_reset
from check_rom32 import RESULTS, ROOT, check_inherited, check_new, read


def check_arithmetic(root):
    reports = {}
    for name in ("integer_edges", "address_walk"):
        fixture = ROOT / "eval/frozen/resident_alu32_v1" / f"{name}.json"
        report = read(root / f"{name}.json")
        assert (
            report["kernel_sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
        )
        assert report["outcome"] == "halted"
        assert report["reference"] and all(report["reference"].values())
        assert report["transport"]["resident_rom_exact"]
        assert report["transport"]["resident_memory_exact"]
        assert report["transport"]["resident_state_exact"]
        reports[name] = report

    mask = 0xFFFFFFFF
    edges = reports["integer_edges"]
    initial = read(ROOT / "eval/frozen/resident_alu32_v1/integer_edges.json")[
        "initial_registers"
    ]

    def signed(v):
        return v if v < 0x80000000 else v - (1 << 32)

    def first(v):
        if signed(v) < -128:
            v = (-(255 - 1) * 31) & mask
            v = mask if v & 1 else 0
        else:
            v = 0
        return int((v >> 16) < 255)

    def second(v):
        v = (((signed(v) >> 4) - 1) * 17) & mask
        return (int(signed(v) < 127) - 128) & mask

    assert edges["warp_registers"] == [
        [first(v) for v in initial[0]],
        [second(v) for v in initial[1]],
    ]
    # The signed comparison drives the mask used by the four subsequent operations.
    signed_cmp = edges["trace"][0]
    assert signed_cmp["warp"] == 0
    assert [word & 15 for word in signed_cmp["registers"][::4]] == [
        int(signed(v) < -128) for v in initial[0]
    ]

    grid = reports["address_walk"]
    words = read(ROOT / "eval/frozen/resident_alu32_v1/address_walk.json")[
        "global_words"
    ]
    initial_words = words[:]
    for group in range(2):
        for lane in range(5):
            base = 2 * (16 * group + lane)
            first, second = (base - 2) % 256, (base - 130) % 256
            lo = (
                (initial_words[first] & initial_words[second]) ^ initial_words[second]
            ) | initial_words[second]
            hi = (
                (initial_words[first + 1] | initial_words[second + 1])
                ^ initial_words[second + 1]
            ) & initial_words[second + 1]
            words[base + 64 : base + 66] = [lo, hi]
        for lane in range(8):
            base = 2 * (16 * group + 8 + lane)
            source = (base - 128) % 256
            lo = (initial_words[source] ^ initial_words[base]) & initial_words[base]
            hi = initial_words[source + 1] | initial_words[base + 1]
            words[base + 64 : base + 66] = [lo, hi]
    assert grid["global_words"] == words
    assert grid["transport"]["matrix_patch_tokens"] == 2 * (5 + 8) * 8
    print(
        f"{edges['device']}: arithmetic edge cases and all 256 address-walk words exact"
    )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--target-only", action="store_true")
    args = parser.parse_args()
    mps = RESULTS / "alu32_mps"
    standalone = check_arithmetic(mps)
    if not args.target_only:
        standalone_rom = check_new(mps, "mps")
        check_inherited(mps)
    if args.cuda:
        cuda = RESULTS / "alu32_cuda"
        merged = check_arithmetic(cuda)
        for name, report in standalone.items():
            fields = report.keys() - {"device", "checkpoint"}
            assert all(report[key] == merged[name][key] for key in fields), name
        if not args.target_only:
            merged_rom = check_new(cuda, "cuda")
            for name, report in standalone_rom.items():
                fields = report.keys() - {"device", "checkpoint"}
                assert all(report[key] == merged_rom[name][key] for key in fields), name
            memory = check_inherited(cuda)
            standalone_memory = read(mps / "memory512.json")
            for field in ("raw", "cases", "manifest_sha256"):
                assert memory[field] == standalone_memory[field], field
            text = read(cuda / "text.json")
            assert text["next_top1_same"] == 24 and text["greedy_identical"] >= 21
            assert text["merged_chip_tokens"] == 0
            manifest = read(cuda / "manifest.json")
            meta = manifest["files"]["qwengpu9b.json"]
            preflight = read(RESULTS / "alu32_preflight.json")
            assert meta == text["merge"] == memory["merge"]
            assert meta["layout"] == "qwengpu-resident-alu32-9b-v1"
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
            handoff = read(cuda / "handoff.json")
            assert handoff["merge"] == memory["merge"] and handoff["exact"]
            assert handoff["ram_before"] == handoff["ram_after_text"]
            assert len(handoff["ram_before"]) == 8192
            primary = next(
                case for case in memory["cases"] if case["name"] == "reduce_127"
            )
            assert [handoff["producer"], handoff["consumer"]] == primary["launches"]
            assert handoff["sum"] == primary["output"]
            check_reset(handoff["text"]["comparison"])
            print(
                "Merged text, shard hashes, and retained RAM across native text verified"
            )


if __name__ == "__main__":
    main()
