"""Check shared visibility and private isolation from a frozen inference trace."""

import argparse
import hashlib

from check_rom32 import ROOT, read


def check(path):
    fixture = ROOT / "eval/frozen/scoped_memory_v1/isolation.json"
    inputs = read(fixture)
    result = read(path)
    assert result["kernel_sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert result["outcome"] == "halted" and all(result["reference"].values())
    transport = result["transport"]
    assert transport["resident_scoped_exact"]
    assert transport["resident_memory_exact"]
    assert transport["scoped_initial_tokens"] == 512
    assert transport["scoped_patch_tokens"] == 4 * 4 * (8 + 8 + 8)
    assert transport["matrix_patch_tokens"] == 4 * 4 * (8 + 8 + 8)
    expected_scoped = inputs["scoped_words"][:]
    expected_global = inputs["global_words"][:]
    for group in range(4):
        for lane in range(8):
            base = 16 * group
            private0, private1 = base + lane + 100, base + lane + 208
            expected_scoped[64 + base + lane] = private0
            expected_scoped[72 + base + lane] = private1
            expected_scoped[base + ((120 + lane) & 15)] = private0 + 20
            expected_global[base + lane] = private0
            expected_global[128 + base + lane] = private0 + 20
            expected_global[64 + base + lane] = private1
    assert result["global_words"] == expected_global
    assert result["scoped_words"] == expected_scoped
    print(f"{result['device']}: all global, shared, and private words exact")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="+")
    for path in parser.parse_args().report:
        from pathlib import Path

        check(Path(path))
