"""Independently check all uint32 indexed results and the untouched RAM words."""

import argparse
import hashlib
import json
import operator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "eval/frozen/full_integer_v1/indexed_words.json"
MASK32 = (1 << 32) - 1


def signed(x: int) -> int:
    return x if x < 1 << 31 else x - (1 << 32)


def check(path: Path) -> dict:
    report = json.loads(path.read_bytes())
    payload = FIXTURE.read_bytes()
    assert report["kernel_sha256"] == hashlib.sha256(payload).hexdigest()
    assert report["outcome"] == "halted" and all(report["reference"].values())
    assert all(
        report["transport"][f"resident_{name}_exact"]
        for name in ("memory", "state", "rom")
    )
    words = json.loads(payload)["global_words"]

    def get(i):
        return words[i] | words[i + 1] << 16

    expected = words[:]
    for lane in range(16):
        a, b = get(2 * lane), get(128 + 2 * lane)
        compare = signed(a) < signed(b) if lane < 8 else a < b
        first = operator.mul if lane < 8 else operator.xor
        last = operator.and_ if lane < 8 else operator.or_
        for start, result in ((64, first(a, b)), (96, compare), (32, last(a, b))):
            result &= MASK32
            addr = start + 2 * lane
            expected[addr : addr + 2] = [result & 65535, result >> 16]
    assert report["global_words"] == expected
    print(f"{report['device']}: 48 indexed uint32 results and all 256 global words exact")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    first, *rest = [check(path) for path in args.reports]
    for report in rest:
        assert all(first[key] == report[key] for key in first.keys() - {"device", "checkpoint"})
