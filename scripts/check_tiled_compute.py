"""Check tiled matrix outputs, guard words and actual retained-panel upload tags."""

import argparse
import hashlib
import json
from pathlib import Path

MASK32 = (1 << 32) - 1


def check(report_path: Path, fixture: Path) -> dict:
    payload = fixture.read_bytes()
    request = json.loads(payload)
    report = json.loads(report_path.read_bytes())
    assert report["input_sha256"] == hashlib.sha256(payload).hexdigest()
    assert report["exact"] and report["kernel"] == "matmul_tiled"
    a, b, c = (request[key] for key in ("a", "b", "c"))
    m, k, n = len(a), len(b), len(b[0])

    def scalar(i, j, end):
        return (c[i][j] + sum(a[i][t] * b[t][j] for t in range(end))) & MASK32

    assert report["output"] == [
        [scalar(i, j, k) for j in range(n)] for i in range(m)
    ]
    tiles = []
    runs = iter(report["launches"])
    steps = 0
    for row in range(0, m, 8):
        for col in range(0, n, 4):
            rows = range(row, min(row + 8, m))
            cols = range(col, min(col + 4, n))
            for start in range(0, k, 4):
                end = min(start + 4, k)
                run = next(runs)
                assert run["tile"] == [row, col] and run["k_panel"] == [start, end]
                transport = run["transport"]
                assert run["outcome"] == "halted"
                assert all(
                    transport[f"resident_{kind}_exact"]
                    for kind in ("grid", "memory", "rom", "state")
                )
                assert transport["fresh_caches"] == int(start == 0)
                assert transport["matrix_initial_tokens"] == (0 if start else 1024)
                assert run["audit"][0]["memory"] == (192 if start else 1024)
                left = [a[i][t] for i in rows for t in range(start, end)]
                right = [b[t][j] for t in range(start, end) for j in cols]
                prefix = left + [0xAB00 + i for i in range(32 - len(left))]
                prefix += right + [0xBC00 + i for i in range(16 - len(right))]
                if start:
                    expected_tags = [
                        0x20000 + 64 * i + 16 * nib + ((word >> (4 * nib)) & 15)
                        for i, word in enumerate(prefix)
                        for nib in range(4)
                    ]
                    assert transport["global_refresh_tokens"] == 192
                    assert run["audit"][0]["global_refresh_tags"] == expected_tags
                expected = prefix + [0xCD00 + i for i in range(16)]
                values = [scalar(i, j, end) for i in rows for j in cols]
                values += [0xD00D0000 + i for i in range(32 - len(values))]
                expected += [part for v in values for part in (v & 65535, v >> 16)]
                expected += [0xBEEF] * 128
                assert run["global_words"] == expected
                steps += len(run["trace"])
            tiles.append({"tile": [row, col], "words": expected})
    assert next(runs, None) is None and report["global_words"] == tiles
    print(
        f"{report['device']}: {m}x{k}x{n}, {len(tiles)} tiles, "
        f"{len(report['launches'])} panels, {steps} steps, all output and guard words exact"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    reference, *other = [check(path, args.fixture) for path in args.reports]
    for report in other:
        for field in ("output", "launches", "global_words"):
            assert report[field] == reference[field], field


if __name__ == "__main__":
    main()
