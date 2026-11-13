"""Check the resident round-robin schedule, independent masks, and publication."""

import argparse
from pathlib import Path

from check_rom32 import read


def check(report_path, *, deadlock=False, barrier_groups=False):
    report = read(report_path)
    assert report["outcome"] == ("deadlock" if deadlock else "halted")
    assert report["reference"] and all(report["reference"].values())
    assert report["transport"]["fresh_caches"] == 1
    assert report["transport"]["resident_state_exact"]
    assert report["transport"]["resident_memory_exact"]
    assert report["transport"]["resident_rom_exact"]
    assert [step["warp"] for step in report["trace"][:4]] == [0, 1, 2, 3]
    assert [step["workgroup"] for step in report["trace"][:4]] == [0, 0, 1, 1]
    if deadlock:
        assert report["global_words"][:8] == [0xA0A0] * 8
        assert report["global_words"][8:16] == list(range(72, 80))
        assert report["global_words"][16:24] == [0xA0A0] * 8
        assert report["global_words"][24:32] == list(range(88, 96))
        assert all(w == 0xA0A0 for w in report["global_words"][32:])
    elif barrier_groups:
        # One group's barrier releases while the other group is still waiting.
        release = next(
            i
            for i, state in enumerate(report["schedules"])
            if state["ticket"] == 2304 and state["controls"][0] % 128 == 5
        )
        waiting = report["schedules"][release]["controls"][3] % 128
        assert 64 <= waiting < 96
        assert report["global_words"][:32] == [
            lane + (16 if lane % 16 < 8 else 64) for lane in range(32)
        ]
        assert report["global_words"][32:] == [0xA0A0] * 224
    else:
        for group in range(2):
            base = 16 * group
            left = [i + (32 if i % 2 else 64) for i in range(base, base + 8)]
            right = list(range(base + 120, base + 128))
            assert report["global_words"][base : base + 16] == left + right
        assert report["global_words"][32:] == [0xA0A0] * 224
        assert report["transport"]["resident_scoped_exact"]
    print(
        f"{report['device']}: {report['instructions']} steps, {report['outcome']}, four warp banks exact"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--deadlock", action="store_true")
    parser.add_argument("--barrier-groups", action="store_true")
    args = parser.parse_args()
    check(args.report, deadlock=args.deadlock, barrier_groups=args.barrier_groups)
