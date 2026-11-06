"""Compare resident ADD32I traces, high writes, and tiled regressions."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIXTURES = ROOT / "eval/frozen/resident_add32_gate_v1"
CANONICAL = (
    "outcome",
    "instructions",
    "matrix",
    "shared_words",
    "trace",
    "schedules",
    "warp_registers",
)
HIGH_SHIFT = 260


def read_result(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def check_suite(report: dict) -> dict[str, dict]:
    assert report["all_exact"]
    cases = {case["fixture"]: case for case in report["cases"]}
    assert cases.keys() == {
        "masked_carry.json",
        "cooperative_add32.json",
        "seeded_cooperative.json",
        "dense_seeded.json",
    }
    old = {
        case["fixture"]: case for case in read_result("resident_wide_mps.json")["cases"]
    }
    for name in ("dense_seeded.json", "seeded_cooperative.json"):
        case = cases[name]
        assert case == {
            **old[name],
            "transport": {**old[name]["transport"], "high_register_updates": 0},
        }, name
    for name in ("masked_carry.json", "cooperative_add32.json"):
        payload = (FIXTURES / name).read_bytes()
        case = cases[name]
        assert case["fixture_sha256"] == hashlib.sha256(payload).hexdigest()
        initial = json.loads(payload)["initial_registers"]
        high = [
            sum((word >> 16) << (16 * lane) for lane, word in enumerate(warp))
            for warp in initial
        ]
        changed = 0
        for step in case["trace"]:
            warp = step["warp"]
            current = int(step["snapshot"], 16) >> HIGH_SHIFT
            diff = current ^ high[warp]
            changed += sum(bool((diff >> bit) & 15) for bit in range(0, 128, 4))
            high[warp] = current
        assert changed == case["transport"]["high_register_updates"] > 0
        assert case["transport"]["resident_state_exact"]
        assert case["transport"]["fresh_caches"] == 1
    assert (
        cases["cooperative_add32.json"]["matrix"]
        == old["seeded_cooperative.json"]["matrix"]
    )
    return cases


def main() -> None:
    mps = read_result("resident_add32_mps.json")
    cuda = read_result("resident_add32_9b_cuda_fla.json")
    cases = check_suite(mps)
    assert cuda["all_exact"] and cuda["cases"] == mps["cases"]
    assert cuda["merge"]["layout"] == "qwengpu-resident-add32-9b-v1"

    case = cases["masked_carry.json"]
    for mode in ("fixed", "state-plan"):
        for backend, layout in (
            ("mps", "standalone-resident-add32"),
            ("9b_cuda_fla", cuda["merge"]["layout"]),
        ):
            launch = read_result(f"resident_add32_launch_{mode}_{backend}.json")
            assert launch["checkpoint"] == layout and launch["transport_mode"] == mode
            assert launch["kernel_sha256"] == case["fixture_sha256"]
            assert launch["transport"]["resident_state_exact"]
            assert (
                launch["transport"]["high_register_updates"]
                == case["transport"]["high_register_updates"]
            )
            assert all(launch[key] == case[key] for key in CANONICAL)
    print(
        "Four exact cases across MPS and merged CUDA, including a joint ADD32I and tile kernel"
    )
    print("Both legacy cases match exactly; all three transports agree on high writes")


if __name__ == "__main__":
    main()
