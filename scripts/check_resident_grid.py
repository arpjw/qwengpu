"""Compare the saved resident grid reports against the prior checkpoint."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIXTURES = ROOT / "eval" / "frozen" / "grid_gate_v1"
LAYOUT = "qwengpu-resident-grid9b-v1"
TRANSPORTS = ("fixed", "state-plan", "state-store-plan")
CANONICAL = ("outcome", "instructions", "matrix", "shared_words", "trace", "schedules")


def check_grid() -> None:
    mps, cuda = (
        json.loads((RESULTS / f"resident_grid_{backend}.json").read_text())
        for backend in ("mps", "9b_cuda_fla")
    )
    assert mps["all_exact"] and cuda["all_exact"]
    assert cuda["merge"]["layout"] == LAYOUT
    assert mps["cases"] == cuda["cases"]
    old = json.loads((RESULTS / "resident_io_mps.json").read_text())["cases"]
    observed = {case["fixture"]: case for case in mps["cases"]}
    assert len(old) == 12 and len(observed) == 15
    for baseline in old:
        case = observed[baseline["fixture"]]
        transport = baseline["transport"]
        assert case == {
            **baseline,
            "transport": {
                **transport,
                "input_tokens": transport["input_tokens"] + 1,
                "grid_initial_tokens": 1,
                "resident_grid_exact": True,
            },
        }, case["fixture"]

    fixtures = sorted(FIXTURES.glob("*.json"))
    assert len(fixtures) == 3
    for fixture in fixtures:
        payload = fixture.read_bytes()
        launch = json.loads(payload)
        pairs = launch["row_pairs"]
        case = observed[fixture.name]
        assert case["fixture_sha256"] == hashlib.sha256(payload).hexdigest()
        assert case["outcome"] == "halted" and case["instructions"] == 29 * pairs
        assert case["matrix"][2 * pairs :] == [
            launch["c"][i : i + 8] for i in range(16 * pairs, 64, 8)
        ], fixture.name
        assert all(
            case["transport"][f"resident_{bank}_exact"]
            for bank in ("grid", "memory", "rom", "state")
        )
        assert case["transport"]["fresh_caches"] == 1

    case = observed["partial_1_pair.json"]
    for mode in TRANSPORTS:
        path = RESULTS / f"resident_grid_launch_{mode}_9b_cuda_fla.json"
        launch = json.loads(path.read_text())
        assert launch["checkpoint"] == LAYOUT
        assert launch["kernel_sha256"] == case["fixture_sha256"]
        assert launch["transport_mode"] == mode and launch["row_pairs"] == 1
        assert launch["transport"]["resident_grid_exact"]
        assert all(launch[key] == case[key] for key in CANONICAL), mode

    instructions = sum(case["instructions"] for case in observed.values())
    decisions = sum(len(case["schedules"]) for case in observed.values())
    print(
        f"{len(observed)} exact cases, {instructions} instructions, {decisions} decisions"
    )
    print("12 prior traces unchanged; three partial extents and three transports exact")


if __name__ == "__main__":
    check_grid()
