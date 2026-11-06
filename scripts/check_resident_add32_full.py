"""Qualify the unified checkpoint against the saved resident reports."""

import json
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
GRID_SNAPSHOT_DIGITS = 65


def load_report(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def cases_by_name(report: dict) -> dict[str, dict]:
    assert report["all_exact"]
    cases = {case["fixture"]: case for case in report["cases"]}
    assert len(cases) == len(report["cases"])
    return cases


def check_suite(report: dict) -> dict[str, dict]:
    cases = cases_by_name(report)
    grid = cases_by_name(load_report("resident_grid_mps"))
    windows = cases_by_name(load_report("resident_windows_mps"))
    wide = cases_by_name(load_report("resident_wide_mps"))
    add32 = cases_by_name(load_report("resident_add32_mps"))
    assert cases.keys() == grid.keys() | windows.keys() | wide.keys() | add32.keys()
    assert len(cases) == 23

    for name, old in {**grid, **windows}.items():
        observed = cases[name]
        transport = old["transport"]
        instructions = old["instructions"]
        assert all(
            word >> 16 == 0 for bank in observed["warp_registers"] for word in bank
        ), name
        assert observed["transport"] == {
            **transport,
            "warp_initial_tokens": transport["warp_initial_tokens"] + 64,
            "input_tokens": transport["input_tokens"] + 64 + 4 * instructions,
            "state_plan_tokens": transport["state_plan_tokens"] + 4 * instructions,
            "high_register_updates": 0,
        }, name
        projected = {
            **observed,
            "trace": [
                {**tick, "snapshot": tick["snapshot"][-GRID_SNAPSHOT_DIGITS:]}
                for tick in observed["trace"]
            ],
            "transport": transport,
        }
        del projected["warp_registers"]
        assert projected == old, name

    for name, old in wide.items():
        assert cases[name] == {
            **old,
            "transport": {**old["transport"], "high_register_updates": 0},
        }, name
    for name, old in add32.items():
        assert cases[name] == old, name

    assert Counter(case["outcome"] for case in cases.values()) == {
        "halted": 21,
        "deadlock": 1,
        "step_limit": 1,
    }
    return cases


def main() -> None:
    mps = load_report("resident_add32_full_mps")
    cuda = load_report("resident_add32_full_9b_cuda_fla")
    cases = check_suite(mps)
    assert cuda["all_exact"] and cuda["cases"] == mps["cases"]
    assert cuda["merge"] == load_report("resident_add32_9b_cuda_fla")["merge"]
    instructions = sum(case["instructions"] for case in cases.values())
    decisions = sum(len(case["schedules"]) for case in cases.values())
    print(
        f"{len(cases)} exact cases, {instructions} instructions, {decisions} decisions"
    )
    print(
        "All inherited traces match their baselines, with identical MPS and CUDA records"
    )


if __name__ == "__main__":
    main()
