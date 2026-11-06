"""Verify the addmem32 text corpus and persistent GPU cache checks."""

import math

from check_resident_add32_full import load_report


def check_reset(comparison: dict) -> None:
    assert comparison["native_next_top1_same"]
    assert comparison["greedy_identical"]
    assert comparison["next_max_abs_logit_difference"] == 0
    assert comparison["next_mean_abs_logit_difference"] == 0


def main() -> None:
    layout = load_report("resident_addmem32_full_9b_cuda_fla")["merge"]
    assert layout["layout"] == "qwengpu-resident-addmem32-9b-v1"
    fresh, reset, roundtrip = (
        load_report(f"resident_addmem32_{name}_9b_cuda_fla")
        for name in ("text", "handoff_reset", "cache_relaunch")
    )
    assert all(report["merge"] == layout for report in (fresh, reset, roundtrip))

    control = load_report("resident_add32_text_9b_cuda_h100_control")
    old_fresh = load_report("resident_add32_text_9b_cuda_fla")
    assert control["merge"] == old_fresh["merge"]
    for key in (
        "prompts_sha256",
        "tokenized_prompts_sha256",
        "max_new_tokens",
        "generation_config",
        "prompt_count",
    ):
        assert fresh[key] == control[key] == old_fresh[key]
    records = fresh["records"]
    assert len(records) == fresh["prompt_count"] == 24
    for key in ("next_top1_same", "greedy_identical", "merged_chip_tokens"):
        assert fresh[key] == sum(record[key] for record in records)
    assert fresh["merged_chip_tokens"] == 0
    assert fresh == {**control, "merge": layout}
    assert all(
        math.isfinite(record["next_max_abs_logit_difference"]) for record in records
    )
    print(
        f"Fresh text: {fresh['next_top1_same']}/24 first tokens, "
        f"{fresh['greedy_identical']}/24 greedy continuations. "
        f"Same-H100 ADD32 control: {control['next_top1_same']}/24 first tokens, "
        f"{control['greedy_identical']}/24 greedy continuations"
    )

    old_reset = load_report("resident_add32_handoff_reset_9b_cuda_fla")
    for key in (
        "kernel_sha256",
        "corpus_sha256",
        "max_new_tokens",
        "eos_token_ids",
        "cache_policy",
    ):
        assert reset[key] == old_reset[key]
    checked = 0
    for record, old in zip(reset["records"], old_reset["records"], strict=True):
        for key in ("prompt_index", "prompt_ids", "gpu", "gpu_rows"):
            assert record[key] == old[key]
        check_reset(record["reset_vs_fresh"])
        for rows in record["gpu_rows"].values():
            assert rows["exact"] and rows["before_sha256"] == rows["after_sha256"]
            checked += rows["count"]
    assert len(reset["records"]) == 4 and checked == 18_944
    print(f"Text reset: four exact prompts and {checked:,} GPU nibbles preserved")

    old_roundtrip = load_report("resident_add32_cache_relaunch_9b_cuda_fla")
    for key in (
        "producer_sha256",
        "consumer_sha256",
        "producer",
        "gpu_rows_preserved_after_text",
        "retained_transport",
        "cold_transport",
        "shared_tokens_by_call",
        "consumer",
    ):
        assert roundtrip[key] == old_roundtrip[key]
    check_reset(roundtrip["text"]["comparison"])
    assert roundtrip["retained_transport"]["matrix_initial_tokens"] == 0
    print("GPU-to-text-to-GPU: exact inherited trace, zero shared-memory uploads")


if __name__ == "__main__":
    main()
