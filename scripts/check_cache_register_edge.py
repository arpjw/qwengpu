"""Check that the runtime tensor edge removes register tokens and preserves traces."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def check(baseline, result):
    assert result["transport_mode"] in ("cache-register-edge", "fused-cache-edge")
    assert result["reference"] and all(result["reference"].values())
    for key in baseline.keys() - {"checkpoint", "device", "transport", "transport_mode"}:
        assert baseline[key] == result[key], key
    old, new = baseline["transport"], result["transport"]
    assert new["resident_state_exact"]
    assert new["state_plan_tokens"] == new["state_read_tokens"] == new["warp_update_tokens"] == 0
    assert new["cache_register_trace_tags"] == old["warp_update_tokens"]
    assert new["cache_register_copies"] > 0
    if "internal_store_commands" in new:
        assert new["internal_store_commands"] == old["internal_store_commands"]
        assert new["matrix_patch_tokens"] == new["internal_store_read_tags"] == 0
        assert new.get("scoped_patch_tokens", 0) == 0
    assert new["input_tokens"] < old["input_tokens"]
    assert new["forward_calls"] <= old["forward_calls"]
    print(
        f"Trace exact, {old['input_tokens']} to {new['input_tokens']} input tokens; "
        f"{old['forward_calls']} to {new['forward_calls']} model calls."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    check(read(args.baseline), read(args.result))
