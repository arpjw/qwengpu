"""Consume a publication from two resident workgroups without re-uploading RAM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from qwengpu.assemble import assemble
from qwengpu.eval_globalgrid import launch
from qwengpu.eval_tile8 import load_tile_model

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "eval/frozen/resident_interleaved_v1/interleaved_publish.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--full-integer", action="store_true")
    parser.add_argument("--resident-generate", action="store_true")
    parser.add_argument("--device", default="mps", choices=("cuda", "mps"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = FIXTURE.read_bytes()
    initial = json.loads(payload)
    meta = (
        json.loads((args.model / "qwengpu9b.json").read_bytes()) if args.model else None
    )
    if args.resident_generate:
        from qwengpu.eval_resident_generation import launch_compute, load_generation_model

        model, circuit, offset = load_generation_model(args.model, args.device)

        def run(data, cache=None):
            return launch_compute(
                model, circuit, data, args.device, offset, cache,
                transport_mode="stock_generation",
            )
    else:
        model, circuit = load_tile_model(
            args.model, args.device, resident_interleaved=True, full_integer=args.full_integer,
        )
        offset = meta["chip_token_offset"] if meta else 0

        def run(data, cache=None):
            return launch(model, circuit, data, args.device, offset, cache)

    cache, producer = run(initial)
    body = [
        "GLOBAL_ID", "SAVE_INDEX", "LOAD_INDEX", "ADD32I 7",
        "ADDIDX 64", "STORE_INDEX", "HALT",
    ]
    program = "\n".join(
        f".warp {warp}\n" + "\n".join(body) for warp in range(2)
    )
    dependent = assemble(
        program,
        {"workgroups": 2, "global_words": producer["global_words"], "interleaved": True},
        resident_interleaved=True,
    )
    retained, consumer = run(dependent, cache)
    expected = initial["global_words"][:]
    for group in range(2):
        base = 16 * group
        expected[base : base + 16] = [
            i + (32 if i % 2 else 64) for i in range(base, base + 8)
        ] + list(range(base + 120, base + 128))
    expected[64:96] = [value + 7 for value in expected[:32]]
    assert retained is cache
    assert consumer["global_words"] == expected
    if args.resident_generate:
        assert consumer["registers_exact"]
        assert consumer["global_upload_words"] == consumer["host_execution_tokens"] == 0
    else:
        transport = consumer["transport"]
        assert not transport["fresh_caches"]
        assert not transport["matrix_initial_tokens"] and not consumer["audit"][0]["memory"]
        assert consumer["warp_registers"] == [
            expected[i : i + 8] for i in range(64, 96, 8)
        ]
    report = {
        "device": args.device,
        "merge": meta,
        "fixture_sha256": hashlib.sha256(payload).hexdigest(),
        "producer": producer,
        "consumer": consumer,
        "exact": True,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("All 32 published values consumed by four resident warps without global RAM upload")


if __name__ == "__main__":
    main()
