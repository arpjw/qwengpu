"""Verify scoped memory resets while dependent launches retain global RAM."""

import argparse
import json
from pathlib import Path

from qwengpu.eval_globalgrid import launch
from qwengpu.eval_tile8 import load_tile_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--full-integer", action="store_true")
    parser.add_argument("--resident-generate", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fixture = (
        Path(__file__).resolve().parents[1]
        / "eval/frozen/scoped_memory_v1/isolation.json"
    )
    data = json.loads(fixture.read_bytes())
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
            args.model, args.device, resident_scoped=True, full_integer=args.full_integer,
        )
        offset = meta["chip_token_offset"] if meta else 0

        def run(data, cache=None):
            return launch(model, circuit, data, args.device, offset, cache)

    cache, producer = run(data)
    consumer = {
        "workgroups": data["workgroups"],
        "global_words": producer["global_words"],
        "programs": [
            [{"op": "LOAD_PRIVATE"}, {"op": "LOAD_SHARED"}, {"op": "HALT"}]
            for _ in range(2)
        ],
    }
    retained, observed = run(consumer, cache)
    assert retained is cache
    assert observed["global_words"] == producer["global_words"]
    assert observed["scoped_words"] == [0] * 128
    if args.resident_generate:
        assert observed["registers_exact"]
        assert observed["global_upload_words"] == observed["host_execution_tokens"] == 0
    else:
        assert observed["warp_registers"] == [[0] * 8, [0] * 8]
        transport = observed["transport"]
        assert transport["matrix_initial_tokens"] == transport["matrix_patch_tokens"] == 0
        assert transport["scoped_initial_tokens"] == 512
        assert transport["scoped_patch_tokens"] == 0
    args.out.write_text(
        json.dumps(
            {
                "device": args.device,
                "merge": meta,
                "producer": producer,
                "consumer": observed,
                "exact": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
