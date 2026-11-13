"""Run the combined GPU's feature gates using one standalone or merged model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwengpu.eval_globalgrid import launch
from qwengpu.eval_tile8 import execute, load_tile_model, oracle, parse_launch
from qwengpu.run_compute import execute_request, parse_request

ROOT = Path(__file__).resolve().parents[1] / "eval/frozen"
GATES = (
    ("resident_interleaved_v1", "interleaved_publish"),
    ("resident_interleaved_v1", "barrier_groups"),
    ("resident_interleaved_v1", "deadlock"),
    ("full_integer_v1", "indexed_words"),
    ("scoped_memory_v1", "isolation"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="mps", choices=("mps", "cuda"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = (
        json.loads((args.model / "qwengpu9b.json").read_bytes()) if args.model else None
    )
    model, circuit = load_tile_model(args.model, args.device, full_integer=True)
    offset = meta["chip_token_offset"] if meta else 0
    args.out.mkdir(parents=True, exist_ok=True)

    def save(name, result):
        (args.out / f"{name}.json").write_text(
            json.dumps({"device": args.device, "merge": meta, "result": result}, indent=2)
            + "\n"
        )

    for group, name in GATES:
        data = json.loads((ROOT / group / f"{name}.json").read_bytes())
        if data.get("interleaved"):
            case = parse_launch(data, full_integer=True)
            expected = oracle(case, 256, looping=True)
            result = execute(
                model, circuit, case, args.device, 256, offset,
                resident_add32=True, resident_io=True, fused_dispatch=True,
            )
            assert all(result[key] == value for key, value in expected.items())
            for field in ("grid", "memory", "rom", "state"):
                assert result["transport"][f"resident_{field}_exact"]
        else:
            _, result = launch(model, circuit, data, args.device, offset)
        save(name, result)
        print(name, result["outcome"], len(result["trace"]), flush=True)

    for shape in ("3x5x2", "9x7x5"):
        data = json.loads((ROOT / "tiled_compute_v1" / f"{shape}.json").read_bytes())
        result = execute_request(
            model, circuit, parse_request(data, memory512=True), args.device, offset
        )
        save(f"tiled_{shape}", result)
        print(shape, "exact", flush=True)


if __name__ == "__main__":
    main()
