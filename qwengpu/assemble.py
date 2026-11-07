"""Assemble two structured warp programs for the resident tilepipe ISA v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwengpu.eval_tile8 import parse_launch


def assemble(source: str, data: dict | None = None) -> dict:
    if data is not None and (not isinstance(data, dict) or "programs" in data):
        raise ValueError("launch data must be an object without a programs field")
    programs: dict[int, list[dict[str, str | int]]] = {}
    warp = None
    for number, line in enumerate(source.splitlines(), 1):
        fields = line.partition("#")[0].split()
        if not fields:
            continue
        op, *args = fields
        if op == ".warp":
            if len(args) != 1 or args[0] not in ("0", "1") or int(args[0]) in programs:
                raise ValueError(f"line {number}: expected a unique .warp 0 or .warp 1")
            warp = int(args[0])
            programs[warp] = []
            continue
        if warp is None or len(args) > 1:
            raise ValueError(f"line {number}: expected an instruction in a warp section")
        instruction: dict[str, str | int] = {"op": op.upper()}
        if args:
            try:
                instruction["arg"] = int(args[0], 0)
            except ValueError as exc:
                raise ValueError(f"line {number}: expected an integer operand") from exc
        programs[warp].append(instruction)
    if programs.keys() != {0, 1}:
        raise ValueError("expected two warp sections")
    launch = {**(data or {}), "programs": [programs[0], programs[1]]}
    parse_launch(launch, resident_loop32=True)
    return launch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--data", type=Path, help="optional JSON with initial launch data")
    parser.add_argument("--out", type=Path, help="write JSON to a file instead of stdout")
    args = parser.parse_args()
    try:
        data = None
        if args.data:
            data = json.loads(args.data.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("launch data must be a JSON object")
        launch = assemble(args.source.read_text(encoding="utf-8"), data)
        output = json.dumps(launch, indent=2) + "\n"
        if args.out:
            args.out.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
