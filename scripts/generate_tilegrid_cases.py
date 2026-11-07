"""Generate seeded programs for the tilegrid differential runner."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def program(*instructions: str | tuple[str, int]) -> list[dict]:
    return [
        {"op": instruction}
        if isinstance(instruction, str)
        else {"op": instruction[0], "arg": instruction[1]}
        for instruction in instructions
    ]


def generate(seed: int) -> dict[str, dict]:
    rng = random.Random(seed)
    source = rng.sample(range(64), 8)
    indices = [2 * index for index in source]
    consumer = indices[:]
    rng.shuffle(consumer)
    delta = rng.randrange(64, 128)
    receive = program("SAVE_INDEX", "BARRIER", "LOAD_INDEX", "LOAD_INDEX_HI", "HALT")
    loop = {
        "row_pairs": 1,
        "initial_registers": [[0x1FF80 | index for index in indices], consumer],
        "programs": [
            program(
                "SAVE_INDEX",
                ("WHILE_BIT", 16),
                ("ADD32I", delta),
                "ENDWHILE",
                "STORE_INDEX",
                "STORE_INDEX_HI",
                "BARRIER",
                "HALT",
            ),
            receive,
        ],
    }
    collision = rng.randrange(0, 128, 2)
    indices[1] = indices[0] = collision
    predicate = rng.choice((16, 17, 31))
    values = [((rng.getrandbits(16) << 16) | 0xFF80 | index) for index in indices]
    values[0] = values[0] | (1 << predicate)
    values[1] = values[1] & ~(1 << predicate)
    branch = {
        "row_pairs": 1,
        "initial_registers": [values, indices],
        "programs": [
            program(
                "SAVE_INDEX",
                ("IF_BIT", predicate),
                ("ADD32I", rng.randrange(128, 256)),
                "ELSE",
                ("ADD32I", rng.randrange(1, 128)),
                "ENDIF",
                "STORE_INDEX",
                "STORE_INDEX_HI",
                "BARRIER",
                "HALT",
            ),
            receive,
        ],
    }
    deadlock = {
        "row_pairs": 1,
        "initial_registers": [values, indices],
        "programs": [program(("ADD32I", delta), "BARRIER", "HALT"), program("HALT")],
        "expected_outcome": "deadlock",
    }
    return {"loop": loop, "collision": branch, "deadlock": deadlock}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in args.seed:
        for kind, launch in generate(seed).items():
            path = args.out / f"{seed}_{kind}.json"
            path.write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
