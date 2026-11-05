"""Explore several valid host-dispatched SIMT program shapes on one model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import transformers

from qwengpu.compiled_add import build_standalone
from qwengpu.eval_program import evaluate_case
from qwengpu.reference import Instruction as I
from qwengpu.reference import LANES


def cases() -> list[tuple[tuple[int, ...], tuple[I, ...]]]:
    rng = random.Random(20261105)
    result = []
    for index in range(32):
        initial = tuple(rng.randrange(256) for _ in range(LANES))
        a, b, c = (rng.randrange(256) for _ in range(3))
        slot = rng.randrange(4) * LANES
        if index % 4 == 0:
            program = (I("ADDI", a), I("XORI", b), I("STORE", slot),
                       I("LOAD", slot), I("ADDI", c), I("HALT"))
        elif index % 4 == 1:
            program = (I("IF_LT", a), I("ADDI", b), I("ENDIF"),
                       I("XORI", c), I("HALT"))
        elif index % 4 == 2:
            program = (I("STORE", slot), I("IF_LT", a), I("XORI", b),
                       I("STORE", slot), I("ELSE"), I("ADDI", c),
                       I("ENDIF"), I("LOAD", slot), I("HALT"))
        else:
            other_slot = ((slot // LANES + 1) % 4) * LANES
            program = (I("STORE", slot), I("IF_LT", a), I("ADDI", b),
                       I("STORE", other_slot), I("ELSE"), I("XORI", c),
                       I("STORE", other_slot), I("ENDIF"),
                       I("LOAD", slot), I("HALT"))
        result.append((initial, program))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                        else "cpu")
    args = parser.parse_args()
    model, circuit = build_standalone()
    model.to(args.device)
    report = {"device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__, "circuit": circuit.stats(),
              "cases": []}
    for index, (initial, program) in enumerate(cases()):
        result = evaluate_case(model, circuit, initial, program,
                               args.device, probe_final=True)
        report["cases"].append(result)
        print(json.dumps({"index": index, "shape": index % 4,
                          "exact": result["exact"],
                          "seconds": result["seconds"]}), flush=True)
    report["all_exact"] = all(case["exact"] for case in report["cases"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_exact": report["all_exact"], "out": str(args.out)}), flush=True)
    if not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
