"""Run a bounded host-dispatched SIMT program from a JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwengpu.reference import Instruction, LANES


SUPPORTED = frozenset({"ADDI", "XORI", "IF_LT", "ELSE", "ENDIF",
                       "STORE", "LOAD", "HALT"})
MEMORY_ADDRESSES = frozenset({0, 8, 16, 24})


def parse_kernel(data: object) -> tuple[tuple[int, ...], tuple[Instruction, ...]]:
    if not isinstance(data, dict):
        raise ValueError("kernel must be a JSON object")
    raw_initial = data.get("initial")
    raw_program = data.get("program")
    if (not isinstance(raw_initial, list) or len(raw_initial) != LANES or
            any(type(value) is not int or not 0 <= value <= 255
                for value in raw_initial)):
        raise ValueError("initial must contain exactly eight unsigned bytes")
    if not isinstance(raw_program, list) or not 1 <= len(raw_program) <= 64:
        raise ValueError("program must contain 1 through 64 instructions")
    program = []
    depth = 0
    saw_else = False
    for index, raw in enumerate(raw_program):
        if not isinstance(raw, dict) or not isinstance(raw.get("op"), str):
            raise ValueError(f"instruction {index} must have an op")
        op = raw["op"]
        arg = raw.get("arg", 0)
        if op not in SUPPORTED or type(arg) is not int or not 0 <= arg <= 255:
            raise ValueError(f"unsupported opcode or argument at instruction {index}")
        if op in {"STORE", "LOAD"} and arg not in MEMORY_ADDRESSES:
            raise ValueError("compiled scratch addresses are 0, 8, 16, or 24")
        if op == "IF_LT":
            if depth:
                raise ValueError("compiled circuit supports one branch level")
            depth = 1
            saw_else = False
        elif op == "ELSE":
            if depth != 1 or saw_else:
                raise ValueError("ELSE must occur once inside IF_LT")
            saw_else = True
        elif op == "ENDIF":
            if depth != 1:
                raise ValueError("ENDIF must close IF_LT")
            depth = 0
        elif op == "HALT" and (index != len(raw_program) - 1 or depth):
            raise ValueError("HALT must be the final instruction after ENDIF")
        program.append(Instruction(op, arg))
    if depth or program[-1].op != "HALT":
        raise ValueError("program must close its branch and end with HALT")
    return tuple(raw_initial), tuple(program)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--model", type=Path,
                        help="optional merged Qwen3.5-9B checkpoint")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    initial, program = parse_kernel(json.loads(args.kernel.read_text(encoding="utf-8")))

    import torch
    import transformers

    from chipformer.q35.ir import Target

    from qwengpu.compiled_add import VOCAB, build, build_standalone
    from qwengpu.eval_program import evaluate_case

    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    if args.model:
        meta = json.loads((args.model / "qwengpu9b.json").read_text(encoding="utf-8"))
        if int(meta["chip_vocab"]) != VOCAB:
            raise ValueError("merged checkpoint uses a different SIMT token vocabulary")
        offset = int(meta["chip_token_offset"])
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16).to(device).eval()
        cfg = model.config
        circuit = build(Target(cfg.linear_num_key_heads,
                               cfg.linear_num_value_heads,
                               cfg.intermediate_size))
    else:
        model, circuit = build_standalone()
        model.to(device)
        offset = 0
    result = evaluate_case(model, circuit, initial, program, device,
                           offset, probe_final=True)
    report = {"device": device, "model": str(args.model) if args.model else "standalone",
              "circuit": circuit.stats(), **result}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exact": result["exact"],
                      "final_lanes": result["final_lanes"] if result["exact"] else None,
                      "seconds": result["seconds"],
                      "out": str(args.out) if args.out else None}), flush=True)
    if not result["exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
