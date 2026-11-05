"""Run the model-scheduled tile workgroup and compare it with a scalar oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
from chipformer.q35.ir import Target

from qwengpu.compiled_tile import (
    CONTROL,
    DEADLOCK,
    DONE,
    EXPORT,
    OPS,
    PACKET,
    PACKET_BYTES,
    READY0,
    READY1,
    REGISTER_BYTES,
    SCHEDULE,
    TICKET,
    TICK,
    build,
    build_standalone,
    control_token,
    packet_tokens,
    program_tokens,
)
from qwengpu.eval_control import hex_bits, snapshot

FIXTURES = Path(__file__).resolve().parents[1] / "eval/frozen/tile_gate_v1"
TILED_PROGRAM = (
    ("MAC_K", 0),
    ("MAC_K", 1),
    ("BARRIER", 0),
    ("MAC_K", 2),
    ("MAC_K", 3),
    ("STORE_TILE", 0),
    ("BARRIER", 0),
    ("HALT", 0),
)


@dataclass(frozen=True)
class TileCase:
    programs: tuple[tuple[tuple[str, int], ...], ...]
    memory: list[int]
    initial: list[int]
    expected_outcome: str
    matrix_product: list[list[int]] | None


def _unsigned(values: object, length: int, limit: int, name: str) -> list[int]:
    if (
        not isinstance(values, list)
        or len(values) != length
        or any(type(value) is not int or not 0 <= value <= limit for value in values)
    ):
        raise ValueError(f"{name} must contain {length} integers from 0 to {limit}")
    return values


def parse_case(data: object) -> TileCase:
    if not isinstance(data, dict):
        raise ValueError("tile case must be an object")
    a = _unsigned(data.get("a", [0] * 16), 16, 255, "a")
    b = _unsigned(data.get("b", [0] * 16), 16, 255, "b")
    initial = _unsigned(data.get("initial", [0] * 16), 16, 65535, "initial")
    outcome = data.get("expected_outcome", "halted")
    if outcome not in ("halted", "deadlock", "step_limit"):
        raise ValueError("invalid expected_outcome")
    raw = data.get("programs")
    matrix_product = None
    if raw is None:
        programs = (TILED_PROGRAM, (("ADD16I", 0),) + TILED_PROGRAM)
        matrix_product = [
            [
                (
                    initial[4 * row + col]
                    + sum(a[4 * row + k] * b[4 * k + col] for k in range(4))
                )
                & 65535
                for col in range(4)
            ]
            for row in range(4)
        ]
    else:
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError("programs must contain two tile kernels")
        decoded = []
        for program in raw:
            if not isinstance(program, list) or not 1 <= len(program) <= 16:
                raise ValueError(
                    "each tile program must contain 1 through 16 instructions"
                )
            instructions = []
            for index, ins in enumerate(program):
                if not isinstance(ins, dict) or ins.get("op") not in OPS:
                    raise ValueError("invalid tile opcode")
                op, arg = ins["op"], ins.get("arg", 0)
                limit = {"MAC_K": 3, "ADD16I": 255}.get(op, 0)
                if type(arg) is not int or not 0 <= arg <= limit:
                    raise ValueError(f"invalid {op} argument")
                if (op == "HALT") != (index == len(program) - 1):
                    raise ValueError("HALT must appear exactly once at the end")
                instructions.append((op, arg))
            decoded.append(tuple(instructions))
        programs = tuple(decoded)
    return TileCase(programs, a + b + [0] * 32, initial, outcome, matrix_product)


def _word_bytes(words: list[int]) -> list[int]:
    return [byte for word in words for byte in (word & 255, word >> 8)]


def _decode_matrix(data: list[int]) -> list[list[int]]:
    return [
        [
            data[2 * (row * 4 + col)] | data[2 * (row * 4 + col) + 1] << 8
            for col in range(4)
        ]
        for row in range(4)
    ]


def _check_control(token: int, warp: int) -> None:
    if not CONTROL + 64 * warp <= token < CONTROL + 64 * warp + 48:
        raise ValueError(f"invalid control packet for warp {warp}: {token}")


@torch.no_grad()
def execute(
    model, circuit, case: TileCase, device: str, max_steps: int, offset: int = 0
) -> dict:
    controls = [control_token(warp) for warp in range(2)]
    previous = TICKET + 1
    initial_packets = packet_tokens(_word_bytes(case.initial[:8]) + case.memory)
    shared = initial_packets[REGISTER_BYTES:]
    accumulators = [
        initial_packets[:REGISTER_BYTES],
        packet_tokens(_word_bytes(case.initial[8:])),
    ]
    programs = [program_tokens(program) for program in case.programs]
    requests = list(range(EXPORT, EXPORT + PACKET_BYTES))
    schedules = []
    trace = []

    def feed(tokens: list[int], count: int):
        result = model(
            input_ids=torch.tensor(
                [[token + offset for token in tokens]], device=device
            ),
            use_cache=True,
            logits_to_keep=count,
        )
        return [
            token - offset for token in result.logits[0].argmax(dim=-1).tolist()
        ], result.past_key_values

    outcome = "step_limit"
    for index in range(max_steps + 1):
        emitted, _ = feed([*controls, previous, SCHEDULE, READY0, READY1], 3)
        ticket, *controls = emitted
        for warp, token in enumerate(controls):
            _check_control(token, warp)
        schedules.append({"ticket": ticket, "controls": controls})
        if ticket in (DONE, DEADLOCK):
            outcome = "halted" if ticket == DONE else "deadlock"
            break
        if not TICKET <= ticket < DONE:
            raise ValueError(f"invalid model scheduling ticket {ticket}")
        if index == max_steps:
            break
        warp = ticket & 1
        previous = ticket
        emitted, cache = feed(
            [
                *programs[warp],
                *accumulators[warp],
                *shared,
                controls[warp],
                TICK,
                *requests,
            ],
            1 + PACKET_BYTES,
        )
        control, *packet = emitted
        _check_control(control, warp)
        for address, token in enumerate(packet):
            if not PACKET + 256 * address <= token < PACKET + 256 * (address + 1):
                raise ValueError(f"invalid model data packet at byte {address}")
        controls = controls[:]
        controls[warp] = control
        accumulators[warp] = packet[:REGISTER_BYTES]
        shared = packet[REGISTER_BYTES:]
        trace.append(
            {
                "warp": warp,
                "control": control,
                "packet": packet,
                "snapshot": hex_bits(snapshot(cache, circuit, 0)),
            }
        )
    matrix = _decode_matrix([token & 255 for token in shared[32:]])
    return {
        "schedules": schedules,
        "trace": trace,
        "outcome": outcome,
        "matrix": matrix,
    }


def oracle(case: TileCase, max_steps: int) -> dict:
    pcs = [0, 0]
    states = [0, 0]
    last = 1
    accumulators = [case.initial[:8], case.initial[8:]]
    shared = case.memory[:]
    schedules = []
    trace = []
    outcome = "step_limit"
    for index in range(max_steps + 1):
        released = states == [2, 2]
        if released:
            states = [0, 0]
        runnable = [warp for warp, status in enumerate(states) if status == 0]
        if not runnable:
            outcome = "halted" if states == [1, 1] else "deadlock"
            ticket = DONE if outcome == "halted" else DEADLOCK
        else:
            warp = 1 - last if 1 - last in runnable else runnable[0]
            ticket = TICKET + warp + 2 * int(released)
        schedules.append(
            {
                "ticket": ticket,
                "controls": [control_token(w, pcs[w], states[w]) for w in range(2)],
            }
        )
        if not runnable or index == max_steps:
            break
        last = warp
        op, arg = case.programs[warp][pcs[warp]]
        acc = accumulators[warp]
        if op == "HALT":
            states[warp] = 1
        else:
            pcs[warp] += 1
            if op == "MAC_K":
                for lane in range(8):
                    row, col = 2 * warp + lane // 4, lane % 4
                    acc[lane] = (
                        acc[lane] + shared[4 * row + arg] * shared[16 + 4 * arg + col]
                    ) & 65535
            elif op == "ADD16I":
                accumulators[warp] = acc = [(word + arg) & 65535 for word in acc]
            elif op == "STORE_TILE":
                shared[32 + 16 * warp : 48 + 16 * warp] = _word_bytes(acc)
            else:
                states[warp] = 2
        values = _word_bytes(acc) + shared
        bits = [(byte >> bit) & 1 for byte in values for bit in range(8)]
        trace.append(
            {
                "warp": warp,
                "control": control_token(warp, pcs[warp], states[warp]),
                "packet": packet_tokens(values),
                "snapshot": hex_bits(bits),
            }
        )
    return {
        "schedules": schedules,
        "trace": trace,
        "outcome": outcome,
        "matrix": _decode_matrix(shared[32:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path)
    parser.add_argument("--kernel", type=Path)
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    paths = [args.kernel] if args.kernel else sorted(FIXTURES.glob("*.json"))
    cases = []
    for path in paths:
        payload = path.read_bytes()
        cases.append(
            (
                path.name,
                hashlib.sha256(payload).hexdigest(),
                parse_case(json.loads(payload)),
            )
        )
    if not cases:
        raise ValueError("no tile manifests found")

    meta = None
    offset = 0
    if args.model:
        meta = json.loads((args.model / "qwengpu9b.json").read_text())
        if meta["layout"] != "qwengpu-tile9b-v1":
            raise ValueError("not a tile checkpoint")
        offset = meta["chip_token_offset"]
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16
        )
        cfg = model.config
        circuit = build(
            Target(
                cfg.linear_num_key_heads,
                cfg.linear_num_value_heads,
                cfg.intermediate_size,
            )
        )
    else:
        model, circuit = build_standalone()
    model.to(args.device).eval()
    report = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "circuit": circuit.stats(),
        "merge": meta,
        "cases": [],
    }
    for name, digest, case in cases:
        expected = oracle(case, args.max_steps)
        observed = execute(model, circuit, case, args.device, args.max_steps, offset)
        exact = (
            observed == expected
            and observed["outcome"] == case.expected_outcome
            and (
                case.matrix_product is None or observed["matrix"] == case.matrix_product
            )
        )
        result = {
            "fixture": name,
            "fixture_sha256": digest,
            "exact": exact,
            "instructions": len(observed["trace"]),
            **observed,
        }
        if not exact:
            result["expected"] = expected
            result["expected_matrix_product"] = case.matrix_product
        report["cases"].append(result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("fixture", "exact", "instructions", "outcome", "matrix")
                }
            ),
            flush=True,
        )
    report["all_exact"] = all(case["exact"] for case in report["cases"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
