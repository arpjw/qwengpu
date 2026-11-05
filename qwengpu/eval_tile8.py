"""Run 8x8 matmul through four model-selected output tiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from chipformer.q35.ir import Target

from qwengpu.compiled_tile import (
    READY0,
    READY1,
    SCHEDULE,
    TICK,
    TICKET,
    control_token,
)
from qwengpu.compiled_tile8 import (
    ADVANCE,
    C_MEMORY,
    DEADLOCK,
    DONE,
    INDEX_OPERAND,
    INDEX_OPS,
    INDEX_REGISTER,
    INDEX_WORDS,
    NO_WRITE,
    OPERAND,
    OPS,
    READ_INDEX,
    READ_REG,
    READ_STORE,
    REGISTER,
    SNAPSHOT_BITS,
    build,
    build_standalone,
    nibbles,
    program_tokens,
)
from qwengpu.eval_compiled_rom import head_row
from qwengpu.eval_control import hex_bits, snapshot
from qwengpu.eval_tile import _check_control, _unsigned

FIXTURES = Path(__file__).resolve().parents[1] / "eval/frozen/tile8_gate_v1"
RESIDENT_FIXTURES = FIXTURES.with_name("resident_gate_v1")
INDEXED_FIXTURES = FIXTURES.with_name("indexed_gate_v1")
PROGRAM = (
    ("LOAD_TILE", 0),
    ("MAC_K", 0),
    ("MAC_K", 1),
    ("BARRIER", 0),
    ("MAC_K", 2),
    ("MAC_K", 3),
    ("BARRIER", 0),
    ("MAC_K", 4),
    ("MAC_K", 5),
    ("BARRIER", 0),
    ("MAC_K", 6),
    ("MAC_K", 7),
    ("STORE_TILE", 0),
    ("HALT", 0),
)


def _parse_case(data: object, *, indexed: bool = False) -> dict:
    if not isinstance(data, dict):
        raise ValueError("8x8 case must be an object")
    a = _unsigned(data.get("a", [0] * 64), 64, 255, "a")
    b = _unsigned(data.get("b", [0] * 64), 64, 255, "b")
    initial = _unsigned(data.get("c", [0] * 64), 64, 65535, "c")
    outcome = data.get("expected_outcome", "halted")
    if outcome not in ("halted", "deadlock", "step_limit"):
        raise ValueError("invalid expected_outcome")
    raw = data.get("programs")
    if raw is None:
        programs = [PROGRAM, (("ADD16I", 0),) + PROGRAM]
        matrix = [
            [
                (
                    initial[row * 8 + col]
                    + sum(a[row * 8 + k] * b[k * 8 + col] for k in range(8))
                )
                & 65535
                for col in range(8)
            ]
            for row in range(8)
        ]
    else:
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError("programs must contain two kernels")
        programs = []
        names = OPS + INDEX_OPS if indexed else OPS
        for program in raw:
            if not isinstance(program, list) or not 1 <= len(program) <= 16:
                raise ValueError(
                    "each tile program must contain 1 through 16 instructions"
                )
            ops = []
            for index, ins in enumerate(program):
                if not isinstance(ins, dict) or ins.get("op") not in names:
                    raise ValueError("invalid tile opcode")
                op, arg = ins["op"], ins.get("arg", 0)
                limit = {"MAC_K": 7, "ADD16I": 255}.get(op, 0)
                if type(arg) is not int or not 0 <= arg <= limit:
                    raise ValueError(f"invalid {op} argument")
                if (op == "HALT") != (index == len(program) - 1):
                    raise ValueError("HALT must appear exactly once at the end")
                ops.append((op, arg))
            programs.append(tuple(ops))
        matrix = None
    return {
        "a": a,
        "b": b,
        "c": initial,
        "programs": programs,
        "expected_outcome": outcome,
        "expected_matrix": matrix,
    }


def _digits(values: list[int], width: int) -> list[int]:
    return [(value >> shift) & 15 for value in values for shift in range(0, width, 4)]


def _words(tokens: list[int]) -> list[int]:
    return [
        sum((tokens[i + j] & 15) << (4 * j) for j in range(4))
        for i in range(0, len(tokens), 4)
    ]


def _matrix(tokens: list[int]) -> list[list[int]]:
    words = _words(tokens)
    return [words[index : index + 8] for index in range(0, 64, 8)]


@torch.no_grad()
def execute(
    model,
    circuit,
    case: dict,
    device: str,
    max_steps: int,
    offset: int = 0,
    *,
    resident: bool = False,
    indexed: bool = False,
) -> dict:
    resident = resident or indexed
    operand_base = INDEX_OPERAND if indexed else OPERAND
    controls = [control_token(warp) for warp in range(2)]
    previous = TICKET + 1
    operands = nibbles(operand_base, _digits(case["a"] + case["b"], 8))
    memory = nibbles(C_MEMORY, _digits(case["c"], 16))
    registers = [nibbles(REGISTER, [0] * 32) for _ in range(2)]
    rom = [program_tokens(program) for program in case["programs"]]
    indices = [nibbles(INDEX_REGISTER, [0] * 16) for _ in range(2)]
    reads = list(range(READ_REG, READ_REG + 32))
    if indexed:
        reads.extend(range(READ_INDEX, READ_INDEX + 16))
    reads.extend(range(READ_STORE, READ_STORE + 32))
    schedules = []
    trace = []
    cache = None
    pending = [*controls, previous, *operands, *memory] if resident else []
    transport = {
        "input_tokens": 0,
        "fresh_caches": 0,
        "matrix_initial_tokens": 0,
        "matrix_patch_tokens": 0,
    }

    def feed(tokens: list[int], count: int):
        nonlocal cache
        if not resident:
            cache = None
        transport["input_tokens"] += len(tokens)
        transport["fresh_caches"] += int(cache is None)
        out = model(
            input_ids=torch.tensor(
                [[token + offset for token in tokens]], device=device
            ),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=count,
        )
        cache = out.past_key_values
        return [token - offset for token in out.logits[0].argmax(dim=-1).tolist()]

    if resident:
        transport["matrix_initial_tokens"] = len(operands) + len(memory)
    outcome = "step_limit"
    for index in range(max_steps + 1):
        prefix = pending if resident else [*controls, previous]
        emitted = feed([*prefix, SCHEDULE, READY0, READY1], 3)
        ticket, *controls = emitted
        for warp, control in enumerate(controls):
            _check_control(control, warp)
        schedules.append({"ticket": ticket, "controls": controls})
        if ticket in (DONE, DEADLOCK):
            outcome = "halted" if ticket == DONE else "deadlock"
            break
        tag = ticket - TICKET
        if not 0 <= tag < 64 or tag & 4:
            raise ValueError(f"invalid tile scheduling ticket {ticket}")
        if index == max_steps:
            break
        warp = ticket & 1
        previous = ticket
        if resident:
            prefix = controls
        else:
            prefix = [*operands, *memory, controls[warp]]
            transport["matrix_initial_tokens"] += len(operands) + len(memory)
        state = registers[warp] + (indices[warp] if indexed else [])
        emitted = feed(
            [*rom[warp], *prefix, *state, ticket, TICK, *reads], 1 + len(reads)
        )
        control, *packet = emitted
        _check_control(control, warp)
        next_registers, patch = packet[:32], packet[-32:]
        for nib, token in enumerate(next_registers):
            if not REGISTER + 16 * nib <= token < REGISTER + 16 * (nib + 1):
                raise ValueError(f"invalid register packet at nibble {nib}")
        if indexed:
            indices[warp] = packet[32:-32]
            for nib, token in enumerate(indices[warp]):
                start = INDEX_REGISTER + 16 * nib
                if not start <= token < start + (8 if nib % 2 else 16):
                    raise ValueError(f"invalid index packet at nibble {nib}")
        updates = [token for token in patch if token != NO_WRITE]
        if not indexed and updates and len(updates) != 32:
            raise ValueError("incomplete tagged C write")
        if len({token // 16 for token in updates}) != len(updates):
            raise ValueError("repeated tagged write address")
        for token in updates:
            if C_MEMORY <= token < C_MEMORY + 4096:
                memory[(token - C_MEMORY) // 16] = token
            elif indexed and INDEX_OPERAND <= token < INDEX_OPERAND + 4096:
                operands[(token - INDEX_OPERAND) // 16] = token
            else:
                raise ValueError("invalid tagged shared-memory write")
        if resident:
            pending = [control, *updates]
            transport["matrix_patch_tokens"] += len(updates)
        controls = controls[:]
        controls[warp] = control
        registers[warp] = next_registers
        trace.append(
            {
                "warp": warp,
                "control": control,
                "registers": next_registers,
                **({"indices": indices[warp]} if indexed else {}),
                "patch": patch,
                "snapshot": hex_bits(snapshot(cache, circuit, 0)),
            }
        )
    if resident:
        transport["resident_memory_exact"] = _check_resident_memory(
            cache, circuit, operands, memory, indexed=indexed
        )
    return {
        "transport": transport,
        "schedules": schedules,
        "trace": trace,
        "outcome": outcome,
        "matrix": _matrix(memory),
        **({"shared_words": _words(operands + memory)} if indexed else {}),
    }


def _check_resident_memory(
    cache, circuit, operands: list[int], memory: list[int], *, indexed: bool = False
) -> bool:
    for address, token in enumerate(operands):
        byte, nibble = divmod(address, 2)
        if byte < 64:
            head, slot = f"A.{nibble}", byte
        else:
            row, col = divmod(byte - 64, 8)
            head, slot = f"B.{col}.{nibble}", row
        if head_row(cache, circuit, head, slot, 4) != token & 15:
            return False
    for address, token in enumerate(memory):
        row, nibble = divmod(address, 32)
        if head_row(cache, circuit, f"C.{nibble}", row, 4) != token & 15:
            return False
    if indexed:
        for address, token in enumerate(operands + memory):
            word, nib = divmod(address, 4)
            for lane in range(8):
                if (
                    head_row(cache, circuit, f"shared.{lane}.{nib}", word, 4)
                    != token & 15
                ):
                    return False
    return True


def oracle(case: dict, max_steps: int, *, indexed: bool = False) -> dict:
    pcs = [0, 0]
    states = [0, 0]
    tile = 0
    last = 1
    acc = [[0] * 8 for _ in range(2)]
    indices = [[0] * 8 for _ in range(2)]
    operands = bytearray(case["a"] + case["b"])
    c = case["c"][:]
    schedules = []
    trace = []
    outcome = "step_limit"
    for index in range(max_steps + 1):
        released = states == [2, 2]
        advance = states == [1, 1] and tile < 3
        if advance:
            tile += 1
            pcs = [0, 0]
        if advance or released:
            states = [0, 0]
        ready = [warp for warp in range(2) if states[warp] == 0]
        if not ready:
            outcome = "halted" if states == [1, 1] else "deadlock"
            ticket = DONE if outcome == "halted" else DEADLOCK
        else:
            warp = 1 - last if 1 - last in ready else ready[0]
            ticket = (
                TICKET + 8 * tile + warp + 2 * int(released) + ADVANCE * int(advance)
            )
        schedules.append(
            {
                "ticket": ticket,
                "controls": [control_token(w, pcs[w], states[w]) for w in range(2)],
            }
        )
        if not ready or index == max_steps:
            break
        last = warp
        op, arg = case["programs"][warp][pcs[warp]]
        row = 2 * tile + warp
        start = row * 8
        if op == "HALT":
            states[warp] = 1
        else:
            pcs[warp] += 1
        if op == "LOAD_TILE":
            acc[warp] = c[start : start + 8]
        elif op == "MAC_K":
            acc[warp] = [
                (v + operands[start + arg] * operands[64 + 8 * arg + col]) & 65535
                for col, v in enumerate(acc[warp])
            ]
        elif op == "ADD16I":
            acc[warp] = [(v + arg) & 65535 for v in acc[warp]]
        elif op == "BARRIER":
            states[warp] = 2
        elif op == "SAVE_INDEX":
            indices[warp] = [v % INDEX_WORDS for v in acc[warp]]
        elif op == "LOAD_INDEX":
            acc[warp] = [
                int.from_bytes(operands[2 * addr : 2 * addr + 2], "little")
                if addr < 64
                else c[addr - 64]
                for addr in indices[warp]
            ]
        patch = [NO_WRITE] * 32
        if op == "STORE_TILE":
            c[start : start + 8] = acc[warp]
            patch = nibbles(C_MEMORY + 512 * row, _digits(acc[warp], 16))
        grants = 0
        if op == "STORE_INDEX":
            claimed = set()
            for lane, (addr, word) in enumerate(
                zip(indices[warp], acc[warp], strict=True)
            ):
                if addr in claimed:
                    continue
                claimed.add(addr)
                grants |= 1 << lane
                if addr < 64:
                    operands[2 * addr : 2 * addr + 2] = word.to_bytes(2, "little")
                    base = INDEX_OPERAND + 64 * addr
                else:
                    c[addr - 64] = word
                    base = C_MEMORY + 64 * (addr - 64)
                patch[4 * lane : 4 * lane + 4] = nibbles(base, _digits([word], 16))
        packed = sum(word << (16 * lane) for lane, word in enumerate(acc[warp]))
        packed |= (int(op == "STORE_TILE") + 2 * row) << 128
        if indexed:
            packed |= sum(
                value << (SNAPSHOT_BITS + 7 * lane)
                for lane, value in enumerate(indices[warp])
            )
            packed |= grants << (SNAPSHOT_BITS + 56)
        trace.append(
            {
                "warp": warp,
                "control": control_token(warp, pcs[warp], states[warp]),
                "registers": nibbles(REGISTER, _digits(acc[warp], 16)),
                **(
                    {"indices": nibbles(INDEX_REGISTER, _digits(indices[warp], 8))}
                    if indexed
                    else {}
                ),
                "patch": patch,
                "snapshot": f"{packed:0{49 if indexed else 33}x}",
            }
        )
    return {
        "schedules": schedules,
        "trace": trace,
        "outcome": outcome,
        "matrix": [c[i : i + 8] for i in range(0, 64, 8)],
        **(
            {
                "shared_words": [
                    int.from_bytes(operands[i : i + 2], "little")
                    for i in range(0, len(operands), 2)
                ]
                + c,
            }
            if indexed
            else {}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path)
    parser.add_argument("--kernel", type=Path)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--resident", action="store_true")
    variants.add_argument("--indexed", action="store_true")
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    meta = None
    offset = 0
    indexed = args.indexed
    resident = args.resident or indexed
    if args.model:
        meta = json.loads((args.model / "qwengpu9b.json").read_text())
        layout = meta["layout"]
        if layout not in (
            "qwengpu-tile8-9b-v1",
            "qwengpu-resident9b-v1",
            "qwengpu-indexed9b-v1",
        ):
            raise ValueError("not an 8x8 tile checkpoint")
        indexed = layout == "qwengpu-indexed9b-v1"
        resident = indexed or layout == "qwengpu-resident9b-v1"
        if args.resident and not resident:
            raise ValueError("--resident requires a resident checkpoint")
        if args.indexed and not indexed:
            raise ValueError("--indexed requires an indexed checkpoint")
    paths = [args.kernel] if args.kernel else sorted(FIXTURES.glob("*.json"))
    if resident and not args.kernel:
        paths.extend(sorted(RESIDENT_FIXTURES.glob("*.json")))
    if indexed and not args.kernel:
        paths.extend(sorted(INDEXED_FIXTURES.glob("*.json")))
    cases = []
    for path in paths:
        payload = path.read_bytes()
        cases.append(
            (
                path.name,
                hashlib.sha256(payload).hexdigest(),
                _parse_case(json.loads(payload), indexed=indexed),
            )
        )
    if not cases:
        raise ValueError("no tile manifests found")

    if args.model:
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
            ),
            resident=resident,
            indexed=indexed,
        )
    else:
        model, circuit = build_standalone(resident=resident, indexed=indexed)
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
        expected = oracle(case, args.max_steps, indexed=indexed)
        observed = execute(
            model,
            circuit,
            case,
            args.device,
            args.max_steps,
            offset,
            resident=resident,
            indexed=indexed,
        )
        transport = observed.pop("transport")
        exact = (
            observed == expected
            and (
                not resident
                or (
                    transport["fresh_caches"] == 1
                    and transport["resident_memory_exact"]
                )
            )
            and observed["outcome"] == case["expected_outcome"]
            and (
                case["expected_matrix"] is None
                or observed["matrix"] == case["expected_matrix"]
            )
        )
        result = {
            "fixture": name,
            "fixture_sha256": digest,
            "exact": exact,
            "instructions": len(observed["trace"]),
            "transport": transport,
            **observed,
        }
        if not exact:
            result["expected"] = expected
            result["expected_matrix"] = case["expected_matrix"]
        report["cases"].append(result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("fixture", "exact", "instructions", "outcome")
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
