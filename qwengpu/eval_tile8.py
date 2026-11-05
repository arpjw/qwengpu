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
    BRANCH_NIBBLES,
    BRANCH_OFFSET,
    BRANCH_OPS,
    BRANCH_REGISTER,
    C_MEMORY,
    DEADLOCK,
    DONE,
    INDEX_OPERAND,
    INDEX_OPS,
    INDEX_REGISTER,
    INDEX_WORDS,
    LOOP_NIBBLES,
    LOOP_OFFSET,
    LOOP_OPS,
    LOOP_REGISTER,
    MASK_OFFSET,
    MASK_OPS,
    MASK_REGISTER,
    NO_WRITE,
    OPERAND,
    OPS,
    READ_BRANCH,
    READ_INDEX,
    READ_LOOP,
    READ_MASK,
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
MASKED_FIXTURES = FIXTURES.with_name("masked_gate_v1")
DIVERGENT_FIXTURES = FIXTURES.with_name("divergent_gate_v1")
LOOP_FIXTURES = FIXTURES.with_name("loop_gate_v1")
LOOP_LAYOUT = "qwengpu-looping9b-v1"
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


def _control_links(program: list[tuple[str, int]]) -> dict[int, int]:
    regions: list[tuple[str, int, bool]] = []
    links = {}
    for pc, (op, _) in enumerate(program):
        if op in ("IF_BIT", "WHILE_BIT"):
            if sum(kind == op for kind, _, _ in regions) == 2:
                raise ValueError(f"{op} nesting exceeds two levels")
            regions.append((op, pc, False))
        elif op == "ELSE":
            if not regions or regions[-1][0] != "IF_BIT" or regions[-1][2]:
                raise ValueError("ELSE requires an open IF_BIT without an ELSE")
            regions[-1] = ("IF_BIT", regions[-1][1], True)
        elif op in ("ENDIF", "ENDWHILE"):
            opener = "IF_BIT" if op == "ENDIF" else "WHILE_BIT"
            if not regions or regions[-1][0] != opener:
                raise ValueError(f"{op} requires an open {opener}")
            _, start, _ = regions.pop()
            if op == "ENDWHILE":
                links[start], links[pc] = pc + 1, start
        elif regions and op in (*MASK_OPS, "BARRIER", "HALT"):
            raise ValueError(f"{op} requires a reconverged warp")
    if regions:
        raise ValueError(f"unterminated {regions[-1][0]}")
    return links


def parse_launch(
    data: object,
    *,
    indexed: bool = False,
    masked: bool = False,
    divergent: bool = False,
    looping: bool = False,
) -> dict:
    divergent = divergent or looping
    masked = masked or divergent
    if not isinstance(data, dict):
        raise ValueError("8x8 case must be an object")
    a = _unsigned(data.get("a", [0] * 64), 64, 255, "a")
    b = _unsigned(data.get("b", [0] * 64), 64, 255, "b")
    initial = _unsigned(data.get("c", [0] * 64), 64, 65535, "c")
    raw = data.get("programs")
    links = [{}, {}]
    if raw is None:
        programs = [PROGRAM, (("ADD16I", 0),) + PROGRAM]
    else:
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError("programs must contain two kernels")
        programs = []
        names = (
            OPS
            + (INDEX_OPS if indexed or masked else ())
            + (MASK_OPS if masked else ())
            + (BRANCH_OPS if divergent else ())
            + (LOOP_OPS if looping else ())
        )
        for warp, program in enumerate(raw):
            if not isinstance(program, list) or not 1 <= len(program) <= 16:
                raise ValueError(
                    "each tile program must contain 1 through 16 instructions"
                )
            ops = []
            for index, ins in enumerate(program):
                if not isinstance(ins, dict) or ins.get("op") not in names:
                    raise ValueError("invalid tile opcode")
                op, arg = ins["op"], ins.get("arg", 0)
                limit = {
                    "MAC_K": 7,
                    "ADD16I": 255,
                    "MASK_BIT": 15,
                    "IF_BIT": 15,
                    "WHILE_BIT": 15,
                }.get(op, 0)
                if type(arg) is not int or not 0 <= arg <= limit:
                    raise ValueError(f"invalid {op} argument")
                if (op == "HALT") != (index == len(program) - 1):
                    raise ValueError("HALT must appear exactly once at the end")
                ops.append((op, arg))
            if divergent:
                links[warp] = _control_links(ops)
            programs.append(tuple(ops))
    return {"a": a, "b": b, "c": initial, "programs": programs, "links": links}


def _parse_case(
    data: object,
    *,
    indexed: bool = False,
    masked: bool = False,
    divergent: bool = False,
    looping: bool = False,
) -> dict:
    case = parse_launch(
        data, indexed=indexed, masked=masked, divergent=divergent, looping=looping
    )
    outcome = data.get("expected_outcome", "halted")
    if outcome not in ("halted", "deadlock", "step_limit"):
        raise ValueError("invalid expected_outcome")
    matrix = None
    if data.get("programs") is None:
        a, b, initial = case["a"], case["b"], case["c"]
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
    if "expected_matrix" in data:
        expected_matrix = data["expected_matrix"]
        if not isinstance(expected_matrix, list) or len(expected_matrix) != 8:
            raise ValueError("expected_matrix must contain eight rows")
        supplied = [
            _unsigned(row, 8, 65535, "expected_matrix row") for row in expected_matrix
        ]
        if matrix is not None and supplied != matrix:
            raise ValueError(
                "expected_matrix disagrees with the default matrix product"
            )
        matrix = supplied
    return {**case, "expected_outcome": outcome, "expected_matrix": matrix}


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
    masked: bool = False,
    divergent: bool = False,
    looping: bool = False,
) -> dict:
    divergent = divergent or looping
    masked = masked or divergent
    indexed = indexed or masked
    resident = resident or indexed
    operand_base = INDEX_OPERAND if indexed else OPERAND
    controls = [control_token(warp) for warp in range(2)]
    previous = TICKET + 1
    operands = nibbles(operand_base, _digits(case["a"] + case["b"], 8))
    memory = nibbles(C_MEMORY, _digits(case["c"], 16))
    registers = [nibbles(REGISTER, [0] * 32) for _ in range(2)]
    rom = [program_tokens(program) for program in case["programs"]]
    indices = [nibbles(INDEX_REGISTER, [0] * 16) for _ in range(2)]
    masks = [nibbles(MASK_REGISTER, [15, 15]) for _ in range(2)]
    branches = [nibbles(BRANCH_REGISTER, [0] * BRANCH_NIBBLES) for _ in range(2)]
    loops = [nibbles(LOOP_REGISTER, [0] * LOOP_NIBBLES) for _ in range(2)]
    reads = list(range(READ_REG, READ_REG + 32))
    if indexed:
        reads.extend(range(READ_INDEX, READ_INDEX + 16))
    if masked:
        reads.extend(range(READ_MASK, READ_MASK + 2))
    if divergent:
        reads.extend(range(READ_BRANCH, READ_BRANCH + BRANCH_NIBBLES))
    if looping:
        reads.extend(range(READ_LOOP, READ_LOOP + LOOP_NIBBLES))
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
        if masked:
            state += masks[warp]
        if divergent:
            state += branches[warp]
        if looping:
            state += loops[warp]
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
            indices[warp] = packet[32:48]
            for nib, token in enumerate(indices[warp]):
                start = INDEX_REGISTER + 16 * nib
                if not start <= token < start + (8 if nib % 2 else 16):
                    raise ValueError(f"invalid index packet at nibble {nib}")
        if masked:
            masks[warp] = packet[48:50]
            for nib, token in enumerate(masks[warp]):
                start = MASK_REGISTER + 16 * nib
                if not start <= token < start + 16:
                    raise ValueError(f"invalid mask packet at nibble {nib}")
        if divergent:
            branches[warp] = packet[50 : 50 + BRANCH_NIBBLES]
            for nib, token in enumerate(branches[warp]):
                start = BRANCH_REGISTER + 16 * nib
                if not start <= token < start + (3 if nib == 0 else 16):
                    raise ValueError(f"invalid branch packet at nibble {nib}")
        if looping:
            loops[warp] = packet[50 + BRANCH_NIBBLES : -32]
            for nib, token in enumerate(loops[warp]):
                start = LOOP_REGISTER + 16 * nib
                if not start <= token < start + (3 if nib == 0 else 16):
                    raise ValueError(f"invalid loop packet at nibble {nib}")
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
                **({"mask": masks[warp]} if masked else {}),
                **({"branch": branches[warp]} if divergent else {}),
                **({"loop": loops[warp]} if looping else {}),
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


def oracle(
    case: dict,
    max_steps: int,
    *,
    indexed: bool = False,
    masked: bool = False,
    divergent: bool = False,
    looping: bool = False,
) -> dict:
    divergent = divergent or looping
    masked = masked or divergent
    indexed = indexed or masked
    pcs = [0, 0]
    states = [0, 0]
    tile = 0
    last = 1
    acc = [[0] * 8 for _ in range(2)]
    indices = [[0] * 8 for _ in range(2)]
    masks = [255, 255]
    branches: list[list[tuple[int, int]]] = [[], []]
    loops: list[list[int]] = [[], []]
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
        pc = pcs[warp]
        op, arg = case["programs"][warp][pc]
        row = 2 * tile + warp
        start = row * 8
        if op == "HALT":
            states[warp] = 1
        else:
            pcs[warp] += 1
        before = acc[warp]
        candidate = before
        active = masks[warp]
        if op == "LOAD_TILE":
            candidate = c[start : start + 8]
        elif op == "MAC_K":
            candidate = [
                (v + operands[start + arg] * operands[64 + 8 * arg + col]) & 65535
                for col, v in enumerate(acc[warp])
            ]
        elif op == "ADD16I":
            candidate = [(v + arg) & 65535 for v in acc[warp]]
        elif op == "BARRIER":
            states[warp] = 2
        elif op == "SAVE_INDEX":
            indices[warp] = [
                v % INDEX_WORDS if active >> lane & 1 else indices[warp][lane]
                for lane, v in enumerate(acc[warp])
            ]
        elif op == "LOAD_INDEX":
            candidate = [
                int.from_bytes(operands[2 * addr : 2 * addr + 2], "little")
                if addr < 64
                else c[addr - 64]
                for addr in indices[warp]
            ]
        elif op == "MASK_ALL":
            masks[warp] = 255
        elif op == "MASK_BIT":
            masks[warp] = sum(((v >> arg) & 1) << lane for lane, v in enumerate(before))
        elif op == "IF_BIT":
            selected = sum(((v >> arg) & 1) << lane for lane, v in enumerate(before))
            masks[warp] = active & selected
            branches[warp].append((active, masks[warp]))
        elif op == "ELSE":
            parent, then_mask = branches[warp][-1]
            masks[warp] = parent & ~then_mask
        elif op == "ENDIF":
            parent, _ = branches[warp].pop()
            masks[warp] = parent
        elif op == "WHILE_BIT":
            condition = sum(((v >> arg) & 1) << lane for lane, v in enumerate(before))
            if active & condition:
                loops[warp].append(active)
                masks[warp] = active & condition
            else:
                pcs[warp] = case["links"][warp][pc]
        elif op == "ENDWHILE":
            masks[warp] = loops[warp].pop()
            pcs[warp] = case["links"][warp][pc]
        acc[warp] = [
            value if active >> lane & 1 else before[lane]
            for lane, value in enumerate(candidate)
        ]
        patch = [NO_WRITE] * 32
        if op == "STORE_TILE":
            for lane, value in enumerate(acc[warp]):
                if active >> lane & 1:
                    c[start + lane] = value
                    base = C_MEMORY + 512 * row + 64 * lane
                    patch[4 * lane : 4 * lane + 4] = nibbles(base, _digits([value], 16))
        grants = 0
        if op == "STORE_INDEX":
            claimed = set()
            for lane, (addr, word) in enumerate(
                zip(indices[warp], acc[warp], strict=True)
            ):
                if not active >> lane & 1 or addr in claimed:
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
        if masked:
            packed |= masks[warp] << MASK_OFFSET
        width = 51 if masked else 49 if indexed else 33
        branch_tokens = []
        if divergent:
            depth = len(branches[warp])
            frame_bytes = [value for frame in branches[warp] for value in frame]
            frame_bytes.extend([0] * (4 - len(frame_bytes)))
            branch_tokens = nibbles(BRANCH_REGISTER, [depth, *_digits(frame_bytes, 8)])
            branch_word = sum(
                (token & 15) << (4 * i) for i, token in enumerate(branch_tokens)
            )
            packed |= branch_word << BRANCH_OFFSET
            width = (BRANCH_OFFSET + 4 * BRANCH_NIBBLES) // 4
        loop_tokens = []
        if looping:
            frames = loops[warp] + [0] * (2 - len(loops[warp]))
            loop_tokens = nibbles(
                LOOP_REGISTER, [len(loops[warp]), *_digits(frames, 8)]
            )
            loop_word = sum(
                (token & 15) << (4 * i) for i, token in enumerate(loop_tokens)
            )
            packed |= loop_word << LOOP_OFFSET
            width = (LOOP_OFFSET + 4 * LOOP_NIBBLES) // 4
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
                **(
                    {"mask": nibbles(MASK_REGISTER, _digits([masks[warp]], 8))}
                    if masked
                    else {}
                ),
                **({"branch": branch_tokens} if divergent else {}),
                **({"loop": loop_tokens} if looping else {}),
                "snapshot": f"{packed:0{width}x}",
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


def load_tile_model(
    checkpoint: Path | None,
    device: str,
    *,
    resident: bool = False,
    indexed: bool = False,
    masked: bool = False,
    divergent: bool = False,
    looping: bool = False,
):
    if checkpoint is None:
        model, circuit = build_standalone(
            resident=resident,
            indexed=indexed,
            masked=masked,
            divergent=divergent,
            looping=looping,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype=torch.bfloat16
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
            masked=masked,
            divergent=divergent,
            looping=looping,
        )
    return model.to(device).eval(), circuit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path)
    parser.add_argument("--kernel", type=Path)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--resident", action="store_true")
    variants.add_argument("--indexed", action="store_true")
    variants.add_argument("--masked", action="store_true")
    variants.add_argument("--divergent", action="store_true")
    variants.add_argument("--looping", action="store_true")
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    meta = None
    looping = args.looping
    divergent = args.divergent or looping
    masked = args.masked or divergent
    indexed = args.indexed or masked
    resident = args.resident or indexed
    if args.model:
        meta = json.loads((args.model / "qwengpu9b.json").read_text())
        layout = meta["layout"]
        if layout not in (
            "qwengpu-tile8-9b-v1",
            "qwengpu-resident9b-v1",
            "qwengpu-indexed9b-v1",
            "qwengpu-masked9b-v1",
            "qwengpu-divergent9b-v1",
            LOOP_LAYOUT,
        ):
            raise ValueError("not an 8x8 tile checkpoint")
        looping = layout == LOOP_LAYOUT
        divergent = looping or layout == "qwengpu-divergent9b-v1"
        masked = divergent or layout == "qwengpu-masked9b-v1"
        indexed = masked or layout == "qwengpu-indexed9b-v1"
        resident = indexed or layout == "qwengpu-resident9b-v1"
        if args.resident and not resident:
            raise ValueError("--resident requires a resident checkpoint")
        if args.indexed and not indexed:
            raise ValueError("--indexed requires an indexed checkpoint")
        if args.masked and not masked:
            raise ValueError("--masked requires a masked checkpoint")
        if args.divergent and not divergent:
            raise ValueError("--divergent requires a divergent checkpoint")
        if args.looping and not looping:
            raise ValueError("--looping requires a looping checkpoint")
    paths = [args.kernel] if args.kernel else sorted(FIXTURES.glob("*.json"))
    if resident and not args.kernel:
        paths.extend(sorted(RESIDENT_FIXTURES.glob("*.json")))
    if indexed and not args.kernel:
        paths.extend(sorted(INDEXED_FIXTURES.glob("*.json")))
    if masked and not args.kernel:
        paths.extend(sorted(MASKED_FIXTURES.glob("*.json")))
    if divergent and not args.kernel:
        paths.extend(sorted(DIVERGENT_FIXTURES.glob("*.json")))
    if looping and not args.kernel:
        paths.extend(sorted(LOOP_FIXTURES.glob("*.json")))
    cases = []
    for path in paths:
        payload = path.read_bytes()
        cases.append(
            (
                path.name,
                hashlib.sha256(payload).hexdigest(),
                _parse_case(
                    json.loads(payload),
                    indexed=indexed,
                    masked=masked,
                    divergent=divergent,
                    looping=looping,
                ),
            )
        )
    if not cases:
        raise ValueError("no tile manifests found")

    offset = meta["chip_token_offset"] if meta is not None else 0
    model, circuit = load_tile_model(
        args.model,
        args.device,
        resident=resident,
        indexed=indexed,
        masked=masked,
        divergent=divergent,
        looping=looping,
    )
    report = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "circuit": circuit.stats(),
        "merge": meta,
        "cases": [],
    }
    for name, digest, case in cases:
        expected = oracle(
            case,
            args.max_steps,
            indexed=indexed,
            masked=masked,
            divergent=divergent,
            looping=looping,
        )
        observed = execute(
            model,
            circuit,
            case,
            args.device,
            args.max_steps,
            offset,
            resident=resident,
            indexed=indexed,
            masked=masked,
            divergent=divergent,
            looping=looping,
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
