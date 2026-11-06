"""Run 8x8 matmul through four model-selected output tiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from chipformer.q35.ir import Target
from transformers.cache_utils import Cache

from qwengpu.compiled_tile import (
    READY0,
    READY1,
    SCHEDULE,
    TICK,
    TICKET,
    control_token,
)
from qwengpu.compiled_tile8 import (
    ADD32I_OP,
    ADD_INDEX_HI,
    ADD_INDEX_LO,
    ADDMEM32_OPS,
    ADVANCE,
    BRANCH_NIBBLES,
    BRANCH_OFFSET,
    BRANCH_OPS,
    BRANCH_REGISTER,
    C_MEMORY,
    DEADLOCK,
    DONE,
    GRID,
    HIGH_NIBBLES,
    HIGH_REGISTER,
    INDEX_OPERAND,
    INDEX_OPS,
    INDEX_REGISTER,
    INDEX_WORDS,
    LOAD_INDEX_HI,
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
    PLAN_TAG,
    READ_BRANCH,
    READ_HIGH,
    READ_INDEX,
    READ_LOOP,
    READ_MASK,
    READ_PLAN,
    READ_REG,
    READ_STORE,
    REGISTER,
    RESIDENT_ADDMEM32_LAYOUT,
    RESIDENT_BRANCH32_LAYOUT,
    RESIDENT_LOOP32_LAYOUT,
    RESIDENT_SHARED32_LAYOUT,
    ROM_BANK,
    ROM_PAYLOAD_BITS,
    ROW_PAIRS,
    SHARED32_OPS,
    SNAPSHOT_BITS,
    STORE_INDEX_HI,
    WARP_BANKS,
    WARP_DIRTY_OFFSET,
    WARP_TAGS,
    WIDE_DIRTY_OFFSET,
    WIDE_WARP_TAGS,
    build,
    build_standalone,
    nibbles,
    program_tokens,
    rom_payload,
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
GRID_FIXTURES = FIXTURES.with_name("grid_gate_v1")
WIDE_FIXTURES = FIXTURES.with_name("wide_gate_v1")
ADD32I_FIXTURES = FIXTURES.with_name("resident_add32_gate_v1")
SHARED32_FIXTURES = FIXTURES.with_name("resident_shared32_gate_v1")
ADDMEM32_FIXTURES = FIXTURES.with_name("resident_addmem32_gate_v1")
BRANCH32_FIXTURES = FIXTURES.with_name("resident_branch32_gate_v1")
LOOP32_FIXTURES = FIXTURES.with_name("resident_loop32_gate_v1")
LOOP_LAYOUT = "qwengpu-looping9b-v1"
RESIDENT_ROM_LAYOUT = "qwengpu-resident-rom9b-v1"
RESIDENT_STATE_LAYOUT = "qwengpu-resident-state9b-v1"
RESIDENT_DELTA_LAYOUT = "qwengpu-resident-delta9b-v1"
RESIDENT_PLAN_LAYOUT = "qwengpu-resident-plan9b-v1"
RESIDENT_IO_LAYOUT = "qwengpu-resident-io9b-v1"
RESIDENT_GRID_LAYOUT = "qwengpu-resident-grid9b-v1"
RESIDENT_WIDE_LAYOUT = "qwengpu-resident-wide9b-v1"
RESIDENT_ADD32_LAYOUT = "qwengpu-resident-add32-9b-v1"
RESIDENT_BRANCH32_LAYOUTS = (RESIDENT_BRANCH32_LAYOUT, RESIDENT_LOOP32_LAYOUT)
RESIDENT_ADDMEM32_LAYOUTS = (RESIDENT_ADDMEM32_LAYOUT, *RESIDENT_BRANCH32_LAYOUTS)
RESIDENT_SHARED32_LAYOUTS = (RESIDENT_SHARED32_LAYOUT, *RESIDENT_ADDMEM32_LAYOUTS)
RESIDENT_ADD32_LAYOUTS = (RESIDENT_ADD32_LAYOUT, *RESIDENT_SHARED32_LAYOUTS)
RESIDENT_TRANSPORTS = {
    "fixed": {"resident_delta": True},
    "state_plan": {"resident_plan": True},
    "state_store_plan": {"resident_io": True},
}
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


def _row_window(data: dict, *, resident_grid: bool) -> tuple[int, int]:
    first = data.get("first_row_pair", 0)
    if type(first) is not int or not 0 <= first < ROW_PAIRS:
        raise ValueError(
            f"first_row_pair must be an integer from 0 through {ROW_PAIRS - 1}"
        )
    remaining = ROW_PAIRS - first
    count = data.get("row_pairs", remaining)
    if type(count) is not int or not 1 <= count <= remaining:
        raise ValueError(f"row_pairs must be an integer from 1 through {remaining}")
    if count != ROW_PAIRS and not resident_grid:
        raise ValueError("partial row_pairs requires a resident grid checkpoint")
    return first, count


def parse_launch(
    data: object,
    *,
    indexed: bool = False,
    masked: bool = False,
    divergent: bool = False,
    looping: bool = False,
    resident_grid: bool = False,
    resident_wide: bool = False,
    resident_add32: bool = False,
    resident_shared32: bool = False,
    resident_addmem32: bool = False,
    resident_branch32: bool = False,
    resident_loop32: bool = False,
) -> dict:
    resident_branch32 = resident_branch32 or resident_loop32
    resident_addmem32 = resident_addmem32 or resident_branch32
    resident_shared32 = resident_shared32 or resident_addmem32
    resident_add32 = resident_add32 or resident_shared32
    looping = looping or resident_shared32
    divergent = divergent or looping
    masked = masked or divergent
    if not isinstance(data, dict):
        raise ValueError("8x8 case must be an object")
    a = _unsigned(data.get("a", [0] * 64), 64, 255, "a")
    b = _unsigned(data.get("b", [0] * 64), 64, 255, "b")
    initial = _unsigned(data.get("c", [0] * 64), 64, 65535, "c")
    resident_wide = resident_wide or resident_add32
    first_row_pair, row_pairs = _row_window(
        data, resident_grid=resident_grid or resident_wide
    )
    initial_registers = None
    if resident_wide:
        banks = data.get("initial_registers", [[0] * 8, [0] * 8])
        if not isinstance(banks, list) or len(banks) != 2:
            raise ValueError("initial_registers must contain two warp banks")
        initial_registers = [
            _unsigned(bank, 8, 0xFFFFFFFF, "initial_registers bank") for bank in banks
        ]
    elif "initial_registers" in data:
        raise ValueError("initial_registers requires a resident wide checkpoint")
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
            + ((ADD32I_OP,) if resident_add32 else ())
            + (SHARED32_OPS if resident_shared32 else ())
            + (ADDMEM32_OPS if resident_addmem32 else ())
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
                    ADD32I_OP: 255,
                    ADD_INDEX_LO: 126,
                    ADD_INDEX_HI: 126,
                    "MASK_BIT": 15,
                    "IF_BIT": 31 if resident_branch32 else 15,
                    "WHILE_BIT": 31 if resident_loop32 else 15,
                }.get(op, 0)
                if type(arg) is not int or not 0 <= arg <= limit:
                    raise ValueError(f"invalid {op} argument")
                if op in ADDMEM32_OPS and arg % 2:
                    raise ValueError(f"{op} requires an even pair mask")
                if (op == "HALT") != (index == len(program) - 1):
                    raise ValueError("HALT must appear exactly once at the end")
                ops.append((op, arg))
            if divergent:
                links[warp] = _control_links(ops)
            programs.append(tuple(ops))
    return {
        "a": a,
        "b": b,
        "c": initial,
        "programs": programs,
        "links": links,
        "first_row_pair": first_row_pair,
        "row_pairs": row_pairs,
        **({"initial_registers": initial_registers} if resident_wide else {}),
    }


def _parse_case(
    data: object,
    *,
    indexed: bool = False,
    masked: bool = False,
    divergent: bool = False,
    looping: bool = False,
    resident_grid: bool = False,
    resident_wide: bool = False,
    resident_add32: bool = False,
    resident_shared32: bool = False,
    resident_addmem32: bool = False,
    resident_branch32: bool = False,
    resident_loop32: bool = False,
) -> dict:
    case = parse_launch(
        data,
        indexed=indexed,
        masked=masked,
        divergent=divergent,
        looping=looping,
        resident_grid=resident_grid,
        resident_wide=resident_wide,
        resident_add32=resident_add32,
        resident_shared32=resident_shared32,
        resident_addmem32=resident_addmem32,
        resident_branch32=resident_branch32,
        resident_loop32=resident_loop32,
    )
    outcome = data.get("expected_outcome", "halted")
    if outcome not in ("halted", "deadlock", "step_limit"):
        raise ValueError("invalid expected_outcome")
    matrix = None
    if data.get("programs") is None:
        a, b, initial = case["a"], case["b"], case["c"]
        first = case["first_row_pair"]
        last = first + case["row_pairs"]
        matrix = [
            [
                (
                    initial[row * 8 + col]
                    + (
                        sum(a[row * 8 + k] * b[k * 8 + col] for k in range(8))
                        if 2 * first <= row < 2 * last
                        else 0
                    )
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
    resident_rom: bool = False,
    resident_state: bool = False,
    resident_delta: bool = False,
    resident_plan: bool = False,
    resident_io: bool = False,
    resident_grid: bool = False,
    resident_wide: bool = False,
    resident_add32: bool = False,
    resident_cache: Cache | None = None,
    refresh_operands: bool = False,
) -> dict:
    """Run a kernel with optional retained memory and fresh A and B operands."""
    resident_shared32 = circuit.resident_shared32
    resident_addmem32 = circuit.resident_addmem32
    resident_branch32 = circuit.resident_branch32
    resident_loop32 = circuit.resident_loop32
    resident_add32 = resident_add32 or resident_shared32
    if refresh_operands and resident_cache is None:
        raise ValueError("refresh_operands requires a retained cache")
    if resident_cache is not None and not resident_add32:
        raise ValueError("retained shared memory requires the resident ADD32 layout")
    resident_wide = resident_wide or resident_add32
    resident_grid = resident_grid or resident_wide
    warp_tags = WIDE_WARP_TAGS if resident_wide else WARP_TAGS
    dirty_offset = WIDE_DIRTY_OFFSET if resident_wide else WARP_DIRTY_OFFSET
    plan_bytes = len(warp_tags) // 8
    resident_plan = resident_plan or resident_io
    resident_delta = resident_delta or resident_plan or resident_grid
    resident_state = resident_state or resident_delta
    resident_rom = resident_rom or resident_state
    looping = looping or resident_rom
    divergent = divergent or looping
    masked = masked or divergent
    indexed = indexed or masked
    resident = resident or indexed
    first_row_pair, row_pairs = _row_window(case, resident_grid=resident_grid)
    last_row_pair = first_row_pair + row_pairs - 1
    operand_base = INDEX_OPERAND if indexed else OPERAND
    controls = [control_token(warp) for warp in range(2)]
    previous = TICKET + 8 * first_row_pair + 1
    operands = nibbles(operand_base, _digits(case["a"] + case["b"], 8))
    memory = nibbles(C_MEMORY, _digits(case["c"], 16))
    initial_registers = case.get("initial_registers", [[0] * 8, [0] * 8])
    registers = [
        nibbles(REGISTER, _digits([word & 65535 for word in bank], 16))
        for bank in initial_registers
    ]
    high_registers = (
        [
            nibbles(HIGH_REGISTER, _digits([word >> 16 for word in bank], 16))
            for bank in initial_registers
        ]
        if resident_wide
        else []
    )
    rom = [program_tokens(program) for program in case["programs"]]
    indices = [nibbles(INDEX_REGISTER, [0] * 16) for _ in range(2)]
    masks = [nibbles(MASK_REGISTER, [15, 15]) for _ in range(2)]
    branches = [nibbles(BRANCH_REGISTER, [0] * BRANCH_NIBBLES) for _ in range(2)]
    loops = [nibbles(LOOP_REGISTER, [0] * LOOP_NIBBLES) for _ in range(2)]

    def state_packet(warp: int) -> list[int]:
        packet = registers[warp][:]
        if indexed:
            packet.extend(indices[warp])
        if masked:
            packet.extend(masks[warp])
        if divergent:
            packet.extend(branches[warp])
        if looping:
            packet.extend(loops[warp])
        if resident_wide:
            packet.extend(high_registers[warp])
        return packet

    reads = list(range(READ_REG, READ_REG + 32))
    if indexed:
        reads.extend(range(READ_INDEX, READ_INDEX + 16))
    if masked:
        reads.extend(range(READ_MASK, READ_MASK + 2))
    if divergent:
        reads.extend(range(READ_BRANCH, READ_BRANCH + BRANCH_NIBBLES))
    if looping:
        reads.extend(range(READ_LOOP, READ_LOOP + LOOP_NIBBLES))
    if resident_wide:
        reads.extend(range(READ_HIGH, READ_HIGH + HIGH_NIBBLES))
    store_reads = list(range(READ_STORE, READ_STORE + 32))
    plan_reads = list(range(READ_PLAN, READ_PLAN + plan_bytes + int(resident_io)))
    state_reads = reads
    reads = state_reads + store_reads
    schedules = []
    trace = []
    cache = resident_cache
    pending = [*controls, previous] if resident else []
    uploads = []
    if resident:
        if resident_cache is None or refresh_operands:
            uploads.extend(operands)
        if resident_cache is None:
            uploads.extend(memory)
    pending.extend(uploads)
    transport = {
        "input_tokens": 0,
        "fresh_caches": 0,
        "matrix_initial_tokens": len(uploads),
        "matrix_patch_tokens": 0,
    }

    if resident_grid:
        pending.append(GRID + last_row_pair)
        transport["grid_initial_tokens"] = 1

    if resident_add32:
        transport["high_register_updates"] = 0

    if resident_rom:
        for warp, tokens in enumerate(rom):
            pending.extend([ROM_BANK + warp, *tokens])
        transport["rom_program_tokens"] = sum(len(tokens) for tokens in rom)
        transport["rom_select_tokens"] = len(rom)

    if resident_state:
        initial_state = [
            bank + 16 * nib + (token & 15)
            for warp, bank in enumerate(WARP_BANKS)
            for nib, token in enumerate(state_packet(warp))
        ]
        pending.extend(initial_state)
        transport["warp_initial_tokens"] = len(initial_state)
        transport["warp_update_tokens"] = 0

    if resident_plan:
        transport.update(forward_calls=0, state_plan_tokens=0, state_read_tokens=0)

    if resident_io:
        transport.update(store_plan_tokens=0, store_read_tokens=0, skipped_read_calls=0)

    def feed(tokens: list[int], count: int):
        nonlocal cache
        if not resident:
            cache = None
        transport["input_tokens"] += len(tokens)
        transport["fresh_caches"] += int(cache is None)
        if resident_plan:
            transport["forward_calls"] += 1
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
        state = [] if resident_state else state_packet(warp)
        program = [] if resident_rom else rom[warp]
        tick_input = [*program, *prefix, *state, ticket, TICK]
        if resident_plan:
            control, *bitmap = feed([*tick_input, *plan_reads], 1 + len(plan_reads))
            plan_bits = []
            for byte, token in enumerate(bitmap):
                base = PLAN_TAG + 256 * byte
                if not base <= token < base + 256:
                    raise ValueError(f"invalid state plan byte {byte}")
                plan_bits.extend((token >> bit) & 1 for bit in range(8))
            store_bits = plan_bits[8 * plan_bytes :]
            plan_bits = plan_bits[: 8 * plan_bytes]
            selected = [nib for nib, changed in enumerate(plan_bits) if changed]
            store_slots = (
                [
                    4 * lane + digit
                    for lane, writes in enumerate(store_bits)
                    if writes
                    for digit in range(4)
                ]
                if resident_io
                else list(range(len(store_reads)))
            )
            requested = [state_reads[nib] for nib in selected]
            requested.extend(store_reads[nib] for nib in store_slots)
            output = feed(requested, len(requested)) if requested else []
            state_output = [NO_WRITE] * len(state_reads)
            for nib, token in zip(selected, output[: len(selected)], strict=True):
                state_output[nib] = token
            store_output = [NO_WRITE] * len(store_reads)
            for nib, token in zip(store_slots, output[len(selected) :], strict=True):
                if resident_io and token == NO_WRITE:
                    raise ValueError(f"missing planned shared store at nibble {nib}")
                store_output[nib] = token
            packet = state_output + store_output
            transport["state_plan_tokens"] += plan_bytes
            transport["state_read_tokens"] += len(selected)
            if resident_io:
                transport["store_plan_tokens"] += 1
                transport["store_read_tokens"] += len(store_slots)
                transport["skipped_read_calls"] += int(not requested)
        else:
            control, *packet = feed([*tick_input, *reads], 1 + len(reads))
        _check_control(control, warp)
        observed_bits = snapshot(cache, circuit, 0)
        if resident_plan and (
            observed_bits is None
            or plan_bits != observed_bits[dirty_offset : dirty_offset + len(warp_tags)]
        ):
            raise ValueError("state read plan does not match the model snapshot")
        if resident_io:
            expected_stores = [
                int(
                    bool(observed_bits[128] and observed_bits[MASK_OFFSET + lane])
                    or bool(observed_bits[SNAPSHOT_BITS + 56 + lane])
                )
                for lane in range(8)
            ]
            if store_bits != expected_stores:
                raise ValueError("shared store plan does not match the model snapshot")
        tagged_state = []
        if resident_state:
            state_output = packet[:-32]
            tagged_state = [token for token in state_output if token != NO_WRITE]
            if resident_add32:
                transport["high_register_updates"] += sum(
                    token != NO_WRITE for token in state_output[len(WARP_TAGS) :]
                )
            prior = state_packet(warp) if resident_delta else []
            expected_digits = (
                _state_snapshot_digits(observed_bits, dirty_offset)
                if resident_delta
                else []
            )
            canonical = []
            for nib, (token, tag) in enumerate(
                zip(state_output, warp_tags, strict=True)
            ):
                if resident_delta:
                    changed = expected_digits[nib] != prior[nib] & 15
                    if changed != (token != NO_WRITE) or changed != bool(
                        observed_bits[dirty_offset + nib]
                    ):
                        raise ValueError(f"incorrect warp delta at nibble {nib}")
                    if not changed:
                        canonical.append(prior[nib])
                        continue
                    if token & 15 != expected_digits[nib]:
                        raise ValueError(f"incorrect warp delta value at nibble {nib}")
                start = WARP_BANKS[warp] + 16 * nib
                if not start <= token < start + 16:
                    raise ValueError(f"invalid banked warp packet at nibble {nib}")
                canonical.append(tag + (token & 15))
            packet = canonical + packet[-32:]
        if resident_delta:
            observed_bits = observed_bits[:dirty_offset]
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
            start = 50 + BRANCH_NIBBLES
            loops[warp] = packet[start : start + LOOP_NIBBLES]
            for nib, token in enumerate(loops[warp]):
                start = LOOP_REGISTER + 16 * nib
                if not start <= token < start + (3 if nib == 0 else 16):
                    raise ValueError(f"invalid loop packet at nibble {nib}")
        if resident_wide:
            high_registers[warp] = packet[len(WARP_TAGS) : len(warp_tags)]
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
            pending = [control, *tagged_state, *updates]
            transport["matrix_patch_tokens"] += len(updates)
            if resident_state:
                transport["warp_update_tokens"] += len(tagged_state)
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
                "snapshot": hex_bits(observed_bits),
            }
        )
    if resident_grid:
        transport["resident_grid_exact"] = (
            head_row(cache, circuit, "grid", 0, 2) == last_row_pair
        )
    if resident:
        transport["resident_memory_exact"] = _check_resident_memory(
            cache, circuit, operands, memory, indexed=indexed
        )
    if resident_rom:
        transport["resident_rom_exact"] = all(
            head_row(
                cache,
                circuit,
                "rom",
                16 * warp + address,
                ROM_PAYLOAD_BITS + int(resident_add32),
            )
            == rom_payload(
                token,
                indexed=True,
                masked=True,
                divergent=True,
                looping=True,
                resident_add32=resident_add32,
                resident_shared32=resident_shared32,
                resident_addmem32=resident_addmem32,
                resident_branch32=resident_branch32,
                resident_loop32=resident_loop32,
            )
            for warp, tokens in enumerate(rom)
            for address, token in enumerate(tokens)
        )
    if resident_state:
        heads = [
            *(f"register.{nib}" for nib in range(32)),
            *(f"index.{lane}.{nib}" for lane in range(8) for nib in range(2)),
            *(f"mask.{nib}" for nib in range(2)),
            *(f"branch.{nib}" for nib in range(BRANCH_NIBBLES)),
            *(f"loop.{nib}" for nib in range(LOOP_NIBBLES)),
        ]
        if resident_wide:
            heads.extend(f"high.{nib}" for nib in range(HIGH_NIBBLES))
        transport["resident_state_exact"] = all(
            head_row(cache, circuit, head, warp, 4) == token & 15
            for warp in range(2)
            for head, token in zip(heads, state_packet(warp), strict=True)
        )
    return {
        "transport": transport,
        "schedules": schedules,
        "trace": trace,
        "outcome": outcome,
        "matrix": _matrix(memory),
        **({"shared_words": _words(operands + memory)} if indexed else {}),
        **(
            {
                "warp_registers": [
                    [
                        low | (high << 16)
                        for low, high in zip(_words(lows), _words(highs), strict=True)
                    ]
                    for lows, highs in zip(registers, high_registers, strict=True)
                ]
            }
            if resident_wide
            else {}
        ),
    }


def launch_cached_tile(
    model,
    circuit,
    case: dict,
    device: str,
    offset: int,
    *,
    cache: Cache | None = None,
) -> tuple[Cache, dict, list[dict[str, int]]]:
    """Run one ADD32 tile slice and audit actual cache and memory traffic."""
    start = cache
    audit: list[dict[str, int]] = []

    def capture(_model, _args, kwargs, output) -> None:
        nonlocal cache
        if kwargs["past_key_values"] is not cache:
            raise ValueError("tile call did not receive its previous cache")
        if cache is not None and output.past_key_values is not cache:
            raise ValueError("tile call replaced its previous cache")
        cache = output.past_key_values
        tokens = [token - offset for token in kwargs["input_ids"][0].tolist()]
        audit.append(
            {
                "operands": sum(
                    INDEX_OPERAND <= token < INDEX_OPERAND + 4096 for token in tokens
                ),
                "c": sum(C_MEMORY <= token < C_MEMORY + 4096 for token in tokens),
            }
        )

    handle = model.register_forward_hook(capture, with_kwargs=True)
    try:
        result = execute(
            model,
            circuit,
            case,
            device,
            256,
            offset,
            resident_add32=True,
            resident_io=True,
            resident_cache=start,
            refresh_operands=start is not None,
        )
    finally:
        handle.remove()
    transport = result["transport"]
    initial_c = 256 if start is None else 0
    if (
        cache is None
        or not audit
        or audit[0] != {"operands": 256, "c": initial_c}
        or sum(call["operands"] for call in audit) != 256
        or sum(call["c"] for call in audit)
        != initial_c + transport["matrix_patch_tokens"]
        or transport["matrix_initial_tokens"] != 256 + initial_c
        or transport["fresh_caches"] != int(start is None)
    ):
        raise ValueError("tile slice memory transport audit failed")
    return cache, result, audit


def _state_snapshot_digits(bits: list[int] | None, dirty_offset: int) -> list[int]:
    if bits is None:
        raise ValueError("missing warp snapshot")
    index_bits = [
        bit
        for start in range(SNAPSHOT_BITS, SNAPSHOT_BITS + 56, 7)
        for bit in (*bits[start : start + 7], 0)
    ]
    state = bits[:128] + index_bits + bits[MASK_OFFSET:dirty_offset]
    return [
        sum(state[start + bit] << bit for bit in range(4))
        for start in range(0, len(state), 4)
    ]


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
    tile, row_pairs = _row_window(case, resident_grid=True)
    last_tile = tile + row_pairs - 1
    last = 1
    wide = "initial_registers" in case
    initial = case.get("initial_registers", [[0] * 8, [0] * 8])
    acc = [[word & 65535 for word in bank] for bank in initial]
    high = [[word >> 16 for word in bank] for bank in initial]
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
        advance = states == [1, 1] and tile < last_tile
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
        elif op == ADD32I_OP:
            words = [
                ((upper << 16) + lower + arg) & 0xFFFFFFFF
                for upper, lower in zip(high[warp], before, strict=True)
            ]
            candidate = [word & 65535 for word in words]
            high[warp] = [
                word >> 16 if active >> lane & 1 else high[warp][lane]
                for lane, word in enumerate(words)
            ]
        elif op == "BARRIER":
            states[warp] = 2
        elif op == "SAVE_INDEX":
            indices[warp] = [
                v % INDEX_WORDS if active >> lane & 1 else indices[warp][lane]
                for lane, v in enumerate(acc[warp])
            ]
        elif op in ("LOAD_INDEX", LOAD_INDEX_HI, *ADDMEM32_OPS):
            high_access = op in (LOAD_INDEX_HI, ADD_INDEX_HI)
            addresses = [addr ^ arg ^ int(high_access) for addr in indices[warp]]
            loaded = [
                int.from_bytes(operands[2 * addr : 2 * addr + 2], "little")
                if addr < 64
                else c[addr - 64]
                for addr in addresses
            ]
            if op == ADD_INDEX_LO:
                sums = [
                    left + right for left, right in zip(before, loaded, strict=True)
                ]
                candidate = [value & 65535 for value in sums]
                high[warp] = [
                    (upper + (sums[lane] >> 16)) & 65535
                    if active >> lane & 1
                    else upper
                    for lane, upper in enumerate(high[warp])
                ]
            elif high_access:
                high[warp] = [
                    (
                        (high[warp][lane] + value) & 65535
                        if op == ADD_INDEX_HI
                        else value
                    )
                    if active >> lane & 1
                    else high[warp][lane]
                    for lane, value in enumerate(loaded)
                ]
            else:
                candidate = loaded
        elif op == "MASK_ALL":
            masks[warp] = 255
        elif op == "MASK_BIT":
            masks[warp] = sum(((v >> arg) & 1) << lane for lane, v in enumerate(before))
        elif op == "IF_BIT":
            values = before if arg < 16 else high[warp]
            selected = sum(
                ((v >> (arg % 16)) & 1) << lane for lane, v in enumerate(values)
            )
            masks[warp] = active & selected
            branches[warp].append((active, masks[warp]))
        elif op == "ELSE":
            parent, then_mask = branches[warp][-1]
            masks[warp] = parent & ~then_mask
        elif op == "ENDIF":
            parent, _ = branches[warp].pop()
            masks[warp] = parent
        elif op == "WHILE_BIT":
            values = before if arg < 16 else high[warp]
            condition = sum(
                ((v >> (arg % 16)) & 1) << lane for lane, v in enumerate(values)
            )
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
        if op in ("STORE_INDEX", STORE_INDEX_HI):
            high_store = op == STORE_INDEX_HI
            words = high[warp] if high_store else acc[warp]
            claimed = set()
            for lane, (addr, word) in enumerate(zip(indices[warp], words, strict=True)):
                addr ^= int(high_store)
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
        if wide:
            packed |= sum(
                word << (WARP_DIRTY_OFFSET + 16 * lane)
                for lane, word in enumerate(high[warp])
            )
            width = WIDE_DIRTY_OFFSET // 4
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
                "warp_registers": [
                    [low | (hi << 16) for low, hi in zip(lows, highs, strict=True)]
                    for lows, highs in zip(acc, high, strict=True)
                ]
            }
            if wide
            else {}
        ),
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
    resident_rom: bool = False,
    resident_state: bool = False,
    resident_delta: bool = False,
    resident_plan: bool = False,
    resident_io: bool = False,
    resident_grid: bool = False,
    resident_wide: bool = False,
    resident_add32: bool = False,
    resident_shared32: bool = False,
    resident_addmem32: bool = False,
    resident_branch32: bool = False,
    resident_loop32: bool = False,
):
    if checkpoint is not None and resident_add32:
        layout = json.loads((checkpoint / "qwengpu9b.json").read_text())["layout"]
        resident_loop32 = layout == RESIDENT_LOOP32_LAYOUT
        resident_branch32 = layout in RESIDENT_BRANCH32_LAYOUTS
        resident_addmem32 = layout in RESIDENT_ADDMEM32_LAYOUTS
        resident_shared32 = layout in RESIDENT_SHARED32_LAYOUTS
    if checkpoint is None:
        model, circuit = build_standalone(
            resident=resident,
            indexed=indexed,
            masked=masked,
            divergent=divergent,
            looping=looping,
            resident_rom=resident_rom,
            resident_state=resident_state,
            resident_delta=resident_delta,
            resident_plan=resident_plan,
            resident_io=resident_io,
            resident_grid=resident_grid,
            resident_wide=resident_wide,
            resident_add32=resident_add32,
            resident_shared32=resident_shared32,
            resident_addmem32=resident_addmem32,
            resident_branch32=resident_branch32,
            resident_loop32=resident_loop32,
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
            resident_rom=resident_rom,
            resident_state=resident_state,
            resident_delta=resident_delta,
            resident_plan=resident_plan,
            resident_io=resident_io,
            resident_grid=resident_grid,
            resident_wide=resident_wide,
            resident_add32=resident_add32,
            resident_shared32=resident_shared32,
            resident_addmem32=resident_addmem32,
            resident_branch32=resident_branch32,
            resident_loop32=resident_loop32,
        )
    return model.to(device).eval(), circuit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path)
    parser.add_argument("--kernel", type=Path, action="append")
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--resident", action="store_true")
    variants.add_argument("--indexed", action="store_true")
    variants.add_argument("--masked", action="store_true")
    variants.add_argument("--divergent", action="store_true")
    variants.add_argument("--looping", action="store_true")
    variants.add_argument("--resident-rom", action="store_true")
    variants.add_argument("--resident-state", action="store_true")
    variants.add_argument("--resident-delta", action="store_true")
    variants.add_argument("--resident-plan", action="store_true")
    variants.add_argument("--resident-io", action="store_true")
    variants.add_argument("--resident-grid", action="store_true")
    variants.add_argument("--resident-wide", action="store_true")
    variants.add_argument("--resident-add32", action="store_true")
    variants.add_argument("--resident-shared32", action="store_true")
    variants.add_argument("--resident-addmem32", action="store_true")
    variants.add_argument("--resident-branch32", action="store_true")
    variants.add_argument("--resident-loop32", action="store_true")
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    meta = None
    resident_loop32 = args.resident_loop32
    resident_branch32 = args.resident_branch32 or resident_loop32
    resident_addmem32 = args.resident_addmem32 or resident_branch32
    resident_shared32 = args.resident_shared32 or resident_addmem32
    resident_add32 = args.resident_add32 or resident_shared32
    resident_wide = args.resident_wide or resident_add32
    resident_grid = args.resident_grid or resident_wide
    resident_io = args.resident_io or resident_grid
    resident_plan = args.resident_plan or resident_io
    resident_delta = args.resident_delta or resident_plan
    resident_state = args.resident_state or resident_delta
    resident_rom = args.resident_rom or resident_state
    looping = args.looping or resident_rom
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
            RESIDENT_ROM_LAYOUT,
            RESIDENT_STATE_LAYOUT,
            RESIDENT_DELTA_LAYOUT,
            RESIDENT_PLAN_LAYOUT,
            RESIDENT_IO_LAYOUT,
            RESIDENT_GRID_LAYOUT,
            RESIDENT_WIDE_LAYOUT,
            *RESIDENT_ADD32_LAYOUTS,
        ):
            raise ValueError("not an 8x8 tile checkpoint")
        resident_loop32 = layout == RESIDENT_LOOP32_LAYOUT
        resident_branch32 = layout in RESIDENT_BRANCH32_LAYOUTS
        resident_addmem32 = layout in RESIDENT_ADDMEM32_LAYOUTS
        resident_shared32 = layout in RESIDENT_SHARED32_LAYOUTS
        resident_add32 = layout in RESIDENT_ADD32_LAYOUTS
        resident_wide = resident_add32 or layout == RESIDENT_WIDE_LAYOUT
        resident_grid = resident_wide or layout == RESIDENT_GRID_LAYOUT
        resident_io = resident_grid or layout == RESIDENT_IO_LAYOUT
        resident_plan = resident_io or layout == RESIDENT_PLAN_LAYOUT
        resident_delta = resident_plan or layout == RESIDENT_DELTA_LAYOUT
        resident_state = resident_delta or layout == RESIDENT_STATE_LAYOUT
        resident_rom = resident_state or layout == RESIDENT_ROM_LAYOUT
        looping = resident_rom or layout == LOOP_LAYOUT
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
        if args.resident_rom and not resident_rom:
            raise ValueError("--resident-rom requires a resident ROM checkpoint")
        if args.resident_state and not resident_state:
            raise ValueError("--resident-state requires a resident state checkpoint")
        if args.resident_delta and not resident_delta:
            raise ValueError("--resident-delta requires a resident delta checkpoint")
        if args.resident_plan and not resident_plan:
            raise ValueError("--resident-plan requires a resident plan checkpoint")
        if args.resident_io and not resident_io:
            raise ValueError("--resident-io requires a resident I/O checkpoint")
        if args.resident_grid and not resident_grid:
            raise ValueError("--resident-grid requires a resident grid checkpoint")
        if args.resident_wide and not resident_wide:
            raise ValueError("--resident-wide requires a resident wide checkpoint")
        if args.resident_add32 and not resident_add32:
            raise ValueError("--resident-add32 requires a resident ADD32 checkpoint")
        if args.resident_shared32 and not resident_shared32:
            raise ValueError("--resident-shared32 requires a shared32 checkpoint")
        if args.resident_addmem32 and not resident_addmem32:
            raise ValueError("--resident-addmem32 requires an addmem32 checkpoint")
        if args.resident_branch32 and not resident_branch32:
            raise ValueError("--resident-branch32 requires a branch32 checkpoint")
        if args.resident_loop32 and not resident_loop32:
            raise ValueError("--resident-loop32 requires a loop32 checkpoint")
    paths = args.kernel or sorted(FIXTURES.glob("*.json"))
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
    if resident_grid and not args.kernel:
        paths.extend(sorted(GRID_FIXTURES.glob("*.json")))
    if resident_wide and not args.kernel:
        paths.extend(sorted(WIDE_FIXTURES.glob("*.json")))
    if resident_add32 and not args.kernel:
        paths.extend(sorted(ADD32I_FIXTURES.glob("*.json")))
    if resident_shared32 and not args.kernel:
        paths.extend(sorted(SHARED32_FIXTURES.glob("*.json")))
    if resident_addmem32 and not args.kernel:
        paths.extend(sorted(ADDMEM32_FIXTURES.glob("*.json")))
    if resident_branch32 and not args.kernel:
        paths.extend(sorted(BRANCH32_FIXTURES.glob("*.json")))
    if resident_loop32 and not args.kernel:
        paths.extend(sorted(LOOP32_FIXTURES.glob("*.json")))
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
                    resident_grid=resident_grid,
                    resident_wide=resident_wide,
                    resident_add32=resident_add32,
                    resident_shared32=resident_shared32,
                    resident_addmem32=resident_addmem32,
                    resident_branch32=resident_branch32,
                    resident_loop32=resident_loop32,
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
        resident_rom=resident_rom,
        resident_state=resident_state,
        resident_delta=resident_delta,
        resident_plan=resident_plan,
        resident_io=resident_io,
        resident_grid=resident_grid,
        resident_wide=resident_wide,
        resident_add32=resident_add32,
        resident_shared32=resident_shared32,
        resident_addmem32=resident_addmem32,
        resident_branch32=resident_branch32,
        resident_loop32=resident_loop32,
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
            resident_rom=resident_rom,
            resident_state=resident_state,
            resident_delta=resident_delta,
            resident_plan=resident_plan,
            resident_io=resident_io,
            resident_grid=resident_grid,
            resident_wide=resident_wide,
            resident_add32=resident_add32,
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
            and (not resident_grid or transport["resident_grid_exact"])
            and (not resident_rom or transport["resident_rom_exact"])
            and (not resident_state or transport["resident_state_exact"])
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
