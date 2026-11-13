"""Run 8x8 matmul through four model-selected output tiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
from chipformer.q35.ir import Target
from transformers.cache_utils import Cache

from qwengpu.alu32 import (
    ALU_LIMITS, ALU_OPS, ALU_SIGNED_ARGS, FULL_INTEGER_LAYOUT, FULL_INTEGER_OPS,
    LOGIC_OPS, RUNTIME_EXTENT_BASE, RUNTIME_EXTENT_MAX, RUNTIME_EXTENT_OP, WIDE_INDEX_OPS,
)
from qwengpu.compiled_tile import (
    READY0,
    READY1,
    SCHEDULE,
    TICK,
    TICKET,
    control_token as legacy_control_token,
)
from qwengpu.compiled_tile8 import (
    ADD32I_OP,
    ADD_INDEX_HI,
    ADD_INDEX_LO,
    ADDMEM32_OPS,
    ADVANCE,
    ALT_OPERAND,
    BOTTOM_OPERANDS,
    BRANCH_NIBBLES,
    BRANCH_OFFSET,
    BRANCH_OPS,
    BRANCH_REGISTER,
    C_MEMORY,
    COMPUTE_OPS,
    DEADLOCK,
    DONE,
    GLOBAL_GRID,
    GLOBAL_OPS,
    GRID,
    HIGH_NIBBLES,
    HIGH_REGISTER,
    ID_OPS,
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
    NEXT_BANK,
    NO_WRITE,
    OPERAND,
    OPS,
    OUTPUT_BANKS,
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
    RESET_WARP,
    RESIDENT_ADDMEM32_LAYOUT,
    RESIDENT_BRANCH32_LAYOUT,
    RESIDENT_COMPUTE_LAYOUT,
    RESIDENT_GLOBALGRID_LAYOUT,
    RESIDENT_LOOP32_LAYOUT,
    RESIDENT_SHARED32_LAYOUT,
    RESIDENT_TILEBANKS_LAYOUT,
    RESIDENT_TILEGRID_LAYOUT,
    RESIDENT_TILEPIPE_LAYOUT,
    RIGHT_OPERANDS,
    ROM_BANK,
    ROM_PAYLOAD_BITS,
    ROM32_PC_HIGH,
    ROW_PAIRS,
    SECOND_OUTPUT,
    SHARED32_OPS,
    SNAPSHOT_BITS,
    STORE_INDEX_HI,
    WARP_BANKS,
    WARP_DIRTY_OFFSET,
    WARP_TAGS,
    WIDE_DIRTY_OFFSET,
    WIDE_WARP_TAGS,
    WORKGROUP_TICKETS,
    build,
    build_standalone,
    nibbles,
    program_tokens,
    rom_payload,
)
from qwengpu.eval_compiled_rom import head_row
from qwengpu.eval_control import hex_bits, snapshot
from qwengpu.eval_tile import _check_control as check_legacy_control
from qwengpu.eval_tile import _unsigned
from qwengpu.memory_geometry import (
    LEGACY_MEMORY,
    ALU32_LAYOUTS,
    MEMORY512_LAYOUTS,
    ROM32_LAYOUTS,
    WIDE_MEMORY,
)

from qwengpu.resident_scheduler import (
    EXTRA_WARP_BANKS,
    RESIDENT_DISABLE,
    RESIDENT_ENABLE,
    RESIDENT_LAYOUT,
    RESIDENT_READY,
    RESIDENT_TICKET,
    reference_dispatch,
    resident_control,
)
from qwengpu.scoped_memory import (
    SCOPED_BASE,
    SCOPED_END,
    SCOPED_LAYOUT,
    SCOPED_LOADS,
    SCOPED_OPS,
    SCOPED_STORES,
    SCOPED_WORDS,
    physical_word,
)

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
RESIDENT_TILEPIPE_LAYOUTS = (
    *MEMORY512_LAYOUTS,
    RESIDENT_TILEPIPE_LAYOUT,
    RESIDENT_TILEGRID_LAYOUT,
    RESIDENT_GLOBALGRID_LAYOUT,
    RESIDENT_COMPUTE_LAYOUT,
)
RESIDENT_GLOBALGRID_LAYOUTS = (
    RESIDENT_GLOBALGRID_LAYOUT,
    RESIDENT_COMPUTE_LAYOUT,
    *MEMORY512_LAYOUTS,
)
RESIDENT_TILEBANKS_LAYOUTS = (RESIDENT_TILEBANKS_LAYOUT, *RESIDENT_TILEPIPE_LAYOUTS)
RESIDENT_LOOP32_LAYOUTS = (RESIDENT_LOOP32_LAYOUT, *RESIDENT_TILEBANKS_LAYOUTS)
RESIDENT_BRANCH32_LAYOUTS = (RESIDENT_BRANCH32_LAYOUT, *RESIDENT_LOOP32_LAYOUTS)
RESIDENT_ADDMEM32_LAYOUTS = (RESIDENT_ADDMEM32_LAYOUT, *RESIDENT_BRANCH32_LAYOUTS)
RESIDENT_SHARED32_LAYOUTS = (RESIDENT_SHARED32_LAYOUT, *RESIDENT_ADDMEM32_LAYOUTS)
RESIDENT_ADD32_LAYOUTS = (RESIDENT_ADD32_LAYOUT, *RESIDENT_SHARED32_LAYOUTS)
RESIDENT_TRANSPORTS = {
    "fixed": {"resident_delta": True},
    "state_plan": {"resident_plan": True},
    "state_store_plan": {"resident_io": True},
    "fused_dispatch": {"resident_io": True, "fused_dispatch": True},
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


@dataclass(frozen=True)
class BottomRow:
    operands: tuple[list[int], list[int]]
    outputs: tuple[list[int], list[int]]


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


def control_token(warp: int, pc: int = 0, status: int = 0) -> int:
    return legacy_control_token(warp, pc & 15, status) + (pc >> 4) * ROM32_PC_HIGH


def _check_control(
    token: int, warp: int, resident_rom32: bool, interleaved: bool = False
) -> None:
    if interleaved:
        if not resident_control(warp) <= token < resident_control(warp) + 96:
            raise ValueError(
                f"invalid resident control packet for warp {warp}: {token}"
            )
        return
    if resident_rom32 and token & ROM32_PC_HIGH:
        token -= ROM32_PC_HIGH
    check_legacy_control(token, warp)


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
    resident_globalgrid: bool = False,
    resident_compute: bool = False,
    resident_memory512: bool = False,
    resident_rom32: bool = False,
    resident_alu32: bool = False,
    resident_scoped: bool = False,
    resident_interleaved: bool = False,
    full_integer: bool = False,
) -> dict:
    resident_interleaved = resident_interleaved or full_integer
    resident_scoped = resident_scoped or resident_interleaved
    resident_alu32 = resident_alu32 or resident_scoped
    resident_rom32 = resident_rom32 or resident_alu32
    resident_memory512 = resident_memory512 or resident_rom32
    resident_compute = resident_compute or resident_memory512
    resident_globalgrid = resident_globalgrid or resident_compute
    resident_loop32 = resident_loop32 or resident_globalgrid
    resident_branch32 = resident_branch32 or resident_loop32
    resident_addmem32 = resident_addmem32 or resident_branch32
    resident_shared32 = resident_shared32 or resident_addmem32
    resident_add32 = resident_add32 or resident_shared32
    looping = looping or resident_shared32
    divergent = divergent or looping
    masked = masked or divergent
    if not isinstance(data, dict):
        raise ValueError("8x8 case must be an object")
    workgroups = data.get("workgroups")
    interleaved = data.get("interleaved", False)
    if type(interleaved) is not bool or (
        interleaved and not (resident_interleaved and workgroups == 2)
    ):
        raise ValueError(
            "interleaved mode requires the resident layout and exactly two workgroups"
        )
    if "scoped_words" in data and not (resident_scoped and workgroups):
        raise ValueError("scoped_words requires a scoped workgroup launch")
    extent = data.get("extent")
    if "extent" in data and not (
        full_integer and workgroups and type(extent) is int and 0 <= extent <= RUNTIME_EXTENT_MAX
    ):
        raise ValueError("extent requires a full integer grid and a value from 0 through 64")
    if "workgroups" in data:
        if (
            not resident_globalgrid
            or type(workgroups) is not int
            or not 1 <= workgroups <= 4
        ):
            raise ValueError(
                "workgroups requires globalgrid and an integer from 1 through 4"
            )
        if any(
            name in data
            for name in (
                "a",
                "b",
                "c",
                "initial_registers",
                "row_pairs",
                "first_row_pair",
            )
        ):
            raise ValueError(
                "globalgrid launches use global_words and zero-initialized warp state"
            )
        geometry = WIDE_MEMORY if resident_memory512 else LEGACY_MEMORY
        words = _unsigned(
            data.get("global_words", [0] * geometry.words),
            geometry.words,
            65535,
            "global_words",
        )
        operands = [
            byte
            for word in (words[:64] if not resident_memory512 else [0] * 64)
            for byte in (word & 255, word >> 8)
        ]
        a, b, initial = (
            operands[:64],
            operands[64:],
            words[64:] if not resident_memory512 else [0] * 64,
        )
    else:
        if "global_words" in data:
            raise ValueError("global_words requires a workgroups descriptor")
        a = _unsigned(data.get("a", [0] * 64), 64, 255, "a")
        b = _unsigned(data.get("b", [0] * 64), 64, 255, "b")
        initial = _unsigned(data.get("c", [0] * 64), 64, 65535, "c")
    resident_wide = resident_wide or resident_add32
    first_row_pair, row_pairs = (
        (0, 1)
        if workgroups
        else _row_window(data, resident_grid=resident_grid or resident_wide)
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
        if workgroups:
            raise ValueError("globalgrid requires two explicit warp programs")
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
            + (GLOBAL_OPS if workgroups else ())
            + (COMPUTE_OPS if workgroups and resident_compute else ())
            + (ALU_OPS if resident_alu32 else ())
            + (SCOPED_OPS if resident_scoped and workgroups else ())
            + (FULL_INTEGER_OPS if full_integer and workgroups else ())
        )
        slots = 32 if resident_rom32 else 16
        for warp, program in enumerate(raw):
            if not isinstance(program, list) or not 1 <= len(program) <= slots:
                raise ValueError(
                    f"each tile program must contain 1 through {slots} instructions"
                )
            ops = []
            for index, ins in enumerate(program):
                if not isinstance(ins, dict) or ins.get("op") not in names:
                    raise ValueError("invalid tile opcode")
                op, arg = ins["op"], ins.get("arg", 0)
                if workgroups and op in ("LOAD_TILE", "MAC_K", "STORE_TILE"):
                    raise ValueError(f"{op} requires a tiled launch")
                limit = {
                    "MAC_K": 7,
                    "ADD16I": 255,
                    ADD32I_OP: 255,
                    ADD_INDEX_LO: 254 if resident_memory512 and workgroups else 126,
                    ADD_INDEX_HI: 254 if resident_memory512 and workgroups else 126,
                    "MASK_BIT": 15,
                    "IF_BIT": 31 if resident_branch32 else 15,
                    "WHILE_BIT": 31 if resident_loop32 else 15,
                    "SHL32I": 7,
                    "MASK_LANES": 8,
                    "MASK_GLOBAL_LT": 64,
                    "MATRIX_MAC8": 63,
                    **ALU_LIMITS,
                }.get(op, 0)
                lower, upper = (-128, 127) if op in ALU_SIGNED_ARGS else (0, limit)
                if type(arg) is not int or not lower <= arg <= upper:
                    raise ValueError(f"invalid {op} argument")
                if op in ADDMEM32_OPS and arg % 2:
                    raise ValueError(f"{op} requires an even pair mask")
                if op == RUNTIME_EXTENT_OP and extent is None:
                    raise ValueError("MASK_EXTENT requires an extent descriptor")
                if op == "MATRIX_MAC8" and arg % 4 > arg // 16:
                    raise ValueError(
                        "MATRIX_MAC8 selects a K position outside the matrix"
                    )
                if (op == "HALT") != (index == len(program) - 1):
                    raise ValueError("HALT must appear exactly once at the end")
                ops.append((op, arg))
            if divergent:
                links[warp] = _control_links(ops)
            programs.append(tuple(ops))
    scoped_words = None
    if (
        resident_scoped
        and workgroups
        and (
            "scoped_words" in data
            or any(op in SCOPED_OPS for program in programs for op, _ in program)
        )
    ):
        scoped_words = _unsigned(
            data.get("scoped_words", [0] * SCOPED_WORDS),
            SCOPED_WORDS,
            65535,
            "scoped_words",
        )
    return {
        "a": a,
        "b": b,
        "c": initial,
        "programs": programs,
        "links": links,
        "first_row_pair": first_row_pair,
        "row_pairs": row_pairs,
        **({"initial_registers": initial_registers} if resident_wide else {}),
        **({"workgroups": workgroups} if workgroups else {}),
        **({"interleaved": True} if interleaved else {}),
        **({"extent": extent} if extent is not None else {}),
        **({"memory512": True} if resident_memory512 else {}),
        **({"global_words": words} if resident_memory512 and workgroups else {}),
        **({"scoped_words": scoped_words} if scoped_words is not None else {}),
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
    resident_memory512: bool = False,
    resident_rom32: bool = False,
    resident_alu32: bool = False,
    resident_scoped: bool = False,
    resident_interleaved: bool = False,
    full_integer: bool = False,
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
        resident_memory512=resident_memory512,
        resident_rom32=resident_rom32,
        resident_alu32=resident_alu32,
        resident_scoped=resident_scoped,
        resident_interleaved=resident_interleaved,
        full_integer=full_integer,
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
    fused_dispatch: bool = False,
    refresh_operands: bool = False,
    refresh_global_prefix: int = 0,
    alternate_operands: list[int] | None = None,
    operand_bank: int = 0,
    auto_operand_bank: bool = False,
    auto_workgroups: bool = False,
    second_output: list[int] | None = None,
    right_operands: tuple[list[int], list[int]] | None = None,
    bottom: BottomRow | None = None,
) -> dict:
    """Run a kernel with optional retained memory and fresh A and B operands."""
    resident_shared32 = circuit.resident_shared32
    resident_addmem32 = circuit.resident_addmem32
    resident_branch32 = circuit.resident_branch32
    resident_loop32 = circuit.resident_loop32
    workgroups = case.get("workgroups", 0)
    interleaved = case.get("interleaved", False)
    if fused_dispatch and not (interleaved and resident_io):
        raise ValueError("fused dispatch requires interleaved workgroups and store plans")
    if interleaved and not circuit.resident_interleaved:
        raise ValueError("interleaved mode requires the resident layout")
    warp_banks = (*WARP_BANKS, *EXTRA_WARP_BANKS) if interleaved else WARP_BANKS
    warp_count = len(warp_banks)
    encode_control = resident_control if interleaved else control_token
    memory512 = circuit.resident_memory512
    rom32 = circuit.resident_rom32
    alu32 = circuit.resident_alu32
    scoped = circuit.resident_scoped and "scoped_words" in case
    geometry = circuit.memory_geometry
    global512 = bool(workgroups and memory512)
    if type(refresh_global_prefix) is not int or not 0 <= refresh_global_prefix <= (
        geometry.words if global512 and resident_cache is not None else 0
    ):
        raise ValueError("global refresh requires a retained 512-byte cache")
    if workgroups and (
        not circuit.resident_globalgrid
        or auto_workgroups
        or auto_operand_bank
        or refresh_operands
    ):
        raise ValueError("expected a globalgrid circuit and one global memory image")
    resident_add32 = resident_add32 or resident_shared32
    if type(operand_bank) is not int or operand_bank not in (0, 1):
        raise ValueError("operand_bank must be 0 or 1")
    if operand_bank and alternate_operands is None:
        raise ValueError("bank 1 requires an alternate operand image")
    if alternate_operands is not None:
        if not circuit.resident_tilebanks or refresh_operands:
            raise ValueError("alternate operands require tilebanks without refresh")
        _unsigned(alternate_operands, 128, 255, "alternate_operands")
    if refresh_operands and resident_cache is None:
        raise ValueError("refresh_operands requires a retained cache")
    if resident_cache is not None and not resident_add32:
        raise ValueError("retained shared memory requires the resident ADD32 layout")
    resident_wide = resident_wide or resident_add32
    resident_grid = resident_grid or resident_wide
    warp_tags = WIDE_WARP_TAGS if resident_wide else WARP_TAGS
    dirty_offset = WIDE_DIRTY_OFFSET if resident_wide else WARP_DIRTY_OFFSET
    if memory512:
        dirty_offset = geometry.dirty_offset
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
    if auto_operand_bank and (
        not circuit.resident_tilepipe
        or alternate_operands is None
        or first_row_pair != 0
        or operand_bank != 0
        or resident_cache is not None
    ):
        raise ValueError(
            "automatic bank selection requires a fresh tilepipe launch from row zero"
        )
    if auto_workgroups and (not auto_operand_bank or not circuit.resident_tilegrid):
        raise ValueError(
            "automatic workgroups require the tilegrid circuit and bank selection"
        )
    if second_output is not None:
        if not auto_workgroups:
            raise ValueError("a second output requires automatic workgroups")
        _unsigned(second_output, 64, 65535, "second output")
        if any(op not in OPS for program in case["programs"] for op, _ in program):
            raise ValueError("separate outputs require tile-only programs")
        if right_operands is None:
            right_operands = (case["b"], alternate_operands[64:])
    if right_operands is not None:
        if second_output is None:
            raise ValueError("right operand panels require a second output")
        for panel in right_operands:
            _unsigned(panel, 64, 255, "right operand panel")
    if bottom is not None:
        if second_output is None:
            raise ValueError("a second workgroup row requires separate outputs")
        for panel in bottom.operands:
            _unsigned(panel, 64, 255, "bottom operand panel")
        for panel in bottom.outputs:
            _unsigned(panel, 64, 65535, "bottom output")
    operand_base = INDEX_OPERAND if indexed else OPERAND
    controls = [encode_control(warp) for warp in range(warp_count)]
    previous = RESIDENT_TICKET + 3 if interleaved else TICKET + 8 * first_row_pair + 1
    operands = nibbles(operand_base, _digits(case["a"] + case["b"], 8))
    alternate = (
        nibbles(ALT_OPERAND, _digits(alternate_operands, 8))
        if alternate_operands is not None
        else []
    )
    memory = nibbles(C_MEMORY, _digits(case["c"], 16))
    ram_tokens = (
        nibbles(geometry.base, _digits(case["global_words"], 16)) if global512 else []
    )
    scoped_tokens = (
        nibbles(SCOPED_BASE, _digits(case["scoped_words"], 16)) if scoped else []
    )
    second_memory = (
        nibbles(SECOND_OUTPUT, _digits(second_output, 16))
        if second_output is not None
        else []
    )
    right_tokens = (
        [
            nibbles(base, _digits(panel, 8))
            for base, panel in zip(RIGHT_OPERANDS, right_operands, strict=True)
        ]
        if right_operands is not None
        else []
    )
    output_memories = [memory]
    if second_memory:
        output_memories.append(second_memory)
    bottom_tokens = []
    if bottom is not None:
        output_memories.extend(
            nibbles(base, _digits(panel, 16))
            for base, panel in zip(OUTPUT_BANKS[2:], bottom.outputs, strict=True)
        )
        bottom_tokens = [
            nibbles(base, _digits(panel, 8))
            for base, panel in zip(BOTTOM_OPERANDS, bottom.operands, strict=True)
        ]
    initial_registers = (
        [[0] * 8 for _ in warp_banks]
        if interleaved
        else case.get("initial_registers", [[0] * 8, [0] * 8])
    )
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
    rom = [
        program_tokens(
            program, resident_rom32=rom32, resident_alu32=alu32, resident_scoped=scoped,
            full_integer=circuit.full_integer,
        )
        for program in case["programs"]
    ]
    indices = [nibbles(INDEX_REGISTER, [0] * 16) for _ in range(warp_count)]
    masks = [nibbles(MASK_REGISTER, [15, 15]) for _ in range(warp_count)]
    branches = [
        nibbles(BRANCH_REGISTER, [0] * BRANCH_NIBBLES) for _ in range(warp_count)
    ]
    loops = [nibbles(LOOP_REGISTER, [0] * LOOP_NIBBLES) for _ in range(warp_count)]

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
    bank_transitions = []
    workgroup_transitions = []
    cache = resident_cache
    pending = [*controls, previous] if resident else []
    if circuit.resident_interleaved:
        pending.insert(0, RESIDENT_ENABLE if interleaved else RESIDENT_DISABLE)
    uploads = []
    if resident:
        if resident_cache is None or refresh_operands:
            uploads.extend(ram_tokens if global512 else operands)
        if resident_cache is None:
            uploads.extend(alternate)
            for panel in (
                [] if global512 else output_memories + right_tokens + bottom_tokens
            ):
                uploads.extend(panel)
    refresh = ram_tokens[: 4 * refresh_global_prefix]
    pending.extend(uploads)
    pending.extend(refresh)
    pending.extend(scoped_tokens)
    if "extent" in case:
        pending.append(RUNTIME_EXTENT_BASE + case["extent"])
    transport = {
        "input_tokens": 0,
        "fresh_caches": 0,
        "matrix_initial_tokens": len(uploads),
        "matrix_patch_tokens": 0,
    }

    if refresh:
        transport["global_refresh_tokens"] = len(refresh)

    if scoped:
        transport.update(
            scoped_initial_tokens=len(scoped_tokens), scoped_patch_tokens=0
        )

    if resident_grid:
        pending.append(
            GRID + GLOBAL_GRID + ROW_PAIRS * (workgroups - 1)
            if workgroups
            else GRID
            + last_row_pair
            + ROW_PAIRS * operand_bank
            + 2 * ROW_PAIRS * int(auto_operand_bank)
            + 4 * ROW_PAIRS * int(auto_workgroups)
            + 8 * ROW_PAIRS * int(second_output is not None)
            + 16 * ROW_PAIRS * int(bottom is not None)
        )
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
            for warp, bank in enumerate(warp_banks)
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

    def memory_exact() -> bool:
        if global512:
            return all(
                head_row(
                    cache, circuit, f"ram.{lane}.{nib}.{word // 128}", word % 128, 4
                )
                == token & 15
                for word in range(256)
                for nib in range(4)
                for lane in range(8)
                for token in [ram_tokens[4 * word + nib]]
            )
        return (
            _check_resident_memory(cache, circuit, operands, memory, indexed=indexed)
            and all(
                head_row(
                    cache, circuit, f"C.{address % 32}", 8 * group + address // 32, 4
                )
                == token & 15
                for group, output in enumerate(output_memories[1:], 1)
                for address, token in enumerate(output)
            )
            and all(
                head_row(cache, circuit, f"B.{column}.{nibble}", 16 + 8 * bank + row, 4)
                == token & 15
                for bank, panel in enumerate(right_tokens)
                for address, token in enumerate(panel)
                for row, position in [divmod(address, 16)]
                for column, nibble in [divmod(position, 2)]
            )
            and all(
                head_row(
                    cache,
                    circuit,
                    f"A.bottom.{address % 2}",
                    64 * bank + address // 2,
                    4,
                )
                == token & 15
                for bank, panel in enumerate(bottom_tokens)
                for address, token in enumerate(panel)
            )
        )

    outcome = "step_limit"
    scheduled = None
    if fused_dispatch:
        transport.update(fused_dispatch_calls=0, fused_tail_flush_calls=0)
    for index in range(max_steps + 1):
        prefix = pending if resident else [*controls, previous]
        ready_tokens = RESIDENT_READY if interleaved else (READY0, READY1)
        prefetched = scheduled is not None
        if prefetched:
            emitted, scheduled = scheduled, None
        else:
            emitted = feed([*prefix, SCHEDULE, *ready_tokens], 1 + warp_count)
        ticket, *controls = emitted
        for warp, control in enumerate(controls):
            _check_control(control, warp, rom32, interleaved)
        schedules.append({"ticket": ticket, "controls": controls})
        if prefetched and (ticket in (DONE, DEADLOCK) or index == max_steps) and pending:
            feed(pending, 1)
            transport["fused_tail_flush_calls"] += 1
        if ticket in (DONE, DEADLOCK):
            outcome = "halted" if ticket == DONE else "deadlock"
            break
        group_index = (
            (ticket - RESIDENT_TICKET) // 2
            if interleaved
            else len(workgroup_transitions)
        )
        if ticket == NEXT_BANK or ticket in WORKGROUP_TICKETS:
            if ticket == NEXT_BANK:
                if not auto_operand_bank or len(bank_transitions) != group_index:
                    raise ValueError("unexpected operand bank transition")
            elif workgroups:
                if (
                    group_index >= workgroups - 1
                    or ticket != WORKGROUP_TICKETS[group_index]
                ):
                    raise ValueError("unexpected global workgroup transition")
            elif (
                not auto_workgroups
                or group_index >= (3 if bottom is not None else 1)
                or ticket != WORKGROUP_TICKETS[group_index]
                or len(bank_transitions) != group_index + 1
            ):
                raise ValueError("unexpected workgroup transition")
            if controls != [control_token(0), control_token(1)]:
                raise ValueError("transition did not reset both PCs")
            transition = {
                "ticket": ticket,
                "instructions": len(trace),
                "schedule_index": len(schedules) - 1,
                "matrix": _matrix(
                    output_memories[group_index] if second_memory else memory
                ),
                "resident_memory_exact": memory_exact(),
                "bank_before": head_row(cache, circuit, "operand.bank", 0, 1),
            }
            if ticket == NEXT_BANK:
                bank_transitions.append(transition)
            else:
                workgroup_transitions.append(
                    {
                        **transition,
                        "group_before": head_row(cache, circuit, "workgroup", 0, 2),
                    }
                )
                group_index += 1
        tag = ticket - TICKET
        if (
            not interleaved
            and ticket != NEXT_BANK
            and ticket not in WORKGROUP_TICKETS
            and (not 0 <= tag < 64 or tag & 4)
        ):
            raise ValueError(f"invalid tile scheduling ticket {ticket}")
        if index == max_steps:
            break
        if interleaved and not RESIDENT_TICKET <= ticket < RESIDENT_TICKET + warp_count:
            raise ValueError(f"invalid resident workgroup ticket {ticket}")
        warp = ticket - RESIDENT_TICKET if interleaved else ticket & 1
        previous = ticket
        if resident:
            prefix = [*(pending if prefetched else ()), *controls]
            if workgroups and ticket in WORKGROUP_TICKETS:
                prefix = [*prefix, *RESET_WARP]
                registers = [nibbles(REGISTER, [0] * 32) for _ in range(warp_count)]
                high_registers = [
                    nibbles(HIGH_REGISTER, [0] * 32) for _ in range(warp_count)
                ]
                indices = [nibbles(INDEX_REGISTER, [0] * 16) for _ in range(warp_count)]
                masks = [nibbles(MASK_REGISTER, [15, 15]) for _ in range(warp_count)]
                branches = [
                    nibbles(BRANCH_REGISTER, [0] * BRANCH_NIBBLES) for _ in range(2)
                ]
                loops = [
                    nibbles(LOOP_REGISTER, [0] * LOOP_NIBBLES)
                    for _ in range(warp_count)
                ]
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
            if fused_dispatch and requested:
                output = feed(
                    [control, *requested, SCHEDULE, *ready_tokens],
                    len(requested) + 1 + warp_count,
                )
                output, scheduled = output[: len(requested)], output[len(requested) :]
                transport["fused_dispatch_calls"] += 1
            else:
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
        _check_control(control, warp, rom32, interleaved)
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
                start = warp_banks[warp] + 16 * nib
                if not start <= token < start + 16:
                    raise ValueError(f"invalid banked warp packet at nibble {nib}")
                canonical.append(tag + (token & 15))
            packet = canonical + packet[-32:]
        if resident_delta:
            if (
                memory512
                and not workgroups
                and any(observed_bits[WIDE_DIRTY_OFFSET:dirty_offset])
            ):
                raise ValueError("legacy launch changed expanded index or mode bits")
            observed_bits = observed_bits[
                : WIDE_DIRTY_OFFSET if memory512 and not workgroups else dirty_offset
            ]
        next_registers, patch = packet[:32], packet[-32:]
        for nib, token in enumerate(next_registers):
            if not REGISTER + 16 * nib <= token < REGISTER + 16 * (nib + 1):
                raise ValueError(f"invalid register packet at nibble {nib}")
        if indexed:
            indices[warp] = packet[32:48]
            for nib, token in enumerate(indices[warp]):
                start = INDEX_REGISTER + 16 * nib
                if not start <= token < start + (16 if memory512 or not nib % 2 else 8):
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
        output_index = group_index if second_memory else 0
        output_base = OUTPUT_BANKS[output_index]
        for token in updates:
            if (
                global512
                and geometry.base <= token < geometry.base + 64 * geometry.words
            ):
                ram_tokens[(token - geometry.base) // 16] = token
            elif scoped and SCOPED_BASE <= token < SCOPED_END:
                scoped_tokens[(token - SCOPED_BASE) // 16] = token
                transport["scoped_patch_tokens"] += 1
            elif output_base <= token < output_base + 4096:
                output_memories[output_index][(token - output_base) // 16] = token
            elif indexed and INDEX_OPERAND <= token < INDEX_OPERAND + 4096:
                operands[(token - INDEX_OPERAND) // 16] = token
            else:
                raise ValueError("invalid tagged shared-memory write")
        if resident:
            pending = [*([control] if scheduled is None else []), *tagged_state, *updates]
            transport["matrix_patch_tokens"] += sum(
                not scoped or not SCOPED_BASE <= token < SCOPED_END for token in updates
            )
            if resident_state:
                transport["warp_update_tokens"] += len(tagged_state)
        controls = controls[:]
        controls[warp] = control
        registers[warp] = next_registers
        trace.append(
            {
                "warp": warp,
                **({"workgroup": group_index} if workgroups else {}),
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
        transport["resident_memory_exact"] = memory_exact()
    if alternate:
        expected_grid = last_row_pair + ROW_PAIRS * operand_bank
        bank_exact = head_row(cache, circuit, "grid", 0, 3) == expected_grid
        if circuit.resident_tilepipe:
            expected_bank = (
                int(bool(bank_transitions)) if auto_operand_bank else operand_bank
            )
            bank_exact = bank_exact and (
                head_row(cache, circuit, "operand.bank", 0, 1) == expected_bank
            )
        transport["resident_operand_banks_exact"] = bank_exact and _check_tile_operands(
            cache, circuit, alternate, bank=1
        )
    if resident_rom:
        transport["resident_rom_exact"] = all(
            head_row(
                cache,
                circuit,
                "rom",
                len(tokens) * warp + address,
                ROM_PAYLOAD_BITS + int(resident_add32) + int(rom32) + int(alu32),
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
                resident_globalgrid=circuit.resident_globalgrid,
                resident_compute=circuit.resident_compute,
                resident_memory512=memory512,
                resident_rom32=rom32,
                resident_alu32=alu32,
                resident_scoped=scoped,
                full_integer=circuit.full_integer,
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
            for warp in range(warp_count)
            for head, token in zip(heads, state_packet(warp), strict=True)
        )
    if scoped:
        transport["resident_scoped_exact"] = all(
            head_row(cache, circuit, f"shared.{lane}.{nib}", word, 4)
            == scoped_tokens[4 * word + nib] & 15
            for word in range(SCOPED_WORDS)
            for nib in range(4)
            for lane in range(8)
        )
    return {
        "transport": transport,
        **({"scoped_words": _words(scoped_tokens)} if scoped else {}),
        **(
            {
                "bank_transitions": bank_transitions,
                "workgroup_transitions": workgroup_transitions,
                "workgroup_cursor": (trace[-1]["workgroup"] if trace else 0)
                if interleaved
                else head_row(cache, circuit, "workgroup", 0, 2),
            }
            if auto_workgroups or workgroups
            else {"bank_transition": bank_transitions[0] if bank_transitions else None}
            if auto_operand_bank
            else {}
        ),
        "schedules": schedules,
        "trace": trace,
        "outcome": outcome,
        "matrix": _matrix(ram_tokens[256:512])
        if global512
        else _matrix(output_memories[-1]),
        **(
            {"global_words": _words(ram_tokens if global512 else operands + memory)}
            if workgroups
            else {}
        ),
        **(
            {"workgroup_outputs": [_matrix(output) for output in output_memories]}
            if second_memory
            else {}
        ),
        **(
            {"shared_words": _words(ram_tokens if global512 else operands + memory)}
            if indexed
            else {}
        ),
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
    alternate_operands: list[int] | None = None,
    operand_bank: int = 0,
    auto_operand_bank: bool = False,
    auto_workgroups: bool = False,
    second_output: list[int] | None = None,
    right_operands: tuple[list[int], list[int]] | None = None,
    bottom: BottomRow | None = None,
    retain_operands: bool = False,
) -> tuple[Cache, dict, list[dict[str, int]]]:
    """Run one ADD32 tile slice and audit actual cache and memory traffic."""
    start = cache
    audit: list[dict[str, int]] = []
    banked = alternate_operands is not None

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
                **(
                    {
                        "grid": sum(
                            GRID
                            <= token
                            < GRID + (32 if auto_workgroups else 4) * ROW_PAIRS
                            for token in tokens
                        ),
                        "transition": tokens.count(NEXT_BANK),
                    }
                    if auto_operand_bank
                    else {}
                ),
                **(
                    {
                        "workgroup_transition": sum(
                            token in WORKGROUP_TICKETS for token in tokens
                        )
                    }
                    if auto_workgroups
                    else {}
                ),
                **(
                    {
                        "second_output": sum(
                            SECOND_OUTPUT <= token < SECOND_OUTPUT + 4096
                            for token in tokens
                        ),
                        "right_operands": sum(
                            base <= token < base + 2048
                            for token in tokens
                            for base in RIGHT_OPERANDS
                        ),
                    }
                    if second_output is not None
                    else {}
                ),
                **(
                    {
                        "alternate_operands": sum(
                            ALT_OPERAND <= token < ALT_OPERAND + 4096
                            for token in tokens
                        )
                    }
                    if banked
                    else {}
                ),
                **(
                    {
                        "bottom_operands": sum(
                            base <= token < base + 2048
                            for token in tokens
                            for base in BOTTOM_OPERANDS
                        ),
                        "bottom_outputs": sum(
                            base <= token < base + 4096
                            for token in tokens
                            for base in OUTPUT_BANKS[2:]
                        ),
                    }
                    if bottom is not None
                    else {}
                ),
            }
        )

    handle = model.register_forward_hook(capture, with_kwargs=True)
    try:
        result = execute(
            model,
            circuit,
            case,
            device,
            1024 if bottom is not None else 512 if auto_workgroups else 256,
            offset,
            resident_add32=True,
            resident_io=True,
            resident_cache=start,
            refresh_operands=start is not None and not banked and not retain_operands,
            alternate_operands=alternate_operands,
            operand_bank=operand_bank,
            auto_operand_bank=auto_operand_bank,
            auto_workgroups=auto_workgroups,
            second_output=second_output,
            right_operands=right_operands,
            bottom=bottom,
        )
    finally:
        handle.remove()
    transport = result["transport"]
    initial_c = 256 if start is None else 0
    initial_operands = 0 if start is not None and (banked or retain_operands) else 256
    initial_counts = {"operands": initial_operands, "c": initial_c}
    initial_second = 256 if second_output is not None else 0
    initial_bottom = 768 if bottom is not None else 0
    if bottom is not None:
        initial_counts.update(bottom_operands=256, bottom_outputs=512)
    if second_output is not None:
        initial_counts["second_output"] = initial_second
        initial_counts["right_operands"] = 256
    if auto_operand_bank:
        initial_counts.update(grid=1, transition=0)
    if auto_workgroups:
        initial_counts["workgroup_transition"] = 0
    if banked:
        initial_counts["alternate_operands"] = initial_operands
    if (
        cache is None
        or not audit
        or audit[0] != initial_counts
        or sum(call["operands"] for call in audit) != initial_operands
        or (
            banked
            and sum(call["alternate_operands"] for call in audit) != initial_operands
        )
        or sum(
            call["c"] + call.get("second_output", 0) + call.get("bottom_outputs", 0)
            for call in audit
        )
        != initial_c
        + initial_second
        + (512 if bottom is not None else 0)
        + transport["matrix_patch_tokens"]
        or any(
            call.get("right_operands", 0) or call.get("bottom_operands", 0)
            for call in audit[1:]
        )
        or transport["matrix_initial_tokens"]
        != initial_c
        + 2 * initial_second
        + initial_bottom
        + initial_operands * (1 + int(banked))
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
        for bit in (
            *bits[start : start + 7],
            bits[WIDE_MEMORY.extra_index_offset + (start - SNAPSHOT_BITS) // 7]
            if dirty_offset == WIDE_MEMORY.dirty_offset
            else 0,
        )
    ]
    state = (
        bits[:128]
        + index_bits
        + bits[
            MASK_OFFSET : WIDE_DIRTY_OFFSET
            if dirty_offset == WIDE_MEMORY.dirty_offset
            else dirty_offset
        ]
    )
    return [
        sum(state[start + bit] << bit for bit in range(4))
        for start in range(0, len(state), 4)
    ]


def _check_tile_operands(cache, circuit, operands: list[int], *, bank: int = 0) -> bool:
    for address, token in enumerate(operands):
        byte, nibble = divmod(address, 2)
        if byte < 64:
            head, slot = f"A.{nibble}", byte
        else:
            row, col = divmod(byte - 64, 8)
            head, slot = f"B.{col}.{nibble}", row
        if (
            head_row(cache, circuit, head, slot + bank * (64 if byte < 64 else 8), 4)
            != token & 15
        ):
            return False
    return True


def _check_resident_memory(
    cache, circuit, operands: list[int], memory: list[int], *, indexed: bool = False
) -> bool:
    if not _check_tile_operands(cache, circuit, operands):
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
    alternate_operands: list[int] | None = None,
    operand_bank: int = 0,
) -> dict:
    divergent = divergent or looping
    masked = masked or divergent
    indexed = indexed or masked
    interleaved = case.get("interleaved", False)
    warp_count = 4 if interleaved else 2
    pcs = [0] * warp_count
    states = [0] * warp_count
    encode_control = resident_control if interleaved else control_token
    tile, row_pairs = _row_window(case, resident_grid=True)
    last_tile = tile + row_pairs - 1
    last = warp_count - 1
    workgroups = case.get("workgroups", 0)
    global512 = bool(workgroups and case.get("memory512"))
    ram = case["global_words"][:] if global512 else None
    scoped_ram = case.get("scoped_words")
    if scoped_ram is not None:
        scoped_ram = scoped_ram[:]
    group = 0
    wide = "initial_registers" in case
    initial = (
        [[0] * 8 for _ in range(4)]
        if interleaved
        else case.get("initial_registers", [[0] * 8, [0] * 8])
    )
    acc = [[word & 65535 for word in bank] for bank in initial]
    high = [[word >> 16 for word in bank] for bank in initial]
    indices = [[0] * 8 for _ in range(warp_count)]
    masks = [255] * warp_count
    branches: list[list[tuple[int, int]]] = [[] for _ in range(warp_count)]
    loops: list[list[int]] = [[] for _ in range(warp_count)]
    operands = bytearray(case["a"] + case["b"])
    mac_operands = alternate_operands if operand_bank else operands
    c = case["c"][:]
    schedules = []
    trace = []
    outcome = "step_limit"
    for index in range(max_steps + 1):
        if interleaved:
            dispatch_result = reference_dispatch(tuple(states), last)
            for start_warp in (0, 2):
                if states[start_warp : start_warp + 2] == [2, 2]:
                    states[start_warp : start_warp + 2] = [0, 0]
            warp = dispatch_result.warp
            ready = warp is not None
            if ready:
                group, _local = divmod(warp, 2)
                ticket = RESIDENT_TICKET + warp
            else:
                outcome = "halted" if dispatch_result.halted else "deadlock"
                ticket = DONE if dispatch_result.halted else DEADLOCK
        else:
            released = states == [2, 2]
            advance = states == [1, 1] and tile < last_tile
            next_group = states == [1, 1] and group + 1 < workgroups
            if next_group:
                group += 1
                pcs = [0, 0]
                acc = [[0] * 8 for _ in range(warp_count)]
                high = [[0] * 8 for _ in range(warp_count)]
                indices = [[0] * 8 for _ in range(warp_count)]
                masks = [255] * warp_count
                branches = [[], []]
                loops = [[], []]
            if advance:
                tile += 1
                pcs = [0, 0]
            if advance or released or next_group:
                states = [0, 0]
            ready = [warp for warp in range(2) if states[warp] == 0]
            if not ready:
                outcome = "halted" if states == [1, 1] else "deadlock"
                ticket = DONE if outcome == "halted" else DEADLOCK
            elif next_group:
                warp = 0
                ticket = WORKGROUP_TICKETS[group - 1]
            else:
                warp = 1 - last if 1 - last in ready else ready[0]
                ticket = (
                    TICKET
                    + 8 * tile
                    + warp
                    + 2 * int(released)
                    + ADVANCE * int(advance)
                )
        schedules.append(
            {
                "ticket": ticket,
                "controls": [
                    encode_control(w, pcs[w], states[w]) for w in range(warp_count)
                ],
            }
        )
        if not ready or index == max_steps:
            break
        last = warp
        pc = pcs[warp]
        local_warp = warp % 2
        op, arg = case["programs"][local_warp][pc]
        row = 2 * tile + local_warp
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
                (v + mac_operands[start + arg] * mac_operands[64 + 8 * arg + col])
                & 65535
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
        elif op == "MATRIX_MAC8":
            k_size, n_size, k = arg // 16 + 1, (arg // 4) % 4 + 1, arg % 4
            words = []
            for lane, (low, upper) in enumerate(zip(before, high[warp], strict=True)):
                value = low | upper << 16
                output_row, col = divmod(16 * group + 8 * local_warp + lane, n_size)
                if active >> lane & 1 and output_row < 8:
                    left = (
                        ram[output_row * k_size + k]
                        if global512
                        else int.from_bytes(
                            operands[
                                2 * (output_row * k_size + k) : 2
                                * (output_row * k_size + k)
                                + 2
                            ],
                            "little",
                        )
                    ) & 255
                    address = 32 + k * n_size + col
                    right = (
                        ram[address]
                        if global512
                        else int.from_bytes(
                            operands[2 * address : 2 * address + 2], "little"
                        )
                    ) & 255
                    value = (value + left * right) & 0xFFFFFFFF
                words.append(value)
            candidate = [value & 65535 for value in words]
            high[warp] = [value >> 16 for value in words]
        elif op in ("MASK_GLOBAL_LT", RUNTIME_EXTENT_OP):
            limit = case["extent"] if op == RUNTIME_EXTENT_OP else arg
            masks[warp] &= sum(
                int(16 * group + 8 * local_warp + lane < limit) << lane
                for lane in range(8)
            )
        elif op in GLOBAL_OPS and op != "MASK_LANES":
            if op in ID_OPS:
                words = {
                    "LANE_ID": list(range(8)),
                    "WORKGROUP_ID": [group] * 8,
                    "THREAD_ID": [8 * local_warp + lane for lane in range(8)],
                    "GLOBAL_ID": [
                        16 * group + 8 * local_warp + lane for lane in range(8)
                    ],
                }[op]
            else:
                prior = [
                    lo | hi << 16 for lo, hi in zip(before, high[warp], strict=True)
                ]
                if op == "WARP_SUM":
                    total = sum(
                        word for lane, word in enumerate(prior) if active >> lane & 1
                    )
                    words = [total & 0xFFFFFFFF] * 8
                else:
                    words = [(word << arg) & 0xFFFFFFFF for word in prior]
            candidate = [word & 65535 for word in words]
            high[warp] = [
                word >> 16 if active >> lane & 1 else high[warp][lane]
                for lane, word in enumerate(words)
            ]
        elif op in SCOPED_LOADS:
            candidate = [
                scoped_ram[physical_word(op, group, local_warp, lane, addr)]
                for lane, addr in enumerate(indices[warp])
            ]
        elif op in WIDE_INDEX_OPS:
            from operator import and_, lt, mul, or_, xor

            def signed(word):
                return word if word < 1 << 31 else word - (1 << 32)

            apply = {
                "MUL_INDEX32": mul,
                "SLTU_INDEX32": lt,
                "SLT_INDEX32": lambda a, b: signed(a) < signed(b),
                "AND_INDEX32": and_,
                "OR_INDEX32": or_,
                "XOR_INDEX32": xor,
            }[op]
            words = [
                apply(lo | hi << 16, ram[addr] | ram[addr ^ 1] << 16) & 0xFFFFFFFF
                for lo, hi, addr in zip(before, high[warp], indices[warp], strict=True)
            ]
            candidate = [word & 65535 for word in words]
            high[warp] = [
                word >> 16 if active >> lane & 1 else high[warp][lane]
                for lane, word in enumerate(words)
            ]
        elif op == "ADDIDX":
            indices[warp] = [
                (addr + arg) % (256 if global512 else INDEX_WORDS)
                if active >> lane & 1
                else addr
                for lane, addr in enumerate(indices[warp])
            ]
        elif op in ALU_OPS and op not in LOGIC_OPS:
            import operator

            def signed32(word):
                return word - (1 << 32) if word >= (1 << 31) else word

            fn = {
                "SHL32": operator.lshift,
                "SHR32": operator.rshift,
                "SAR32": lambda word, shift: signed32(word) >> shift,
                "MUL32I": operator.mul,
                "SUB32I": operator.sub,
                "SLTU32I": operator.lt,
                "SLT32I": lambda word, rhs: signed32(word) < rhs,
            }[op]
            words = [
                fn(lo | hi << 16, arg) & 0xFFFFFFFF
                for lo, hi in zip(before, high[warp], strict=True)
            ]
            candidate = [word & 65535 for word in words]
            high[warp] = [
                word >> 16 if active >> lane & 1 else high[warp][lane]
                for lane, word in enumerate(words)
            ]
        elif op == "MASK_LANES":
            masks[warp] &= (1 << arg) - 1
        elif op == "BARRIER":
            states[warp] = 2
        elif op == "SAVE_INDEX":
            indices[warp] = [
                v % (256 if global512 else INDEX_WORDS)
                if active >> lane & 1
                else indices[warp][lane]
                for lane, v in enumerate(acc[warp])
            ]
        elif op in ("LOAD_INDEX", LOAD_INDEX_HI, *ADDMEM32_OPS, *LOGIC_OPS):
            high_access = op in (LOAD_INDEX_HI, ADD_INDEX_HI) or op.endswith("_HI")
            addresses = [addr ^ arg ^ int(high_access) for addr in indices[warp]]
            loaded = [
                ram[addr]
                if global512
                else int.from_bytes(operands[2 * addr : 2 * addr + 2], "little")
                if addr < 64
                else c[addr - 64]
                for addr in addresses
            ]
            if op in LOGIC_OPS:
                from operator import and_, or_, xor

                fn = {"AND_INDEX": and_, "OR_INDEX": or_, "XOR_INDEX": xor}[op[:-3]]
                if high_access:
                    high[warp] = [
                        fn(v, rhs) if active >> lane & 1 else v
                        for lane, (v, rhs) in enumerate(
                            zip(high[warp], loaded, strict=True)
                        )
                    ]
                else:
                    candidate = [
                        fn(v, rhs) for v, rhs in zip(before, loaded, strict=True)
                    ]
            elif op == ADD_INDEX_LO:
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
                pcs[warp] = case["links"][local_warp][pc]
        elif op == "ENDWHILE":
            masks[warp] = loops[warp].pop()
            pcs[warp] = case["links"][local_warp][pc]
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
        if op in SCOPED_STORES:
            claimed = set()
            for lane, (addr, word) in enumerate(
                zip(indices[warp], acc[warp], strict=True)
            ):
                addr = physical_word(op, group, local_warp, lane, addr)
                if not active >> lane & 1 or addr in claimed:
                    continue
                claimed.add(addr)
                grants |= 1 << lane
                scoped_ram[addr] = word
                patch[4 * lane : 4 * lane + 4] = nibbles(
                    SCOPED_BASE + 64 * addr, _digits([word], 16)
                )
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
                if global512:
                    ram[addr] = word
                    base = WIDE_MEMORY.base + 64 * addr
                elif addr < 64:
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
                (value & 127) << (SNAPSHOT_BITS + 7 * lane)
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
        if global512:
            packed |= sum(
                (value >> 7) << (WIDE_MEMORY.extra_index_offset + lane)
                for lane, value in enumerate(indices[warp])
            )
            packed |= 1 << WIDE_MEMORY.mode_offset
            width = WIDE_MEMORY.dirty_offset // 4
        trace.append(
            {
                "warp": warp,
                **({"workgroup": group} if workgroups else {}),
                "control": encode_control(warp, pcs[warp], states[warp]),
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
        **({"scoped_words": scoped_ram} if scoped_ram is not None else {}),
        "matrix": [
            (ram[64:128] if global512 else c)[i : i + 8] for i in range(0, 64, 8)
        ],
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
                "shared_words": ram
                if global512
                else [
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
    resident_tilebanks: bool = False,
    resident_tilepipe: bool = False,
    resident_tilegrid: bool = False,
    resident_globalgrid: bool = False,
    resident_compute: bool = False,
    resident_memory512: bool = False,
    resident_rom32: bool = False,
    resident_alu32: bool = False,
    resident_scoped: bool = False,
    resident_interleaved: bool = False,
    full_integer: bool = False,
):
    if checkpoint is not None:
        layout = json.loads((checkpoint / "qwengpu9b.json").read_text())["layout"]
        if full_integer and layout != FULL_INTEGER_LAYOUT:
            raise ValueError("expected a full integer checkpoint")
        full_integer = layout == FULL_INTEGER_LAYOUT
        if resident_interleaved and layout not in (RESIDENT_LAYOUT, FULL_INTEGER_LAYOUT):
            raise ValueError("expected an interleaved checkpoint")
        resident_interleaved = layout in (RESIDENT_LAYOUT, FULL_INTEGER_LAYOUT)
        if resident_scoped and layout not in (SCOPED_LAYOUT, RESIDENT_LAYOUT, FULL_INTEGER_LAYOUT):
            raise ValueError("expected a merged scoped-memory checkpoint")
        resident_scoped = layout in (SCOPED_LAYOUT, RESIDENT_LAYOUT, FULL_INTEGER_LAYOUT)
        if resident_alu32 and layout not in ALU32_LAYOUTS:
            raise ValueError("expected a merged ALU32 checkpoint")
        resident_alu32 = layout in ALU32_LAYOUTS
        if resident_rom32 and layout not in ROM32_LAYOUTS:
            raise ValueError("expected a merged ROM32 checkpoint")
        resident_rom32 = layout in ROM32_LAYOUTS
    if checkpoint is not None and resident_add32:
        resident_tilepipe = layout in RESIDENT_TILEPIPE_LAYOUTS
        resident_globalgrid = layout in RESIDENT_GLOBALGRID_LAYOUTS
        resident_memory512 = layout in MEMORY512_LAYOUTS
        resident_compute = layout in (RESIDENT_COMPUTE_LAYOUT, *MEMORY512_LAYOUTS)
        resident_tilegrid = layout in (
            RESIDENT_TILEGRID_LAYOUT,
            *RESIDENT_GLOBALGRID_LAYOUTS,
        )
        resident_tilebanks = layout in RESIDENT_TILEBANKS_LAYOUTS
        resident_loop32 = layout in RESIDENT_LOOP32_LAYOUTS
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
            resident_tilebanks=resident_tilebanks,
            resident_tilepipe=resident_tilepipe,
            resident_tilegrid=resident_tilegrid,
            resident_globalgrid=resident_globalgrid,
            resident_compute=resident_compute,
            resident_memory512=resident_memory512,
            resident_rom32=resident_rom32,
            resident_alu32=resident_alu32,
            resident_scoped=resident_scoped,
            resident_interleaved=resident_interleaved,
            full_integer=full_integer,
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
            resident_tilebanks=resident_tilebanks,
            resident_tilepipe=resident_tilepipe,
            resident_tilegrid=resident_tilegrid,
            resident_globalgrid=resident_globalgrid,
            resident_compute=resident_compute,
            resident_memory512=resident_memory512,
            resident_rom32=resident_rom32,
            resident_alu32=resident_alu32,
            resident_scoped=resident_scoped,
            resident_interleaved=resident_interleaved,
            full_integer=full_integer,
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
    variants.add_argument("--resident-tilebanks", action="store_true")
    variants.add_argument("--resident-tilepipe", action="store_true")
    variants.add_argument("--resident-tilegrid", action="store_true")
    variants.add_argument("--resident-globalgrid", action="store_true")
    variants.add_argument("--resident-compute", action="store_true")
    variants.add_argument("--resident-memory512", action="store_true")
    variants.add_argument("--resident-rom32", action="store_true")
    variants.add_argument("--resident-alu32", action="store_true")
    variants.add_argument("--resident-scoped", action="store_true")
    variants.add_argument("--resident-interleaved", action="store_true")
    variants.add_argument("--full-integer", action="store_true")
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    meta = None
    full_integer = args.full_integer
    resident_interleaved = args.resident_interleaved or full_integer
    resident_scoped = args.resident_scoped or resident_interleaved
    resident_alu32 = args.resident_alu32 or resident_scoped
    resident_rom32 = args.resident_rom32 or resident_alu32
    resident_memory512 = args.resident_memory512 or resident_rom32
    resident_compute = args.resident_compute or resident_memory512
    resident_globalgrid = args.resident_globalgrid or resident_compute
    resident_tilegrid = args.resident_tilegrid or resident_globalgrid
    resident_tilepipe = args.resident_tilepipe or resident_tilegrid
    resident_tilebanks = args.resident_tilebanks or resident_tilepipe
    resident_loop32 = args.resident_loop32 or resident_tilebanks
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
        resident_tilepipe = layout in RESIDENT_TILEPIPE_LAYOUTS
        resident_interleaved = layout in (RESIDENT_LAYOUT, FULL_INTEGER_LAYOUT)
        full_integer = layout == FULL_INTEGER_LAYOUT
        resident_scoped = layout in (SCOPED_LAYOUT, RESIDENT_LAYOUT, FULL_INTEGER_LAYOUT)
        resident_alu32 = layout in ALU32_LAYOUTS
        resident_rom32 = layout in ROM32_LAYOUTS
        resident_memory512 = layout in MEMORY512_LAYOUTS
        resident_compute = layout in (RESIDENT_COMPUTE_LAYOUT, *MEMORY512_LAYOUTS)
        resident_globalgrid = layout in RESIDENT_GLOBALGRID_LAYOUTS
        resident_tilegrid = layout in (
            RESIDENT_TILEGRID_LAYOUT,
            *RESIDENT_GLOBALGRID_LAYOUTS,
        )
        resident_tilebanks = layout in RESIDENT_TILEBANKS_LAYOUTS
        resident_loop32 = layout in RESIDENT_LOOP32_LAYOUTS
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
        if args.resident_tilebanks and not resident_tilebanks:
            raise ValueError("--resident-tilebanks requires a tilebanks checkpoint")
        if args.resident_tilepipe and not resident_tilepipe:
            raise ValueError("--resident-tilepipe requires a tilepipe checkpoint")
        if args.resident_tilegrid and not resident_tilegrid:
            raise ValueError("--resident-tilegrid requires a tilegrid checkpoint")
        if args.resident_globalgrid and not resident_globalgrid:
            raise ValueError("--resident-globalgrid requires a globalgrid checkpoint")
        if args.resident_compute and not resident_compute:
            raise ValueError("--resident-compute requires a compute checkpoint")
        if args.resident_scoped and not resident_scoped:
            raise ValueError("--resident-scoped requires a scoped checkpoint")
        if args.resident_alu32 and not resident_alu32:
            raise ValueError("--resident-alu32 requires an ALU32 checkpoint")
        if args.resident_rom32 and not resident_rom32:
            raise ValueError("--resident-rom32 requires a ROM32 checkpoint")
        if args.resident_memory512 and not resident_memory512:
            raise ValueError("--resident-memory512 requires a memory512 checkpoint")
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
                    resident_memory512=resident_memory512,
                    resident_rom32=resident_rom32,
                    resident_alu32=resident_alu32,
                    resident_scoped=resident_scoped,
                    resident_interleaved=resident_interleaved,
                    full_integer=full_integer,
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
        resident_tilebanks=resident_tilebanks,
        resident_tilepipe=resident_tilepipe,
        resident_tilegrid=resident_tilegrid,
        resident_globalgrid=resident_globalgrid,
        resident_compute=resident_compute,
        resident_memory512=resident_memory512,
        resident_rom32=resident_rom32,
        resident_alu32=resident_alu32,
        resident_scoped=resident_scoped,
        resident_interleaved=resident_interleaved,
        full_integer=full_integer,
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
