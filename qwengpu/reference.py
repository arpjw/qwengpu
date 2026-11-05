"""Independent reference semantics for an eight-lane SIMT research machine.

All arithmetic is unsigned, modulo 256. A warp has one program counter and
eight lane-local accumulators. Structured IF/ELSE/ENDIF instructions execute
both sides under disjoint active masks before reconverging. The reference
interpreter is an oracle for the compiled-layer implementation, not its runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


LANES = 8
SCRATCH_SIZE = 64
Opcode = Literal["ADDI", "XORI", "IF_LT", "ELSE", "ENDIF", "STORE", "LOAD", "SUM", "HALT",
                 "WHILE_LT", "ENDWHILE", "MOV_R1", "GATHER_R1", "MUL_R1"]


@dataclass(frozen=True, slots=True)
class Instruction:
    op: Opcode
    arg: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.arg <= 255:
            raise ValueError("instruction argument must be an unsigned byte")


@dataclass(frozen=True, slots=True)
class BranchFrame:
    parent_mask: int
    then_mask: int
    has_else: bool = False


@dataclass(frozen=True, slots=True)
class WarpState:
    pc: int = 0
    active_mask: int = (1 << LANES) - 1
    lanes: tuple[int, ...] = (0,) * LANES
    scratch: bytes = field(default_factory=lambda: bytes(SCRATCH_SIZE))
    branch_stack: tuple[BranchFrame, ...] = ()
    scalar: int = 0
    halted: bool = False

    def __post_init__(self) -> None:
        if self.pc < 0 or self.active_mask & ~((1 << LANES) - 1):
            raise ValueError("invalid program counter or lane mask")
        if len(self.lanes) != LANES or any(not 0 <= x <= 255 for x in self.lanes):
            raise ValueError("expected eight unsigned lane accumulators")
        if len(self.scratch) != SCRATCH_SIZE or not 0 <= self.scalar <= 255:
            raise ValueError("invalid scratch memory or scalar")


def _matching(program: tuple[Instruction, ...], start: int, target: str) -> int:
    depth = 0
    for pc in range(start + 1, len(program)):
        op = program[pc].op
        if op == "IF_LT":
            depth += 1
        elif op == "ENDIF":
            if depth == 0:
                if target == "ENDIF":
                    return pc
                break
            depth -= 1
        elif op == "ELSE" and depth == 0 and target == "ELSE":
            return pc
    raise ValueError(f"unmatched {target} after instruction {start}")


def step(state: WarpState, program: tuple[Instruction, ...]) -> WarpState:
    """Execute one warp instruction, including masked instructions."""
    if state.halted:
        return state
    if state.pc >= len(program):
        raise ValueError(f"fetch outside program at {state.pc}")

    ins = program[state.pc]
    op, arg = ins.op, ins.arg
    lanes = list(state.lanes)
    scratch = bytearray(state.scratch)
    active = state.active_mask
    stack = list(state.branch_stack)
    scalar = state.scalar
    pc = state.pc + 1
    halted = False

    if op in ("ADDI", "XORI", "STORE", "LOAD"):
        if op in ("STORE", "LOAD") and arg + LANES > SCRATCH_SIZE:
            raise ValueError("lane addresses exceed scratch memory")
        for lane in range(LANES):
            if active >> lane & 1:
                if op == "ADDI":
                    lanes[lane] = (lanes[lane] + arg) & 255
                elif op == "XORI":
                    lanes[lane] ^= arg
                elif op == "STORE":
                    scratch[arg + lane] = lanes[lane]
                else:
                    lanes[lane] = scratch[arg + lane]
    elif op == "IF_LT":
        _matching(program, state.pc, "ENDIF")
        then_mask = sum(1 << lane for lane, value in enumerate(lanes)
                        if active >> lane & 1 and value < arg)
        stack.append(BranchFrame(active, then_mask))
        active = then_mask
    elif op == "ELSE":
        if not stack or stack[-1].has_else:
            raise ValueError("ELSE without a matching IF")
        frame = stack[-1]
        stack[-1] = BranchFrame(frame.parent_mask, frame.then_mask, True)
        active = frame.parent_mask & ~frame.then_mask
    elif op == "ENDIF":
        if not stack:
            raise ValueError("ENDIF without a matching IF")
        active = stack.pop().parent_mask
    elif op == "SUM":
        scalar = sum(lanes[lane] for lane in range(LANES) if active >> lane & 1) & 255
    elif op == "HALT":
        if stack:
            raise ValueError("HALT with unclosed branch")
        pc = state.pc
        halted = True
    else:
        raise ValueError(f"unknown opcode {op}")

    return WarpState(pc, active, tuple(lanes), bytes(scratch), tuple(stack), scalar, halted)


def run(program: tuple[Instruction, ...], *, state: WarpState | None = None,
        max_steps: int = 1000) -> WarpState:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    current = state or WarpState()
    for _ in range(max_steps):
        current = step(current, program)
        if current.halted:
            return current
    raise ValueError(f"program did not halt within {max_steps} instructions")
