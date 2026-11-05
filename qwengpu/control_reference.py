"""Independent bounded structured SIMT semantics with data-dependent loops."""

from __future__ import annotations
from dataclasses import dataclass
from qwengpu.reference import Instruction, WarpState, step as base_step

MAX_STEPS = 16
MAX_DEPTH = 2
OPS = (
    "ADDI",
    "XORI",
    "STORE",
    "LOAD",
    "HALT",
    "IF_LT",
    "ELSE",
    "ENDIF",
    "WHILE_LT",
    "ENDWHILE",
)


def validate(program: tuple[Instruction, ...]) -> dict[int, int]:
    if not 1 <= len(program) <= 16 or program[-1].op != "HALT":
        raise ValueError("control ROM requires final HALT within 16 slots")
    stack: list[tuple[str, int, bool]] = []
    links: dict[int, int] = {}
    for pc, ins in enumerate(program):
        if ins.op not in OPS:
            raise ValueError(f"unsupported control opcode {ins.op}")
        if ins.op in ("STORE", "LOAD") and ins.arg not in (0, 8, 16, 24):
            raise ValueError("scratch addresses must be 0, 8, 16, or 24")
        if ins.op in ("ELSE", "ENDIF", "ENDWHILE", "HALT") and ins.arg:
            raise ValueError(f"{ins.op} takes no argument")
        if ins.op in ("IF_LT", "WHILE_LT"):
            if sum(kind == ins.op for kind, _, _ in stack) >= MAX_DEPTH:
                raise ValueError("maximum nesting depth is two per control stack")
            stack.append((ins.op, pc, False))
        elif ins.op == "ELSE":
            if not stack or stack[-1][0] != "IF_LT" or stack[-1][2]:
                raise ValueError("ELSE must occur once inside its IF_LT")
            kind, begin, _ = stack[-1]
            stack[-1] = (kind, begin, True)
        elif ins.op in ("ENDIF", "ENDWHILE"):
            expected = "IF_LT" if ins.op == "ENDIF" else "WHILE_LT"
            if not stack or stack[-1][0] != expected:
                raise ValueError("crossed or unmatched control blocks")
            _, begin, _ = stack.pop()
            if ins.op == "ENDWHILE":
                links[begin] = pc + 1
                links[pc] = begin
        elif ins.op == "HALT" and (stack or pc != len(program) - 1):
            raise ValueError("HALT must be last and outside all control blocks")
    if stack:
        raise ValueError("unclosed control block")
    return links


def parse_kernel(data: object) -> tuple[tuple[int, ...], tuple[Instruction, ...]]:
    if not isinstance(data, dict):
        raise ValueError("kernel must be an object")
    initial = data.get("initial")
    raw = data.get("program")
    if (
        not isinstance(initial, list)
        or len(initial) != 8
        or any(type(v) is not int or not 0 <= v <= 255 for v in initial)
    ):
        raise ValueError("initial must contain eight unsigned bytes")
    if not isinstance(raw, list) or not 1 <= len(raw) <= 16:
        raise ValueError("program must contain 1 through 16 instructions")
    program = []
    for ins in raw:
        if not isinstance(ins, dict) or ins.get("op") not in OPS:
            raise ValueError("invalid opcode")
        arg = ins.get("arg", 0)
        if type(arg) is not int or not 0 <= arg <= 255:
            raise ValueError("argument must be an unsigned byte")
        program.append(Instruction(ins["op"], arg))
    result = tuple(program)
    validate(result)
    return tuple(initial), result


@dataclass(frozen=True)
class LoopFrame:
    parent_mask: int
    begin: int
    end: int


@dataclass(frozen=True)
class ControlState:
    warp: WarpState
    loops: tuple[LoopFrame, ...] = ()


def step(state: ControlState, program: tuple[Instruction, ...]) -> ControlState:
    w = state.warp
    if w.halted:
        return state
    ins = program[w.pc]
    if ins.op == "WHILE_LT":
        # Resolve the matching end independently of uploaded ROM link tokens.
        depth = 0
        end = None
        for pc in range(w.pc + 1, len(program)):
            if program[pc].op == "WHILE_LT":
                depth += 1
            elif program[pc].op == "ENDWHILE":
                if depth == 0:
                    end = pc
                    break
                depth -= 1
        if end is None:
            raise ValueError("unmatched WHILE_LT")
        active = sum(
            1 << lane
            for lane, value in enumerate(w.lanes)
            if w.active_mask >> lane & 1 and value < ins.arg
        )
        if active:
            loops = state.loops + (LoopFrame(w.active_mask, w.pc, end),)
            pc = w.pc + 1
        else:
            loops = state.loops
            active = w.active_mask
            pc = end + 1
        return ControlState(
            WarpState(pc, active, w.lanes, w.scratch, w.branch_stack), loops
        )
    if ins.op == "ENDWHILE":
        if not state.loops or state.loops[-1].end != w.pc:
            raise ValueError("ENDWHILE without active loop")
        frame = state.loops[-1]
        return ControlState(
            WarpState(
                frame.begin, frame.parent_mask, w.lanes, w.scratch, w.branch_stack
            ),
            state.loops[:-1],
        )
    return ControlState(base_step(w, program), state.loops)


def snapshot_bits(state: ControlState) -> list[int]:
    w = state.warp
    fields = [(value, 8) for value in (*w.lanes, *w.scratch[:32])]
    fields.extend([(w.active_mask, 8), (len(w.branch_stack), 2)])
    for i in range(2):
        frame = w.branch_stack[i] if i < len(w.branch_stack) else None
        fields.extend(
            [
                (frame.parent_mask if frame else 0, 8),
                (frame.then_mask if frame else 0, 8),
            ]
        )
    fields.append((len(state.loops), 2))
    for i in range(2):
        fields.append((state.loops[i].parent_mask if i < len(state.loops) else 0, 8))
    return [(value >> bit) & 1 for value, width in fields for bit in range(width)]
