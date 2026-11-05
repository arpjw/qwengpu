"""Independent bounded structured SIMT semantics with data-dependent loops."""

from __future__ import annotations
from dataclasses import dataclass, replace
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


ISA_OPS = ("MOV_R1", "GATHER_R1", "MUL_R1")


def validate(
    program: tuple[Instruction, ...], *, extended: bool = False
) -> dict[int, int]:
    if not 1 <= len(program) <= 16 or program[-1].op != "HALT":
        raise ValueError("control ROM requires final HALT within 16 slots")
    supported = OPS + ISA_OPS if extended else OPS
    stack: list[tuple[str, int, bool]] = []
    links: dict[int, int] = {}
    for pc, ins in enumerate(program):
        if ins.op not in supported:
            raise ValueError(f"unsupported control opcode {ins.op}")
        if ins.op in ("STORE", "LOAD") and ins.arg not in (0, 8, 16, 24):
            raise ValueError("scratch addresses must be 0, 8, 16, or 24")
        if ins.op in ("ELSE", "ENDIF", "ENDWHILE", "HALT", *ISA_OPS) and ins.arg:
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


def parse_kernel(
    data: object, *, extended: bool = False
) -> tuple[tuple[int, ...], tuple[Instruction, ...]]:
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
    supported = OPS + ISA_OPS if extended else OPS
    program = []
    for ins in raw:
        if not isinstance(ins, dict) or ins.get("op") not in supported:
            raise ValueError("invalid opcode")
        arg = ins.get("arg", 0)
        if type(arg) is not int or not 0 <= arg <= 255:
            raise ValueError("argument must be an unsigned byte")
        program.append(Instruction(ins["op"], arg))
    result = tuple(program)
    validate(result, extended=extended)
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
    r1: tuple[int, ...] = (0,) * 8


def step(state: ControlState, program: tuple[Instruction, ...]) -> ControlState:
    w = state.warp
    if w.halted:
        return state
    ins = program[w.pc]
    if ins.op in ISA_OPS:
        lanes = list(w.lanes)
        r1 = list(state.r1)
        for lane in range(8):
            if not (w.active_mask >> lane) & 1:
                continue
            if ins.op == "MOV_R1":
                r1[lane] = lanes[lane]
            elif ins.op == "GATHER_R1":
                lanes[lane] = w.scratch[r1[lane] & 31]
            else:
                lanes[lane] = (lanes[lane] * r1[lane]) & 255
        return replace(
            state, warp=replace(w, pc=w.pc + 1, lanes=tuple(lanes)), r1=tuple(r1)
        )
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
        return replace(state, warp=replace(w, pc=pc, active_mask=active), loops=loops)
    if ins.op == "ENDWHILE":
        if not state.loops or state.loops[-1].end != w.pc:
            raise ValueError("ENDWHILE without active loop")
        frame = state.loops[-1]
        return replace(
            state,
            warp=replace(w, pc=frame.begin, active_mask=frame.parent_mask),
            loops=state.loops[:-1],
        )
    return replace(state, warp=base_step(w, program))


def snapshot_bits(state: ControlState, *, extended: bool = False) -> list[int]:
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
    if extended:
        fields.extend((value, 8) for value in state.r1)
    return [(value >> bit) & 1 for value, width in fields for bit in range(width)]
