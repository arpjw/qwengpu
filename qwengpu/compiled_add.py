"""Eight-lane masked addition compiled into Qwen3.5 circuit IR.

Inputs and outputs move through raw trace tokens. Sixteen Gated DeltaNet heads
hold eight A bytes and eight B bytes; another stores the active mask. One
EXEC_ADD token reads all lanes and computes their sums in parallel fixed
gates. A result head stores the entire resulting warp. READ(lane) emits an
OUT(byte) token for that lane. The host supplies inputs and read requests but
never computes an output byte.

This is an arithmetic circuit milestone, not yet a programmable GPU kernel.
"""

from __future__ import annotations

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target
from chipformer.q35.prims import Circ

from qwengpu.reference import LANES


LOAD_A_BASE = 0
LOAD_B_BASE = LOAD_A_BASE + LANES * 256
MASK_BASE = LOAD_B_BASE + LANES * 256
EXEC_ADD = MASK_BASE + 256
READ_BASE = EXEC_ADD + 1
OUT_BASE = ((READ_BASE + LANES + 255) // 256) * 256
OUT_END = OUT_BASE + LANES * 256
COMPARE_BASE = OUT_END
ELSE_TOKEN = COMPARE_BASE + 256
ENDIF_TOKEN = ELSE_TOKEN + 1
VOCAB = ENDIF_TOKEN + 1
CODE_BITS = (VOCAB - 1).bit_length()
CODE_AMP = 128
KAPPA = 2.0**-20


def load_a(lane: int, value: int) -> int:
    if not 0 <= lane < LANES or not 0 <= value <= 255:
        raise ValueError("invalid lane or byte")
    return LOAD_A_BASE + lane * 256 + value


def load_b(lane: int, value: int) -> int:
    if not 0 <= lane < LANES or not 0 <= value <= 255:
        raise ValueError("invalid lane or byte")
    return LOAD_B_BASE + lane * 256 + value


def mask_token(mask: int) -> int:
    if not 0 <= mask < 1 << LANES:
        raise ValueError("invalid lane mask")
    return MASK_BASE + mask


def read_lane(lane: int) -> int:
    if not 0 <= lane < LANES:
        raise ValueError("invalid lane")
    return READ_BASE + lane


def out_token(lane: int, value: int) -> int:
    if not 0 <= lane < LANES or not 0 <= value <= 255:
        raise ValueError("invalid lane or byte")
    return OUT_BASE + lane * 256 + value


def compare_token(threshold: int) -> int:
    if not 0 <= threshold <= 255:
        raise ValueError("invalid threshold")
    return COMPARE_BASE + threshold


def _add8(c: Circ, left: list[Expr], right: list[Expr], name: str) -> list[Expr]:
    carry = Expr()
    result = []
    for bit, (a, b) in enumerate(zip(left, right, strict=True)):
        ab = c.and_([a, b], f"{name}.ab{bit}")
        pair = a + b - 2 * ab
        pc = c.and_([pair, carry], f"{name}.pc{bit}") if bit else Expr()
        result.append(pair + carry - 2 * pc)
        carry = ab + pc
    return result


def _lt8(c: Circ, left: list[Expr], right: list[Expr], name: str) -> Expr:
    """Unsigned less-than on bit vectors, most significant bit first."""
    equal = c.one
    less = Expr()
    for bit in reversed(range(8)):
        a, b = left[bit], right[bit]
        less += c.and_([equal, c.one - a, b], f"{name}.less{bit}")
        both = c.and_([a, b], f"{name}.both{bit}")
        equal = c.and_([equal, c.one - a - b + 2 * both], f"{name}.eq{bit}")
    return less


class AddCore(Circ):
    def __init__(self, target: Target | None = None) -> None:
        super().__init__(target or Target())
        f = self.token_feature
        self.is_exec = f("is_exec", lambda t: int(t == EXEC_ADD), full=True)
        self.is_compare = f("is_compare",
                            lambda t: int(COMPARE_BASE <= t < ELSE_TOKEN), full=True)
        self.is_else = f("is_else", lambda t: int(t == ELSE_TOKEN), full=True)
        self.is_endif = f("is_endif", lambda t: int(t == ENDIF_TOKEN), full=True)
        self.is_read = f("is_read", lambda t: int(READ_BASE <= t < READ_BASE + LANES),
                         full=True)
        self.is_a = [f(f"is_a.{lane}",
                       lambda t, lane=lane: int(LOAD_A_BASE + lane * 256 <= t
                                                < LOAD_A_BASE + (lane + 1) * 256), full=True)
                     for lane in range(LANES)]
        self.is_b = [f(f"is_b.{lane}",
                       lambda t, lane=lane: int(LOAD_B_BASE + lane * 256 <= t
                                                < LOAD_B_BASE + (lane + 1) * 256), full=True)
                     for lane in range(LANES)]
        self.is_out_lane = [f(f"is_out.{lane}",
                              lambda t, lane=lane: int(OUT_BASE + lane * 256 <= t
                                                       < OUT_BASE + (lane + 1) * 256),
                              full=True) for lane in range(LANES)]
        self.is_mask = f("is_mask", lambda t: int(MASK_BASE <= t < EXEC_ADD), full=True)
        self.is_read_lane = [f(f"is_read.{lane}",
                               lambda t, lane=lane: int(t == READ_BASE + lane), full=True)
                             for lane in range(LANES)]
        self.raw_byte = [f(f"raw.b{bit}", lambda t, bit=bit: (t >> bit) & 1, full=True)
                         for bit in range(8)]
        self.raw_mask = [f(f"mask.b{bit}", lambda t, bit=bit: ((t - MASK_BASE) >> bit) & 1
                           if MASK_BASE <= t < EXEC_ADD else 0, full=True)
                         for bit in range(LANES)]
        self.code = [f(f"code.b{bit}", lambda t, bit=bit:
                       CODE_AMP * (2 * ((t >> bit) & 1) - 1), full=True)
                     for bit in range(CODE_BITS)]
        self.keep = {next(iter(expr.terms)) for expr in self.code}


def build(target: Target | None = None) -> AddCore:
    c = AddCore(target)
    one = c.one
    for code in c.code:
        c.ffn([(-1, one, code), (1, one, -code)], "code.clear", into=code)

    a_lanes = []
    b_lanes = []
    for lane in range(LANES):
        a_write = c.is_a[lane] + c.is_out_lane[lane]
        a_lanes.append(c.slot_head(f"a.{lane}", {0: a_write},
                                   {0: c.is_exec + c.is_compare},
                                   {2 + bit: val for bit, val in enumerate(c.raw_byte)},
                                   a_write))
        b_lanes.append(c.slot_head(f"b.{lane}", {0: c.is_b[lane]}, {0: c.is_exec},
                                   {2 + bit: val for bit, val in enumerate(c.raw_byte)},
                                   c.is_b[lane]))
    mask = c.slot_head("active_mask", {0: c.is_mask},
                       {0: c.is_exec + c.is_compare + c.is_else},
                       {2 + bit: val for bit, val in enumerate(c.raw_mask)}, c.is_mask)
    parent = c.slot_head("parent_mask", {0: c.is_compare},
                          {0: c.is_else + c.is_endif},
                          {2 + bit: val for bit, val in enumerate(mask)}, c.is_compare)

    less_mask = [_lt8(c, a_lanes[lane], c.raw_byte, f"compare.{lane}")
                 for lane in range(LANES)]
    else_mask = [c.and_([parent[lane], one - mask[lane]], f"else.{lane}")
                 for lane in range(LANES)]

    result = []
    for lane in range(LANES):
        summed = _add8(c, a_lanes[lane], b_lanes[lane], f"sum.{lane}")
        result.extend(c.select_bits([([mask[lane]], summed),
                                     ([one - mask[lane]], a_lanes[lane])],
                                    8, f"selected.{lane}"))

    saved = c.slot_head("result", {0: c.is_exec}, {0: c.is_read},
                        {2 + bit: value for bit, value in enumerate(result)}, c.is_exec)
    selected = [sum((c.and_([c.is_read_lane[lane], saved[8 * lane + bit]],
                             f"output.{lane}.{bit}") for lane in range(LANES)), Expr())
                for bit in range(8)]

    floor = max(node.level for node in c.nodes)
    for bit, code in enumerate(c.code):
        read_bit = (selected[bit] if bit < 8 else
                    sum((c.is_read_lane[lane] for lane in range(LANES)
                         if out_token(lane, 0) >> bit & 1), Expr()))
        compare_bit = (less_mask[bit] if bit < 8
                       else (one if MASK_BASE >> bit & 1 else Expr()))
        else_bit = (else_mask[bit] if bit < 8
                    else (one if MASK_BASE >> bit & 1 else Expr()))
        endif_bit = (parent[bit] if bit < 8
                     else (one if MASK_BASE >> bit & 1 else Expr()))
        terms = (c.t_const(-KAPPA)
                 + c.t_and([c.is_read, read_bit], 2 * KAPPA)
                 + c.t_and([c.is_compare, compare_bit], 2 * KAPPA)
                 + c.t_and([c.is_else, else_bit], 2 * KAPPA)
                 + c.t_and([c.is_endif, endif_bit], 2 * KAPPA))
        c.ffn(terms, f"exit.code{bit}", into=code, min_level=floor)
    return c


def build_standalone(*, dtype=None):
    """Build a stock text-only Qwen3.5 model containing only this circuit."""
    import torch

    from chipformer.q35.tiny import build as build_qwen

    circuit = build()
    code_dims = [next(iter(expr.terms)) for expr in circuit.code]
    model = build_qwen(circuit, list(range(VOCAB)), code_dims,
                       dtype=torch.bfloat16 if dtype is None else dtype)
    return model, circuit
