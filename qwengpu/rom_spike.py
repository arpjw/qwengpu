"""Sixteen-slot recurrent instruction ROM and model-held program counter.

This isolated spike tests reusable instruction storage and fetch. It does not
drive the SIMT ALU yet. A model-generated instruction and next-PC token are
fed back after TICK and ADVANCE, respectively.
"""

from __future__ import annotations

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target
from chipformer.q35.prims import Circ


ROM_SLOTS = 16
PAYLOAD_BITS = 11
STATE_END = ROM_SLOTS
TICK = STATE_END
ADVANCE = TICK + 1
PAD = ADVANCE + 1
LOAD_BASE = 32
OUT_BASE = 1 << 16
VOCAB = OUT_BASE + (1 << PAYLOAD_BITS)
CODE_BITS = (VOCAB - 1).bit_length()
CODE_AMP = 128
KAPPA = 2.0**-20
SAMPLE_PROGRAMS = (
    tuple(((37 * slot + 0x101) & 0x7FF) for slot in range(ROM_SLOTS)),
    tuple(((191 * slot + 0x4AA) & 0x7FF) for slot in range(ROM_SLOTS)),
    tuple(0x7FF if slot == ROM_SLOTS - 1 else
          ((53 * slot + 0x300) & 0x7FF) for slot in range(ROM_SLOTS)),
)


def load_token(address: int, payload: int) -> int:
    if not 0 <= address < ROM_SLOTS or not 0 <= payload < 1 << PAYLOAD_BITS:
        raise ValueError("invalid ROM address or instruction payload")
    return LOAD_BASE + address * (1 << PAYLOAD_BITS) + payload


def instruction_token(payload: int) -> int:
    if not 0 <= payload < 1 << PAYLOAD_BITS:
        raise ValueError("invalid instruction payload")
    return OUT_BASE + payload


class RomCore(Circ):
    def __init__(self, target: Target | None = None) -> None:
        super().__init__(target or Target())
        f = self.token_feature
        self.is_state = f("is_state", lambda t: int(0 <= t < STATE_END), full=True)
        self.is_tick = f("is_tick", lambda t: int(t == TICK), full=True)
        self.is_advance = f("is_advance", lambda t: int(t == ADVANCE), full=True)
        self.is_load = f("is_load", lambda t: int(LOAD_BASE <= t <
                                                  LOAD_BASE + ROM_SLOTS * (1 << PAYLOAD_BITS)),
                         full=True)
        self.is_load_address = [
            f(f"is_load.{addr}",
              lambda t, addr=addr: int(LOAD_BASE + addr * (1 << PAYLOAD_BITS) <= t <
                                      LOAD_BASE + (addr + 1) * (1 << PAYLOAD_BITS)),
              full=True) for addr in range(ROM_SLOTS)]
        self.raw_pc = [f(f"pc.b{bit}", lambda t, bit=bit: (t >> bit) & 1
                         if 0 <= t < STATE_END else 0, full=True)
                       for bit in range(4)]
        self.raw_payload = [
            f(f"payload.b{bit}", lambda t, bit=bit: ((t - LOAD_BASE) >> bit) & 1
              if LOAD_BASE <= t < LOAD_BASE + ROM_SLOTS * (1 << PAYLOAD_BITS) else 0,
              full=True) for bit in range(PAYLOAD_BITS)]
        self.code = [f(f"code.b{bit}", lambda t, bit=bit:
                       CODE_AMP * (2 * ((t >> bit) & 1) - 1), full=True)
                     for bit in range(CODE_BITS)]
        self.keep = {next(iter(expr.terms)) for expr in self.code}


def build(target: Target | None = None) -> RomCore:
    c = RomCore(target)
    one = c.one
    for code in c.code:
        c.ffn([(-1, one, code), (1, one, -code)], "code.clear", into=code)

    pc = c.slot_head("pc", {0: c.is_state}, {0: c.is_tick + c.is_advance},
                     {2 + bit: value for bit, value in enumerate(c.raw_pc)},
                     c.is_state)
    pc_onehot = c.onehot_of_bits(pc, "pc.onehot", ROM_SLOTS)
    rom_queries = {addr: c.and_([c.is_tick, pc_onehot[addr]], f"rom.query.{addr}")
                   for addr in range(ROM_SLOTS)}
    payload = c.slot_head("rom", dict(enumerate(c.is_load_address)), rom_queries,
                          {2 + bit: value for bit, value in enumerate(c.raw_payload)},
                          c.is_load)
    next_pc = [one - pc[0]]
    for bit in range(1, 4):
        carry = c.and_(pc[:bit], f"pc.carry.{bit}")
        both = c.and_([pc[bit], carry], f"pc.both.{bit}")
        next_pc.append(pc[bit] + carry - 2 * both)

    floor = max(node.level for node in c.nodes)
    for bit, code in enumerate(c.code):
        tick_bit = (payload[bit] if bit < PAYLOAD_BITS
                    else one if OUT_BASE >> bit & 1 else Expr())
        advance_bit = next_pc[bit] if bit < 4 else Expr()
        terms = (c.t_const(-KAPPA)
                 + c.t_and([c.is_tick, tick_bit], 2 * KAPPA)
                 + c.t_and([c.is_advance, advance_bit], 2 * KAPPA))
        c.ffn(terms, f"exit.code{bit}", into=code, min_level=floor)
    return c


def build_standalone(*, dtype=None):
    import torch

    from chipformer.q35.tiny import build as build_qwen

    circuit = build()
    code_dims = [next(iter(expr.terms)) for expr in circuit.code]
    model = build_qwen(circuit, list(range(VOCAB)), code_dims,
                       dtype=torch.bfloat16 if dtype is None else dtype)
    return model, circuit
