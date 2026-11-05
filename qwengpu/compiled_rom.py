"""Straight-line eight-lane SIMT kernel fetched from model-state ROM.

The caller uploads a 16-slot kernel and lane bytes, then supplies only TICK
and lane READ tokens. The model fetches and executes each instruction, emits
the next PC/halt state, and generates lane result tokens for feedback.
"""

from __future__ import annotations

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Target
from chipformer.q35.prims import Circ

from qwengpu.compiled_add import _add8
from qwengpu.reference import Instruction, LANES


ROM_SLOTS = 16
SCRATCH_SLOTS = 4
LOAD_A_BASE = 0
READ_BASE = LANES * 256
OUT_BASE = 1 << 12
STATE_BASE = 1 << 13
TICK = STATE_BASE + 32
ROM_LOAD_BASE = 1 << 14
ROM_LOAD_END = ROM_LOAD_BASE + ROM_SLOTS * (1 << 11)
VOCAB = 1 << 16
CODE_BITS = (VOCAB - 1).bit_length()
CODE_AMP = 128
KAPPA = 2.0**-20
OP_ADD = 0
OP_XOR = 1
OP_STORE = 2
OP_LOAD = 3
OP_HALT = 4
OPCODES = {"ADDI": OP_ADD, "XORI": OP_XOR, "STORE": OP_STORE,
           "LOAD": OP_LOAD, "HALT": OP_HALT}


def load_a(lane: int, value: int) -> int:
    if not 0 <= lane < LANES or not 0 <= value <= 255:
        raise ValueError("invalid lane or byte")
    return LOAD_A_BASE + lane * 256 + value


def read_lane(lane: int) -> int:
    if not 0 <= lane < LANES:
        raise ValueError("invalid lane")
    return READ_BASE + lane


def out_token(lane: int, value: int) -> int:
    if not 0 <= lane < LANES or not 0 <= value <= 255:
        raise ValueError("invalid lane or byte")
    return OUT_BASE + lane * 256 + value


def state_token(pc: int, halted: bool = False) -> int:
    if not 0 <= pc < ROM_SLOTS:
        raise ValueError("invalid program counter")
    return STATE_BASE + (16 if halted else 0) + pc


def rom_load_token(address: int, payload: int) -> int:
    if not 0 <= address < ROM_SLOTS or not 0 <= payload < 1 << 11:
        raise ValueError("invalid ROM address or instruction payload")
    return ROM_LOAD_BASE + address * (1 << 11) + payload


def instruction_payload(ins: Instruction) -> int:
    if ins.op not in OPCODES:
        raise ValueError(f"ROM does not support {ins.op}")
    arg = ins.arg
    if ins.op in ("STORE", "LOAD"):
        if arg not in (0, 8, 16, 24):
            raise ValueError("compiled scratch addresses are 0, 8, 16, or 24")
        arg //= LANES
    if ins.op == "HALT" and arg:
        raise ValueError("HALT takes no argument")
    return OPCODES[ins.op] * 256 + arg


def program_tokens(program: tuple[Instruction, ...]) -> list[int]:
    if not 1 <= len(program) <= ROM_SLOTS or program[-1].op != "HALT":
        raise ValueError("ROM program must end with HALT within 16 slots")
    if any(ins.op == "HALT" for ins in program[:-1]):
        raise ValueError("HALT must be last")
    payloads = [instruction_payload(ins) for ins in program]
    payloads.extend([OP_HALT * 256] * (ROM_SLOTS - len(payloads)))
    return [rom_load_token(address, payload)
            for address, payload in enumerate(payloads)]


class RomSIMT(Circ):
    def __init__(self, target: Target | None = None) -> None:
        super().__init__(target or Target())
        f = self.token_feature
        self.is_tick = f("is_tick", lambda t: int(t == TICK), full=True)
        self.is_state = f("is_state", lambda t: int(STATE_BASE <= t < TICK), full=True)
        self.is_rom_load = f("is_rom_load",
                             lambda t: int(ROM_LOAD_BASE <= t < ROM_LOAD_END), full=True)
        self.is_rom_address = [
            f(f"is_rom.{address}",
              lambda t, address=address: int(
                  ROM_LOAD_BASE + address * (1 << 11) <= t <
                  ROM_LOAD_BASE + (address + 1) * (1 << 11)), full=True)
            for address in range(ROM_SLOTS)]
        self.is_a = [f(f"is_a.{lane}",
                       lambda t, lane=lane: int(LOAD_A_BASE + lane * 256 <= t <
                                                LOAD_A_BASE + (lane + 1) * 256),
                       full=True) for lane in range(LANES)]
        self.is_out = [f(f"is_out.{lane}",
                         lambda t, lane=lane: int(OUT_BASE + lane * 256 <= t <
                                                  OUT_BASE + (lane + 1) * 256),
                         full=True) for lane in range(LANES)]
        self.is_read = f("is_read", lambda t: int(READ_BASE <= t < READ_BASE + LANES),
                         full=True)
        self.is_read_lane = [f(f"is_read.{lane}",
                               lambda t, lane=lane: int(t == READ_BASE + lane),
                               full=True) for lane in range(LANES)]
        self.raw_byte = [f(f"raw.b{bit}", lambda t, bit=bit: (t >> bit) & 1,
                           full=True) for bit in range(8)]
        self.raw_state = [f(f"state.b{bit}", lambda t, bit=bit:
                            ((t - STATE_BASE) >> bit) & 1
                            if STATE_BASE <= t < TICK else 0, full=True)
                          for bit in range(5)]
        self.raw_rom = [f(f"rom.b{bit}", lambda t, bit=bit:
                          ((t - ROM_LOAD_BASE) >> bit) & 1
                          if ROM_LOAD_BASE <= t < ROM_LOAD_END else 0, full=True)
                        for bit in range(11)]
        self.code = [f(f"code.b{bit}", lambda t, bit=bit:
                       CODE_AMP * (2 * ((t >> bit) & 1) - 1), full=True)
                     for bit in range(CODE_BITS)]
        self.keep = {next(iter(expr.terms)) for expr in self.code}


def _increment4(c: RomSIMT, bits: list[Expr]) -> list[Expr]:
    result = [c.one - bits[0]]
    for bit in range(1, 4):
        carry = c.and_(bits[:bit], f"pc.carry.{bit}")
        both = c.and_([bits[bit], carry], f"pc.both.{bit}")
        result.append(bits[bit] + carry - 2 * both)
    return result


def build(target: Target | None = None) -> RomSIMT:
    c = RomSIMT(target)
    one = c.one
    for code in c.code:
        c.ffn([(-1, one, code), (1, one, -code)], "code.clear", into=code)

    lanes = []
    for lane in range(LANES):
        write = c.is_a[lane] + c.is_out[lane]
        lanes.append(c.slot_head(f"a.{lane}", {0: write}, {0: c.is_tick},
                                 {2 + bit: value for bit, value in enumerate(c.raw_byte)},
                                 write))
    pc = c.slot_head("pc", {0: c.is_state}, {0: c.is_tick},
                     {2 + bit: value for bit, value in enumerate(c.raw_state)},
                     c.is_state)
    pc_onehot = c.onehot_of_bits(pc[:4], "pc.onehot", ROM_SLOTS)
    rom_queries = {address: c.and_([c.is_tick, pc_onehot[address]],
                                  f"rom.query.{address}")
                   for address in range(ROM_SLOTS)}
    payload = c.slot_head("rom", dict(enumerate(c.is_rom_address)), rom_queries,
                          {2 + bit: value for bit, value in enumerate(c.raw_rom)},
                          c.is_rom_load)
    arg = payload[:8]
    opcode = payload[8:11]
    flags = {op: c.and_([opcode[bit] if op >> bit & 1 else one - opcode[bit]
                        for bit in range(3)], f"op.{op}")
             for op in OPCODES.values()}
    is_add, is_xor, is_store, is_load, is_halt = (
        c.and_([c.is_tick, flags[op]], f"do.{op}")
        for op in (OP_ADD, OP_XOR, OP_STORE, OP_LOAD, OP_HALT))

    slot_onehot = c.onehot_of_bits(arg[:2], "scratch.slot", SCRATCH_SLOTS)
    scratch = []
    for lane in range(LANES):
        keys = {slot: c.and_([is_store, slot_onehot[slot]],
                             f"scratch.write.{lane}.{slot}")
                for slot in range(SCRATCH_SLOTS)}
        queries = {slot: c.and_([is_load, slot_onehot[slot]],
                                f"scratch.read.{lane}.{slot}")
                   for slot in range(SCRATCH_SLOTS)}
        scratch.append(c.slot_head(
            f"scratch.{lane}", keys, queries,
            {2 + bit: value for bit, value in enumerate(lanes[lane])},
            is_store))

    result = []
    default = one - is_add - is_xor - is_load
    for lane in range(LANES):
        summed = _add8(c, lanes[lane], arg, f"sum.{lane}")
        xored = [lanes[lane][bit] + arg[bit]
                 - 2 * c.and_([lanes[lane][bit], arg[bit]], f"xor.{lane}.{bit}")
                 for bit in range(8)]
        chosen = c.select_bits([([is_add], summed), ([is_xor], xored),
                                ([is_load], scratch[lane]),
                                ([default], lanes[lane])], 8, f"result.{lane}")
        result.extend(chosen)
    saved = c.slot_head("result", {0: c.is_tick}, {0: c.is_read},
                        {2 + bit: value for bit, value in enumerate(result)},
                        c.is_tick)
    selected = [sum((c.and_([c.is_read_lane[lane], saved[8 * lane + bit]],
                             f"output.{lane}.{bit}") for lane in range(LANES)), Expr())
                for bit in range(8)]

    halt_next = pc[4] + is_halt - c.and_([pc[4], is_halt], "halt.or")
    advance = _increment4(c, pc[:4])
    next_pc = c.select_bits([([halt_next], pc[:4]),
                             ([one - halt_next], advance)], 4, "next.pc")
    floor = max(node.level for node in c.nodes)
    for bit, code in enumerate(c.code):
        read_bit = (selected[bit] if bit < 8 else
                    sum((c.is_read_lane[lane] for lane in range(LANES)
                         if out_token(lane, 0) >> bit & 1), Expr()))
        state_bit = (next_pc[bit] if bit < 4 else
                     halt_next if bit == 4 else
                     one if STATE_BASE >> bit & 1 else Expr())
        terms = (c.t_const(-KAPPA)
                 + c.t_and([c.is_read, read_bit], 2 * KAPPA)
                 + c.t_and([c.is_tick, state_bit], 2 * KAPPA))
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
