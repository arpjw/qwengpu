"""Stock-generation token protocol for resident workgroups and tiled launches."""

from chipformer.dsl.graph import Expr
from chipformer.q35.prims import from_bits, onehot_bits

LAYOUT = "qwengpu-resident-generation-9b-v3"
SCHEDULE = 2704
TICKET = 2708
COMMIT = 2712
GROUP = 2720
RESET = (2736, 2752)
ROW = 2760
PANEL = 2784
STEP_CONTROL = 0x9800
RELEASE_CONTROL = 0x9A00
REGISTER = 0x1C000
REGISTER_STRIDE = 0x800


def first_after(c, request, continuation, cursor, choices, name):
    """Select the first set position past the preceding token's cursor."""
    candidates = []
    for index, choice in enumerate(choices):
        later = request - continuation + c.node(
            c.t_lt(cursor, index, 0, 127, lits=[continuation]),
            f"{name}.after.{index}",
        )
        candidates.append(c.and_([later, choice], f"{name}.candidate.{index}"))
    groups = [candidates[i:i + 8] for i in range(0, len(candidates), 8)]
    occupied = [
        c.node(c.t_ge(sum(group, Expr()), 1, 0, len(group)), f"{name}.occupied.{i}")
        for i, group in enumerate(groups[:-1])
    ]
    selected = []
    for group_index, group in enumerate(groups):
        prior = sum(occupied[:group_index], Expr())
        for index, candidate in enumerate(group):
            selected.append(c.node(
                c.t_eq(prior, 0, 0, group_index + index, lits=[candidate]),
                f"{name}.first.{len(selected)}",
            ))
            prior += candidate
    return selected


def generated_schedule(
    c, schedule, release_inputs, controls, releases, issue, halt, deadlock,
    done_token, deadlock_token, advance_group, next_group, advance_row, next_row,
    panel_transition=None,
):
    """Save the scheduling decision before any barrier state is rewritten."""
    ready = [
        [*ctrl[:6], c.and_([ctrl[6], c.one - releases[warp // 2]],
                          f"generate.schedule.ready.{warp}")]
        for warp, ctrl in enumerate(controls)
    ]
    changed = c.node(c.t_ge(sum(releases, Expr()), 1, 0, 2), "generate.schedule.release")
    scheduled = onehot_bits(issue, 2)
    terminal = [halt, deadlock]
    stored = c.slot_head(
        "generate.schedule.snapshot", {0: schedule}, {0: sum(release_inputs, Expr())},
        {2 + i: bit for i, bit in enumerate(sum(ready, []) + scheduled + terminal)},
        schedule,
    )
    yield [schedule, changed], [*ready[0], *token_bits(RELEASE_CONTROL)[7:]]
    for warp in range(3):
        token = RELEASE_CONTROL + 128 * (warp + 1)
        yield [release_inputs[warp]], [*stored[7 * (warp + 1):7 * (warp + 2)], *token_bits(token)[7:]]
    advance_panel = Expr() if panel_transition is None else panel_transition[0]
    for gate, bits, halted, blocked in (
        ([schedule, c.one - changed, c.one - advance_group - advance_row - advance_panel],
         scheduled, halt, deadlock),
        ([release_inputs[-1]], stored[28:30], *stored[30:32]),
    ):
        yield [*gate, halted], token_bits(done_token)
        yield [*gate, blocked], token_bits(deadlock_token)
        yield [*gate, c.one - halted - blocked], [*bits, *token_bits(TICKET)[2:]]
    yield [schedule, advance_group], [*next_group, *token_bits(GROUP)[2:]]
    yield [schedule, advance_row], [*next_row, *token_bits(ROW)[2:]]
    if panel_transition is not None:
        yield [schedule, advance_panel], [*panel_transition[1], *token_bits(PANEL)[3:]]


def generated_writeback(
    c, request, reg_input, commit_inputs, raw, bank_bits, values, dirty, grants,
):
    """Drain changed register nibbles, then commit stores in lane order."""
    cursor = from_bits(raw[4:11])
    registers = first_after(c, request, reg_input, cursor, dirty, "generate.register")
    value = [
        c.node(
            [term for index, select in enumerate(registers)
             for term in c.t_and([select, values[4 * index + bit]])],
            f"generate.value.{bit}",
        )
        for bit in range(4)
    ]
    any_register = sum(registers, Expr())
    yield [any_register], [
        *value, *onehot_bits(registers, 7), *bank_bits, *token_bits(REGISTER)[13:],
    ]

    commit = sum(commit_inputs, Expr())
    store_request = c.and_([request, c.one - any_register], "generate.store.request") + commit
    lane = sum((i * hit for i, hit in enumerate(commit_inputs)), Expr())
    stores = first_after(c, store_request, commit, lane, grants, "generate.store")
    yield [sum(stores, Expr())], [*onehot_bits(stores, 3), *token_bits(COMMIT)[3:]]
    yield [store_request, c.one - sum(stores, Expr())], token_bits(SCHEDULE)


def token_bits(token):
    return [(token >> i) & 1 for i in range(18)]
