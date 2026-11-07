"""Address uint8 matrix operands in the programmable global memory bank."""

from __future__ import annotations

from collections.abc import Callable

from chipformer.dsl.graph import Expr
from chipformer.q35.prims import Circ, from_bits

from qwengpu.compiled_tile8 import INDEX_OPERAND

Lookup = Callable[[str, int, list[int], list[Expr]], list[Expr]]


def operands(
    c: Circ,
    execute: Expr,
    arg: list[Expr],
    active: list[Expr],
    group: list[Expr],
    warp: Expr,
    lookup: Lookup,
) -> tuple[list[list[Expr]], list[list[Expr]]]:
    k_sizes = c.onehot_of_bits(arg[4:6], "matrix.k_size", 4)
    n_sizes = c.onehot_of_bits(arg[2:4], "matrix.n_size", 4)
    k_positions = c.onehot_of_bits(arg[:2], "matrix.k_position", 4)
    left, right = [], []
    for lane in range(8):
        global_id = 16 * from_bits(group) + 8 * warp + lane
        rows = [
            c.node(
                [
                    term
                    for n in range(1, 5)
                    for term in c.t_in(
                        global_id, row * n, (row + 1) * n, 0, 63, lits=[n_sizes[n - 1]]
                    )
                ],
                f"matrix.row.{lane}.{row}",
            )
            for row in range(8)
        ]
        columns = [
            c.node(
                [
                    term
                    for n in range(col + 1, 5)
                    for row in range(8)
                    for term in c.t_eq(
                        global_id, row * n + col, 0, 63, lits=[n_sizes[n - 1]]
                    )
                ],
                f"matrix.col.{lane}.{col}",
            )
            for col in range(4)
        ]
        left_queries = [
            c.node(
                [
                    term
                    for row in range(8)
                    for k in range(1, 5)
                    for pos in range(k)
                    if row * k + pos == address
                    for term in c.t_and(
                        [
                            execute,
                            active[lane],
                            rows[row],
                            k_sizes[k - 1],
                            k_positions[pos],
                        ]
                    )
                ],
                f"matrix.a.{lane}.{address}",
            )
            for address in range(32)
        ]
        right_queries = [
            c.node(
                [
                    term
                    for n in range(1, 5)
                    for col in range(n)
                    for pos in range(4)
                    if pos * n + col == address
                    for term in c.t_and(
                        [
                            execute,
                            active[lane],
                            columns[col],
                            n_sizes[n - 1],
                            k_positions[pos],
                        ]
                    )
                ],
                f"matrix.b.{lane}.{address}",
            )
            for address in range(16)
        ]
        for name, start, queries, destination in (
            ("a", 0, left_queries, left),
            ("b", 32, right_queries, right),
        ):
            destination.append(
                [
                    bit
                    for nib in range(2)
                    for bit in lookup(
                        f"matrix.{name}.{lane}.{nib}",
                        INDEX_OPERAND,
                        [
                            4 * (start + address) + nib
                            for address in range(len(queries))
                        ],
                        queries,
                    )
                ]
            )
    return left, right
