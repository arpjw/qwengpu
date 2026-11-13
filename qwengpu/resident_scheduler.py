"""Round-robin scheduling for two resident groups of two warps."""

from __future__ import annotations

from dataclasses import dataclass

from chipformer.q35.prims import Circ


@dataclass(frozen=True)
class Dispatch:
    warp: int | None
    halted: bool = False


def reference_dispatch(statuses: tuple[int, int, int, int], last: int) -> Dispatch:
    """Statuses are ready (0), halted (1), and waiting at a barrier (2)."""
    for distance in range(1, 5):
        warp = (last + distance) % 4
        if statuses[warp] == 0 or statuses[warp] == statuses[warp ^ 1] == 2:
            return Dispatch(warp)
    return Dispatch(None, all(status == 1 for status in statuses))


def dispatch(c: Circ, halted, waiting, last):
    """Return one-hot issue gates, per-group release flags, halt, and deadlock."""
    one = c.one
    release = [
        c.and_(waiting[2 * group : 2 * group + 2], f"resident.release.{group}")
        for group in range(2)
    ]
    ready = [
        one - halted[warp] - waiting[warp] + release[warp // 2] for warp in range(4)
    ]
    cursor = c.onehot_of_bits(last, "resident.previous", 4)
    issue = []
    for warp in range(4):
        preceding = [(warp - distance) % 4 for distance in (1, 2, 3)]
        terms = [
            term
            for distance in range(4)
            for term in c.t_and(
                [
                    cursor[(warp - 1 - distance) % 4],
                    ready[warp],
                    *(one - ready[p] for p in preceding[:distance]),
                ]
            )
        ]
        issue.append(c.node(terms, f"resident.issue.{warp}"))
    done = c.and_(halted, "resident.done")
    idle = c.and_([one - gate for gate in ready], "resident.idle")
    return issue, release, done, idle - done
