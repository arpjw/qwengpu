"""Qualify model-generated IDs and a retained-cache cross-workgroup reduction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers

from qwengpu.assemble import assemble
from qwengpu.alu32 import ALU_VOCAB
from qwengpu.scoped_memory import SCOPED_BASE, SCOPED_END
from qwengpu.compiled_tile8 import (
    ROM32_VOCAB,
    C_MEMORY,
    INDEX_OPERAND,
    RESET_WARP,
    RESIDENT_COMPUTE_LAYOUT,
    RESIDENT_GLOBALGRID_LAYOUT,
    TILEGRID_VOCAB,
    WORKGROUP_TICKETS,
)
from qwengpu.eval_tile import _unsigned
from qwengpu.eval_tile8 import (
    RESIDENT_TRANSPORTS,
    execute,
    load_tile_model,
    oracle,
    parse_launch,
)
from qwengpu.memory_geometry import (
    ALU32_LAYOUTS,
    MEMORY512_LAYOUTS,
    ROM32_LAYOUTS,
    WIDE_MEMORY,
)

FIXTURE = Path(__file__).resolve().parents[1] / "eval/frozen/globalgrid_v1/words.json"
MASK32 = 0xFFFFFFFF


def make_launch(
    words: list[int], workgroups: int, warp0: str, warp1: str | None = None
) -> dict:
    source = f".warp 0\n{warp0}\n.warp 1\n{warp1 if warp1 is not None else warp0}\n"
    return assemble(source, {"workgroups": workgroups, "global_words": words})


@torch.no_grad()
def launch(
    model,
    circuit,
    data: dict,
    device: str,
    offset: int,
    cache=None,
    *,
    transport_mode: str = "state_store_plan",
    refresh_global_prefix: int = 0,
) -> tuple:
    case = parse_launch(
        data,
        resident_globalgrid=True,
        resident_compute=circuit.resident_compute,
        resident_memory512=circuit.resident_memory512,
        resident_rom32=circuit.resident_rom32,
        resident_alu32=circuit.resident_alu32,
        resident_scoped=circuit.resident_scoped,
        resident_interleaved=circuit.resident_interleaved,
        full_integer=circuit.full_integer,
    )
    expected = oracle(case, 256, looping=True)
    first = cache is None
    refresh_tokens = 4 * refresh_global_prefix
    audit = []
    scoped = "scoped_words" in case

    def observe(_model, _args, kwargs, result) -> None:
        nonlocal cache
        if kwargs["past_key_values"] is not cache or (
            cache is not None and result.past_key_values is not cache
        ):
            raise ValueError("model call changed the resident cache")
        cache = result.past_key_values
        tokens = [token - offset for token in kwargs["input_ids"][0].tolist()]
        audit.append(
            {
                **(
                    {
                        "global_refresh_tags": [
                            token for token in tokens
                            if circuit.memory_geometry.base <= token < circuit.memory_geometry.vocab
                        ]
                    }
                    if refresh_global_prefix and not audit
                    else {}
                ),
                "memory": sum(
                    INDEX_OPERAND <= token < INDEX_OPERAND + 4096
                    or C_MEMORY <= token < C_MEMORY + 4096
                    or (
                        circuit.resident_memory512
                        and circuit.memory_geometry.base
                        <= token
                        < circuit.memory_geometry.vocab
                    )
                    for token in tokens
                ),
                **(
                    {
                        "scoped": sum(
                            SCOPED_BASE <= token < SCOPED_END for token in tokens
                        )
                    }
                    if scoped
                    else {}
                ),
                "transitions": [
                    token for token in tokens if token in WORKGROUP_TICKETS
                ],
                "resets": [token for token in tokens if token in RESET_WARP],
            }
        )

    handle = model.register_forward_hook(observe, with_kwargs=True)
    try:
        actual = execute(
            model,
            circuit,
            case,
            device,
            256,
            offset,
            resident_add32=True,
            resident_cache=cache,
            refresh_global_prefix=refresh_global_prefix,
            **RESIDENT_TRANSPORTS[transport_mode],
        )
    finally:
        handle.remove()
    for key, value in expected.items():
        if actual[key] != value:
            if key == "trace":
                mismatch = next(
                    (
                        (i, a, b)
                        for i, (a, b) in enumerate(zip(actual[key], value))
                        if a != b
                    ),
                    (len(actual[key]), len(value)),
                )
                raise ValueError(f"globalgrid trace differs from reference: {mismatch}")
            raise ValueError(f"globalgrid {key} differs from reference")
    if scoped:
        transport = actual["transport"]
        if (
            not transport["resident_scoped_exact"]
            or audit[0]["scoped"] != transport["scoped_initial_tokens"]
            or sum(call["scoped"] for call in audit)
            != transport["scoped_initial_tokens"] + transport["scoped_patch_tokens"]
        ):
            raise ValueError("scoped cache, initialization, or write audit failed")
    if refresh_global_prefix:
        expected_refresh = [
            circuit.memory_geometry.base + 64 * word_index + 16 * digit
            + ((word >> (4 * digit)) & 15)
            for word_index, word in enumerate(case["global_words"][:refresh_global_prefix])
            for digit in range(4)
        ]
        if audit[0]["global_refresh_tags"] != expected_refresh:
            raise ValueError("global refresh sent tags outside the operand window")
    groups = case["workgroups"]
    interleaved = case.get("interleaved", False)
    transport = actual["transport"]
    if (
        actual["outcome"] != "halted"
        or (not interleaved and actual["workgroup_cursor"] != groups - 1)
        or transport["fresh_caches"] != int(first)
        or transport["matrix_initial_tokens"]
        != circuit.memory_geometry.upload_tokens * int(first)
        or not all(
            transport[f"resident_{name}_exact"]
            for name in ("grid", "memory", "rom", "state")
        )
        or any(
            not item["resident_memory_exact"]
            for item in actual["workgroup_transitions"]
        )
        or transport.get("global_refresh_tokens", 0) != refresh_tokens
        or audit[0]["memory"]
        != circuit.memory_geometry.upload_tokens * int(first) + refresh_tokens
        or sum(item["memory"] for item in audit)
        != circuit.memory_geometry.upload_tokens * int(first)
        + refresh_tokens + transport["matrix_patch_tokens"]
        or [token for call in audit for token in call["transitions"]]
        != ([] if interleaved else list(WORKGROUP_TICKETS[: groups - 1]))
        or [token for call in audit for token in call["resets"]]
        != ([] if interleaved else list(RESET_WARP) * (groups - 1))
    ):
        raise ValueError("globalgrid cache, launch boundary, or memory audit failed")
    return cache, {**actual, "programs": data["programs"], "audit": audit}


ID_PROBE = """ADD32I 0
GLOBAL_ID
SHL32I 1
SAVE_INDEX
GLOBAL_ID
STORE_INDEX
ADD_INDEX_LO 32
LANE_ID
THREAD_ID
WORKGROUP_ID
MASK_LANES 1
HALT"""

PRODUCER = """GLOBAL_ID
SHL32I 1
SAVE_INDEX
LOAD_INDEX
LOAD_INDEX_HI
WARP_SUM
MASK_LANES 1
STORE_INDEX
STORE_INDEX_HI
BARRIER
ADD_INDEX_LO 16
ADD_INDEX_HI 16
BARRIER
STORE_INDEX
STORE_INDEX_HI
HALT"""

CONSUMER = """LANE_ID
SHL32I 5
SAVE_INDEX
MASK_LANES 2
LOAD_INDEX
LOAD_INDEX_HI
WARP_SUM
MASK_LANES 1
STORE_INDEX
STORE_INDEX_HI
HALT"""


def qualify(
    model, circuit, values: list[int], device: str, offset: int, ids_only: bool
) -> dict:
    sentinel = [0xFFF0 + i % 16 for i in range(circuit.memory_geometry.words)]
    probes = []
    for groups in (1, 2, 3, 4):
        _, probe = launch(
            model, circuit, make_launch(sentinel, groups, ID_PROBE), device, offset
        )
        expected = sentinel[:]
        expected[: 32 * groups : 2] = range(16 * groups)
        if probe["global_words"] != expected:
            raise ValueError("ID probe corrupted an inactive or guard word")
        probes.append(probe)
    report = {"id_probes": probes}
    if ids_only:
        return report

    words = [
        part for value in values for part in (value & 65535, value >> 16)
    ] + sentinel[64:]
    cache, producer = launch(
        model, circuit, make_launch(words, 2, PRODUCER), device, offset
    )
    expected = words[:]
    partials = [sum(values[start : start + 16]) & MASK32 for start in (0, 16)]
    for group, partial in enumerate(partials):
        for address in (group * 32, group * 32 + 16):
            expected[address : address + 2] = [partial & 65535, partial >> 16]
    if producer["global_words"] != expected:
        raise ValueError(
            "producer partials or guard words disagree with scalar calculation"
        )
    _, consumer = launch(
        model,
        circuit,
        make_launch(producer["global_words"], 1, CONSUMER, "HALT"),
        device,
        offset,
        cache=cache,
    )
    total = sum(values) & MASK32
    expected[:2] = [total & 65535, total >> 16]
    if consumer["global_words"] != expected:
        raise ValueError(
            "consumer reduction or guard words disagree with scalar calculation"
        )
    return {
        **report,
        "producer": producer,
        "consumer": consumer,
        "partials": partials,
        "sum": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=FIXTURE)
    parser.add_argument("--model", type=Path)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--resident-memory512", action="store_true")
    variants.add_argument("--resident-rom32", action="store_true")
    variants.add_argument("--resident-alu32", action="store_true")
    variants.add_argument("--resident-scoped", action="store_true")
    variants.add_argument("--resident-interleaved", action="store_true")
    variants.add_argument("--full-integer", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--ids-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = args.input.read_bytes()
    values = _unsigned(json.loads(payload), 32, MASK32, "input")
    meta = (
        json.loads((args.model / "qwengpu9b.json").read_text()) if args.model else None
    )
    if meta is not None and (
        meta.get("layout")
        not in (RESIDENT_GLOBALGRID_LAYOUT, RESIDENT_COMPUTE_LAYOUT, *MEMORY512_LAYOUTS)
        or meta.get("chip_vocab")
        != (
            ALU_VOCAB
            if meta.get("layout") in ALU32_LAYOUTS
            else ROM32_VOCAB
            if meta.get("layout") in ROM32_LAYOUTS
            else WIDE_MEMORY.vocab
            if meta.get("layout") in MEMORY512_LAYOUTS
            else TILEGRID_VOCAB
        )
    ):
        parser.error("expected a merged globalgrid checkpoint")
    model, circuit = load_tile_model(
        args.model,
        args.device,
        resident_add32=True,
        resident_globalgrid=True,
        resident_memory512=args.resident_memory512,
        resident_rom32=args.resident_rom32,
        resident_alu32=args.resident_alu32,
        resident_scoped=args.resident_scoped,
        resident_interleaved=args.resident_interleaved,
        full_integer=args.full_integer,
    )
    result = qualify(
        model,
        circuit,
        values,
        args.device,
        meta["chip_token_offset"] if meta else 0,
        args.ids_only,
    )
    report = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "merge": meta,
        "circuit": circuit.stats(),
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "input": values,
        **result,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Four ID and global memory probes exact on {args.device}")
    if not args.ids_only:
        print(f"Cross-workgroup reduction exact, sum = 0x{result['sum']:08x}")


if __name__ == "__main__":
    main()
