"""Estimate the expanded checkpoint using graph allocation and meta tensors."""

from __future__ import annotations

import argparse
import json
import subprocess
import types
from pathlib import Path

import torch
from chipformer.q35.emit import allocate
from chipformer.q35.ir import Target
from chipformer.q35.merge import pick_hidden
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from qwengpu.alu32 import ALU_VOCAB
from qwengpu.compiled_tile8 import ROM32_VOCAB, build, rom_payload
from qwengpu.memory_geometry import WIDE_MEMORY
from qwengpu.merge9b import merged_text_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--legacy-ref")
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--resident-rom32", action="store_true")
    variants.add_argument("--resident-alu32", action="store_true")
    args = parser.parse_args()
    legacy = None
    if args.legacy_ref:
        old = types.ModuleType("previous_tile8")
        source = subprocess.check_output(
            ["git", "show", f"{args.legacy_ref}:qwengpu/compiled_tile8.py"], text=True
        )
        exec(  # noqa: S102 - compare the pinned local builder without editing the checkout.
            compile(source, args.legacy_ref + ":compiled_tile8.py", "exec"),
            old.__dict__,
        )
        layout = (
            {"resident_memory512": True}
            if args.resident_rom32 or args.resident_alu32
            else {"resident_compute": True}
        )
        for target in (None, Target(16, 32, 12288)):
            previous = old.build(target, **layout)
            current = build(target, **layout)
            assert repr(previous.nodes) == repr(current.nodes)
            assert repr(previous.dims) == repr(current.dims)
        flags = {
            "indexed": True,
            "masked": True,
            "divergent": True,
            "looping": True,
            "resident_add32": True,
            "resident_shared32": True,
            "resident_addmem32": True,
            "resident_branch32": True,
            "resident_loop32": True,
            "resident_globalgrid": True,
            "resident_compute": True,
            **(
                {"resident_memory512": True}
                if args.resident_rom32 or args.resident_alu32
                else {}
            ),
        }
        assert all(
            old.rom_payload(t, **flags) == rom_payload(t, **flags)
            for t in range(old.ROM, old.ROM + 16 * old.STRIDE)
        )
        legacy = {
            "source": args.legacy_ref,
            "ir_identical": True,
            "rom_domain_identical": True,
        }
    root = Path(__file__).resolve().parents[1]
    prior = json.loads((root / "results/compute_checkpoint_manifest.json").read_text())
    base = prior["files"]["config.json"].copy()
    base["hidden_size"] = 4096
    base["vocab_size"] = prior["files"]["qwengpu9b.json"]["chip_token_offset"]
    layers = prior["files"]["qwengpu9b.json"]["chip_layers"]
    base["layer_types"] = base["layer_types"][layers:]
    base["num_hidden_layers"] = len(base["layer_types"])
    circuit = build(
        Target(
            base["linear_num_key_heads"],
            base["linear_num_value_heads"],
            base["intermediate_size"],
        ),
        resident_memory512=True,
        resident_rom32=args.resident_rom32,
        resident_alu32=args.resident_alu32,
    )
    marker = circuit.token_feature("simt.marker", lambda _t: 0, full=True)
    circuit.ffn(
        [(2.0**40, circuit.one, circuit.one)],
        "exit.marker",
        into=marker,
        min_level=max(node.level for node in circuit.nodes),
    )
    keep = set(circuit.keep) | set(marker.terms)
    allocation = allocate(circuit, None, keep, base["intermediate_size"])
    hidden, _m = pick_hidden(4096, allocation.n_slots + 1)
    vocab = (
        ALU_VOCAB
        if args.resident_alu32
        else ROM32_VOCAB
        if args.resident_rom32
        else WIDE_MEMORY.vocab
    )
    config = merged_text_config(
        base,
        hidden=hidden,
        n_chip=circuit.n_layers(),
        chip_vocab=vocab,
        tied=False,
    )
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
    size = sum(parameter.numel() * 2 for parameter in model.parameters())
    maximum = max(
        max(node.k, default=0) for node in circuit.nodes if hasattr(node, "k")
    )
    assert maximum < 128
    report = {
        "legacy": legacy,
        "base_revision": prior["base_revision"],
        "circuit": circuit.stats(),
        "slots": allocation.n_slots,
        "merged_hidden": hidden,
        "estimated_weight_bytes_bf16": size,
        "estimated_weight_gib_bf16": size / 2**30,
        "maximum_head_row": maximum,
        "vocab": vocab,
        "weights_allocated": False,
        **(
            {}
            if args.resident_rom32 or args.resident_alu32
            else {"naive_initial_preflight": {"slots": 6673, "merged_hidden": 11008}}
        ),
        "optimization": (
            "Dedicated opcode extension bit and shared ROM address features."
            if args.resident_alu32
            else "Five binary ROM address features shared across the two warp banks."
            if args.resident_rom32 or args.resident_alu32
            else "Shared row/nibble/bank features and read-value bank selection."
        ),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
