"""Merge the compiled SIMT circuit before pretrained Qwen3.5-9B.

Adapted from QwenCPU's MIT-licensed 9B merge. It uses Qwenputer's generic
Qwen3.5 circuit emitter and preserves the pretrained model's untied output
head. Trace tokens are raw model IDs starting at ``chip_token_offset``; the
copied text tokenizer does not encode or decode them. No checkpoint weights
are included with this source file.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open

from chipformer.q35.emit import Layout, allocate, check_bf16, embedding_rows, emit_layers
from chipformer.q35.merge import IN_COLS, OUT_ROWS, WIDE_NORMS, pick_hidden
from chipformer.q35.tiny import code_gain
SRC_TEXT = "model.language_model."
DST_TEXT = "model."
TAU_TEXT = 2.0**-40
TAU_GAIN = 2.0**60


def output_head_is_tied(cfg: dict) -> bool:
    return bool(cfg.get("tie_word_embeddings",
                        cfg["text_config"].get("tie_word_embeddings", True)))


def merged_text_config(tc0: dict, *, hidden: int, n_chip: int,
                       chip_vocab: int, tied: bool) -> dict:
    tc = {key: value for key, value in tc0.items() if key != "full_attention_interval"}
    tc.update(hidden_size=hidden, num_hidden_layers=n_chip + tc0["num_hidden_layers"],
              mtp_num_hidden_layers=0,
              layer_types=["linear_attention"] * n_chip + tc0["layer_types"],
              vocab_size=-(-(tc0["vocab_size"] + chip_vocab) // 64) * 64,
              tie_word_embeddings=tied, architectures=["Qwen3_5ForCausalLM"],
              model_type="qwen3_5_text")
    return tc


def _copy_tokenizer(src: Path, dst: Path) -> None:
    """Keep the base text tokenizer; SIMT token IDs are handled by the adapter."""
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "special_tokens_map.json", "LICENSE"):
        path = src / name
        if path.exists():
            shutil.copy2(path, dst / name)


def _load_widened_text(src: Path, hidden: int, n_chip: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Copy only the text tower and preserve the learned untied output head."""
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    d0 = cfg["text_config"]["hidden_size"]
    v0 = cfg["text_config"]["vocab_size"]
    scale = (d0 / hidden) ** 0.5
    sd: dict[str, torch.Tensor] = {}
    embedding = head = None
    for shard in sorted(src.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as file:
            for name in file.keys():
                if name == "lm_head.weight":
                    head = file.get_tensor(name)
                    continue
                if not name.startswith(SRC_TEXT):
                    continue
                value = file.get_tensor(name)
                if name == SRC_TEXT + "embed_tokens.weight":
                    embedding = value
                    continue
                if IN_COLS.search(name):
                    value = torch.cat((value, value.new_zeros(*value.shape[:-1], hidden - d0)), -1)
                elif OUT_ROWS.search(name):
                    value = torch.cat((value, value.new_zeros(hidden - d0, *value.shape[1:])), 0)
                elif WIDE_NORMS.search(name):
                    gain = (1 + value.float()) * scale - 1
                    value = torch.cat((gain.to(value.dtype), value.new_zeros(hidden - d0)))
                renamed = DST_TEXT + name[len(SRC_TEXT):]
                renamed = re.sub(r"^(model\.layers\.)(\d+)\.",
                                 lambda match: f"{match[1]}{int(match[2]) + n_chip}.", renamed)
                sd[renamed] = value
    if embedding is None or tuple(embedding.shape) != (v0, d0):
        raise ValueError(f"expected text embeddings shaped {(v0, d0)}")
    if head is None or tuple(head.shape) != (v0, d0):
        raise ValueError(f"expected untied lm_head shaped {(v0, d0)}")
    return sd, embedding, head


def build_merged(src: Path | str, dst: Path | str, *, fit_sinks: bool = True,
                 rom: bool = False, writeback: bool = False, control: bool = False, resume: bool = False, isa: bool = False, workgroup: bool = False, tile: bool = False, tile8: bool = False, resident: bool = False, indexed: bool = False, masked: bool = False, divergent: bool = False, looping: bool = False, resident_rom: bool = False, resident_state: bool = False, resident_delta: bool = False, resident_plan: bool = False, resident_io: bool = False, resident_grid: bool = False, resident_wide: bool = False, resident_add32: bool = False, resident_shared32: bool = False, resident_addmem32: bool = False, resident_branch32: bool = False, resident_loop32: bool = False, resident_tilebanks: bool = False, resident_tilepipe: bool = False, resident_tilegrid: bool = False, resident_globalgrid: bool = False, resident_compute: bool = False, resident_memory512: bool = False, resident_rom32: bool = False, resident_alu32: bool = False, resident_scoped: bool = False, resident_interleaved: bool = False, full_integer: bool = False, internal_stores: bool = False, add32: bool = False) -> dict:
    """Write a text-only stock Qwen3.5 checkpoint with the SIMT circuit first."""
    from safetensors.torch import save_file

    src, dst = Path(src), Path(dst)
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    if output_head_is_tied(cfg):
        raise ValueError("this merge targets the untied Qwen3.5-9B output head")
    tc0 = cfg["text_config"]
    d0, v0, inter = tc0["hidden_size"], tc0["vocab_size"], tc0["intermediate_size"]
    from chipformer.q35.ir import Target

    if sum((rom, writeback, control, resume, isa, workgroup, tile, tile8, resident, indexed, masked, divergent, looping, resident_rom, resident_state, resident_delta, resident_plan, resident_io, resident_grid, resident_wide, resident_add32, resident_shared32, resident_addmem32, resident_branch32, resident_loop32, resident_tilebanks, resident_tilepipe, resident_tilegrid, resident_globalgrid, resident_compute, resident_memory512, resident_rom32, resident_alu32, resident_scoped, resident_interleaved, full_integer, internal_stores, add32)) > 1:
        raise ValueError("choose one circuit variant")
    if tile8 or resident or indexed or masked or divergent or looping or resident_rom or resident_state or resident_delta or resident_plan or resident_io or resident_grid or resident_wide or resident_add32 or resident_shared32 or resident_addmem32 or resident_branch32 or resident_loop32 or resident_tilebanks or resident_tilepipe or resident_tilegrid or resident_globalgrid or resident_compute or resident_memory512 or resident_rom32 or resident_alu32 or resident_scoped or resident_interleaved or full_integer or internal_stores:
        from qwengpu.alu32 import ALU_VOCAB, FULL_INTEGER_LAYOUT
        from qwengpu.compiled_tile8 import ROM32_VOCAB, TILEGRID_VOCAB, VOCAB, build
        from qwengpu.memory_geometry import WIDE_MEMORY
        from qwengpu.model_stores import INTERNAL_STORES_LAYOUT

        chip_vocab = ALU_VOCAB if resident_alu32 or resident_scoped or resident_interleaved or full_integer or internal_stores else ROM32_VOCAB if resident_rom32 else WIDE_MEMORY.vocab if resident_memory512 else TILEGRID_VOCAB if resident_tilegrid or resident_globalgrid or resident_compute else VOCAB
    elif add32:
        from qwengpu.compiled_add32 import VOCAB as chip_vocab
        from qwengpu.compiled_add32 import build
    elif tile:
        from qwengpu.compiled_tile import VOCAB as chip_vocab
        from qwengpu.compiled_tile import build
    elif control or resume or isa or workgroup:
        from qwengpu.compiled_control import VOCAB as chip_vocab
        from qwengpu.compiled_control import build
    elif writeback:
        from qwengpu.compiled_writeback import VOCAB as chip_vocab
        from qwengpu.compiled_writeback import build
    elif rom:
        from qwengpu.compiled_rom import VOCAB as chip_vocab
        from qwengpu.compiled_rom import build
    else:
        from qwengpu.compiled_add import VOCAB as chip_vocab
        from qwengpu.compiled_add import build
    target = Target(tc0["linear_num_key_heads"], tc0["linear_num_value_heads"], inter)
    if resident or indexed or masked or divergent or looping or resident_rom or resident_state or resident_delta or resident_plan or resident_io or resident_grid or resident_wide or resident_add32 or resident_shared32 or resident_addmem32 or resident_branch32 or resident_loop32 or resident_tilebanks or resident_tilepipe or resident_tilegrid or resident_globalgrid or resident_compute or resident_memory512 or resident_rom32 or resident_alu32 or resident_scoped or resident_interleaved or full_integer or internal_stores:
        circuit = build(target, resident=resident, indexed=indexed, masked=masked, divergent=divergent, looping=looping, resident_rom=resident_rom, resident_state=resident_state, resident_delta=resident_delta, resident_plan=resident_plan, resident_io=resident_io, resident_grid=resident_grid, resident_wide=resident_wide, resident_add32=resident_add32, resident_shared32=resident_shared32, resident_addmem32=resident_addmem32, resident_branch32=resident_branch32, resident_loop32=resident_loop32, resident_tilebanks=resident_tilebanks, resident_tilepipe=resident_tilepipe, resident_tilegrid=resident_tilegrid, resident_globalgrid=resident_globalgrid, resident_compute=resident_compute, resident_memory512=resident_memory512, resident_rom32=resident_rom32, resident_alu32=resident_alu32, resident_scoped=resident_scoped, resident_interleaved=resident_interleaved, full_integer=full_integer, internal_stores=internal_stores)
    elif workgroup:
        circuit = build(target, workgroup=True)
    elif resume or isa:
        circuit = build(target, resumable=True, extended=isa)
    else:
        circuit = build(target)
    # The generic emitter clears its ballast at the exit. Leave a separate
    # marker, written only at the exit, for Qwen's mute gates. It starts at
    # zero, so it cannot disturb the SIMT circuit's earlier sublayers.
    marker = circuit.token_feature("simt.marker", lambda _token: 0, full=True)
    floor = max(node.level for node in circuit.nodes)
    circuit.ffn([(2.0**40, circuit.one, circuit.one)], "exit.marker",
                into=marker, min_level=floor)
    keep = set(circuit.keep)
    marker_dim = next(iter(marker.terms))
    keep.add(marker_dim)
    allocation = allocate(circuit, None, keep, inter)
    hidden, m = pick_hidden(d0, max(3269 if resident_alu32 or resident_scoped or resident_interleaved or full_integer or internal_stores else 0, 1 + allocation.n_slots))
    layout = Layout(hidden=hidden, off=d0, m=m, slots=allocation.slots)
    if d0 + m + 1 + allocation.n_slots > hidden:
        raise ValueError("circuit slots exceed widened hidden size")
    layout.scale  # verify exact power-of-two scaling before allocating weights
    n_chip = circuit.n_layers()
    text_config = merged_text_config(tc0, hidden=hidden, n_chip=n_chip,
                                     chip_vocab=chip_vocab, tied=False)
    sd, base_embedding, base_head = _load_widened_text(src, hidden, n_chip)

    # At the exit the generic emitter clears the original ballast. The marker
    # now holds a single 2^40 value on SIMT tokens and zero on text tokens.
    from chipformer.q35.mute import fit_anti_sink, mute

    sinks = fit_anti_sink(src, "cuda" if torch.cuda.is_available() else "cpu") if fit_sinks else None
    mute(sd, text_config, tc0["num_hidden_layers"], layout.dim(marker_dim),
         hidden**0.5, prefix=DST_TEXT, sinks=sinks, first=n_chip)
    chip_layers = emit_layers(circuit, layout, text_config, 0, DST_TEXT, allocation, keep)
    sd.update({name: value.to(torch.bfloat16) for name, value in chip_layers.items()})
    del chip_layers

    vocab = text_config["vocab_size"]
    embedding = torch.zeros(vocab, hidden, dtype=torch.bfloat16)
    embedding[:v0, :d0] = base_embedding
    embedding[:v0, layout.tau] = TAU_TEXT
    chip_rows = embedding_rows(circuit, layout, list(range(chip_vocab)), hidden)
    embedding[v0:v0 + chip_vocab] = chip_rows.to(torch.bfloat16)
    del base_embedding, chip_rows
    sd[DST_TEXT + "embed_tokens.weight"] = embedding

    # The independent 9B head remains exactly intact in its text columns.
    # SIMT rows read only the output-code dimensions. Tau suppresses chip and
    # padded rows at ordinary text positions, where the circuit code is zero.
    head = torch.zeros_like(embedding)
    head[:v0, :d0] = base_head
    code_dims = [layout.dim(next(iter(expr.terms))) for expr in circuit.code]
    head[v0:, layout.tau] = -1
    head[v0:v0 + chip_vocab, code_dims] = embedding[v0:v0 + chip_vocab, code_dims]
    sd["lm_head.weight"] = head
    del base_head

    norm = sd[DST_TEXT + "norm.weight"].float()
    norm[d0:] = -1
    norm[code_dims] = code_gain(layout)
    norm[layout.tau] = TAU_GAIN
    check_bf16("final norm", norm[d0:])
    sd[DST_TEXT + "norm.weight"] = norm.to(torch.bfloat16)

    dst.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, torch.Tensor]] = []
    shard: dict[str, torch.Tensor] = {}
    shard_size = 0
    for name in sorted(sd):
        value = sd[name].contiguous()
        shard[name] = value
        shard_size += value.numel() * value.element_size()
        if shard_size > 2**31:
            shards.append(shard)
            shard, shard_size = {}, 0
    if shard:
        shards.append(shard)
    index = {"metadata": {"total_size": sum(value.numel() * value.element_size()
                                               for value in sd.values())},
             "weight_map": {}}
    for i, values in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(values, str(dst / name), metadata={"format": "pt"})
        index["weight_map"].update({key: name for key in values})
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (dst / "config.json").write_text(json.dumps(text_config, indent=2), encoding="utf-8")
    (dst / "generation_config.json").write_text(json.dumps({"do_sample": False,
        "pad_token_id": tc0.get("eos_token_id", 248044), "eos_token_id": tc0.get("eos_token_id", 248044)},
        indent=2), encoding="utf-8")
    _copy_tokenizer(src, dst)
    meta = {"layout": ("qwengpu-add32-9b-v1" if add32 else
                       INTERNAL_STORES_LAYOUT if internal_stores else
                       FULL_INTEGER_LAYOUT if full_integer else
                       "qwengpu-resident-interleaved-9b-v1" if resident_interleaved else
                       "qwengpu-resident-scoped-9b-v1" if resident_scoped else
                       "qwengpu-resident-alu32-9b-v1" if resident_alu32 else
                       "qwengpu-resident-rom32-9b-v1" if resident_rom32 else
                       "qwengpu-resident-memory512-9b-v1" if resident_memory512 else
                       "qwengpu-resident-compute-9b-v1" if resident_compute else
                       "qwengpu-resident-globalgrid-9b-v1" if resident_globalgrid else
                       "qwengpu-resident-tilegrid-9b-v1" if resident_tilegrid else
                       "qwengpu-resident-tilepipe-9b-v1" if resident_tilepipe else
                       "qwengpu-resident-tilebanks-9b-v1" if resident_tilebanks else
                       "qwengpu-resident-loop32-9b-v1" if resident_loop32 else
                       "qwengpu-resident-branch32-9b-v1" if resident_branch32 else
                       "qwengpu-resident-addmem32-9b-v1" if resident_addmem32 else
                       "qwengpu-resident-shared32-9b-v1" if resident_shared32 else
                       "qwengpu-resident-add32-9b-v1" if resident_add32 else
                       "qwengpu-resident-wide9b-v1" if resident_wide else
                       "qwengpu-resident-grid9b-v1" if resident_grid else
                       "qwengpu-resident-io9b-v1" if resident_io else
                       "qwengpu-resident-plan9b-v1" if resident_plan else
                       "qwengpu-resident-delta9b-v1" if resident_delta else
                       "qwengpu-resident-state9b-v1" if resident_state else
                       "qwengpu-resident-rom9b-v1" if resident_rom else
                       "qwengpu-looping9b-v1" if looping else
                       "qwengpu-divergent9b-v1" if divergent else
                       "qwengpu-masked9b-v1" if masked else
                       "qwengpu-indexed9b-v1" if indexed else
                       "qwengpu-resident9b-v1" if resident else
                       "qwengpu-tile8-9b-v1" if tile8 else
                       "qwengpu-tile9b-v1" if tile else
                       "qwengpu-workgroup9b-v1" if workgroup else
                       "qwengpu-isa9b-v1" if isa else
                       "qwengpu-resume9b-v1" if resume else
                       "qwengpu-control9b-v1" if control else
                       "qwengpu-writeback9b-v1" if writeback else
                       "qwengpu-rom9b-v1" if rom else "qwengpu9b-v1"),
            "chip_token_offset": v0,
            "chip_vocab": chip_vocab,
            "chip_layers": n_chip, "hidden": hidden, "slots": allocation.n_slots,
            "code_dims": code_dims, "tau_dim": layout.tau,
            "mute_dim": layout.dim(marker_dim),
            "fit_anti_sink": fit_sinks, "circuit": circuit.stats(),
            "output_head_tied": False}
    (dst / "qwengpu9b.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Qwen3.5-9B source snapshot")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-anti-sink", action="store_true")
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--full-integer", action="store_true")
    variants.add_argument("--internal-stores", action="store_true")
    variants.add_argument("--resident-interleaved", action="store_true")
    variants.add_argument("--resident-scoped", action="store_true")
    variants.add_argument("--resident-alu32", action="store_true")
    variants.add_argument("--resident-rom32", action="store_true",
                          help="merge the 32-slot resident ROM and 512-byte global memory")
    variants.add_argument("--resident-memory512", action="store_true",
                          help="merge the 512-byte banked programmable memory circuit")
    variants.add_argument("--add32", action="store_true",
                          help="merge the isolated masked 32-bit addition circuit")
    variants.add_argument("--resident-compute", action="store_true",
                          help="merge bounded compute kernels in global memory")
    variants.add_argument("--resident-globalgrid", action="store_true",
                          help="merge programmable workgroups with shared global memory")
    variants.add_argument("--resident-tilegrid", action="store_true",
                          help="merge the one-cache four-workgroup tiled engine")
    variants.add_argument("--resident-tilepipe", action="store_true",
                          help="merge tilebanks with a model-scheduled K-slice transition")
    variants.add_argument("--resident-tilebanks", action="store_true",
                          help="merge loop32 with two preloaded tile operand banks")
    variants.add_argument("--resident-loop32", action="store_true",
                          help="merge branch32 with full-width WHILE_BIT")
    variants.add_argument("--resident-branch32", action="store_true",
                          help="merge addmem32 with full-width IF_BIT")
    variants.add_argument("--resident-addmem32", action="store_true",
                          help="merge shared32 with indexed 32-bit arithmetic")
    variants.add_argument("--resident-shared32", action="store_true",
                          help="merge ADD32I with indexed high-limb loads and stores")
    variants.add_argument("--resident-add32", action="store_true",
                          help="merge the resident wide engine with ADD32I")
    variants.add_argument("--resident-wide", action="store_true",
                          help="merge the resident grid engine with high register banks")
    variants.add_argument("--resident-grid", action="store_true",
                          help="merge the resident I/O engine with programmable row pairs")
    variants.add_argument("--resident-io", action="store_true",
                          help="merge with model-generated state and store read plans")
    variants.add_argument("--resident-plan", action="store_true",
                          help="merge with model-generated state read plans")
    variants.add_argument("--resident-delta", action="store_true",
                          help="merge with model-selected updates to banked warp state")
    variants.add_argument("--resident-state", action="store_true",
                          help="merge with resident programs and banked warp state")
    variants.add_argument("--resident-rom", action="store_true",
                          help="merge the looping tiled engine with two resident warp programs")
    variants.add_argument("--looping", action="store_true",
                          help="merge the resident tiled engine with data-dependent loops")
    variants.add_argument("--divergent", action="store_true",
                          help="merge the resident tiled engine with nested branches")
    variants.add_argument("--masked", action="store_true",
                          help="merge the indexed resident tiled engine with lane masks")
    variants.add_argument("--indexed", action="store_true",
                          help="merge the resident tiled engine with indexed shared memory")
    variants.add_argument("--resident", action="store_true",
                          help="merge the resident 8x8 tiled engine")
    variants.add_argument("--tile8", action="store_true",
                          help="merge the 8x8 tiled engine")
    variants.add_argument("--tile", action="store_true",
                          help="merge the model-scheduled tile engine")
    variants.add_argument("--workgroup", action="store_true",
                          help="merge the cooperative workgroup ISA")
    variants.add_argument("--isa", action="store_true",
                          help="merge the extended resumable integer ISA")
    variants.add_argument("--resume", action="store_true",
                          help="merge control with exportable execution windows")
    variants.add_argument("--control", action="store_true",
                          help="merge bounded loops and nested divergence")
    variants.add_argument("--writeback", action="store_true",
                          help="merge bounded internal-writeback circuit")
    variants.add_argument("--rom", action="store_true",
                        help="merge the autonomous straight-line ROM circuit")
    args = parser.parse_args()
    print(json.dumps(build_merged(args.src, args.out,
                                  fit_sinks=not args.no_anti_sink,
                                  rom=args.rom, writeback=args.writeback, control=args.control,
                                  resume=args.resume, isa=args.isa, workgroup=args.workgroup, tile=args.tile, tile8=args.tile8, resident=args.resident, indexed=args.indexed, masked=args.masked, divergent=args.divergent, looping=args.looping, resident_rom=args.resident_rom, resident_state=args.resident_state, resident_delta=args.resident_delta, resident_plan=args.resident_plan, resident_io=args.resident_io, resident_grid=args.resident_grid, resident_wide=args.resident_wide, resident_add32=args.resident_add32, resident_shared32=args.resident_shared32, resident_addmem32=args.resident_addmem32, resident_branch32=args.resident_branch32, resident_loop32=args.resident_loop32, resident_tilebanks=args.resident_tilebanks, resident_tilepipe=args.resident_tilepipe, resident_tilegrid=args.resident_tilegrid, resident_globalgrid=args.resident_globalgrid, resident_compute=args.resident_compute, resident_memory512=args.resident_memory512, resident_rom32=args.resident_rom32, resident_alu32=args.resident_alu32, resident_scoped=args.resident_scoped, resident_interleaved=args.resident_interleaved, full_integer=args.full_integer, internal_stores=args.internal_stores, add32=args.add32)), flush=True)


if __name__ == "__main__":
    main()
