"""Small fixed text check for the pretrained Qwen3.5-9B SIMT merge."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
import transformers


PROMPTS = (
    "Hello! Tell me one interesting fact about the Moon.",
    "Write a Python function that reverses a list without changing the original.",
    "What is the difference between a mutex and a semaphore?",
    "Explain why ice floats on liquid water.",
    "Continue this JSON object with one field: {\"status\": \"ok\"",
    "The café opened in 東京. Describe the scene in English.",
)


def _ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True,
        enable_thinking=False, tokenize=True)
    if not isinstance(encoded, list):
        encoded = encoded["input_ids"]
    return [int(token) for token in encoded]


@torch.no_grad()
def _measure(model, ids: list[int], vocab: int, device: str,
             pad: int) -> tuple[torch.Tensor, list[int]]:
    sequence = torch.tensor([ids], device=device)
    logits = model(input_ids=sequence).logits[0, -1, :vocab].float().cpu()
    continuation = model.generate(sequence, max_new_tokens=4, do_sample=False,
                                  pad_token_id=pad)[0, len(ids):].tolist()
    return logits, continuation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("merged", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    from transformers import AutoModelForImageTextToText, AutoTokenizer, Qwen3_5ForCausalLM

    meta = json.loads((args.merged / "qwengpu9b.json").read_text(encoding="utf-8"))
    text_vocab = int(meta["chip_token_offset"])
    base_tok = AutoTokenizer.from_pretrained(args.base)
    merged_tok = AutoTokenizer.from_pretrained(args.merged)
    base_ids = [_ids(base_tok, prompt) for prompt in PROMPTS]
    merged_ids = [_ids(merged_tok, prompt) for prompt in PROMPTS]
    if base_ids != merged_ids:
        raise ValueError("base and merged tokenizers differ on fixed prompts")
    pad = base_tok.pad_token_id or base_tok.eos_token_id or 248044

    base_config = json.loads((args.base / "config.json").read_text(encoding="utf-8"))
    base_class = (Qwen3_5ForCausalLM if base_config.get("model_type") == "qwen3_5_text"
                  else AutoModelForImageTextToText)
    base = base_class.from_pretrained(args.base, dtype=torch.bfloat16).to(args.device).eval()
    base_outputs = [_measure(base, ids, text_vocab, args.device, pad) for ids in base_ids]
    del base
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    merged = Qwen3_5ForCausalLM.from_pretrained(
        args.merged, dtype=torch.bfloat16).to(args.device).eval()
    records = []
    for prompt, ids, (base_logits, base_cont) in zip(
            PROMPTS, merged_ids, base_outputs, strict=True):
        merged_logits, merged_cont = _measure(merged, ids, text_vocab, args.device, pad)
        records.append({"prompt": prompt, "next_top1_same":
                        int(base_logits.argmax()) == int(merged_logits.argmax()),
                        "next_max_abs_logit_difference":
                        float((base_logits - merged_logits).abs().max()),
                        "greedy_four_identical": base_cont == merged_cont,
                        "base_greedy_ids": base_cont,
                        "merged_greedy_ids": merged_cont})
    report = {"device": args.device, "torch": torch.__version__,
              "transformers": transformers.__version__,
              "prompts_sha256": hashlib.sha256(json.dumps(PROMPTS).encode()).hexdigest(),
              "prompt_count": len(PROMPTS), "records": records,
              "next_top1_same": sum(x["next_top1_same"] for x in records),
              "greedy_four_identical": sum(x["greedy_four_identical"] for x in records)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prompt_count": len(PROMPTS),
                      "next_top1_same": report["next_top1_same"],
                      "greedy_four_identical": report["greedy_four_identical"],
                      "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
