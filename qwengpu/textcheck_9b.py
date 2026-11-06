"""Small fixed text check for the pretrained Qwen3.5-9B SIMT merge."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
from pathlib import Path

import torch
import transformers

PROMPTS = (
    "Hello! Tell me one interesting fact about the Moon.",
    "Write a Python function that reverses a list without changing the original.",
    "What is the difference between a mutex and a semaphore?",
    "Explain why ice floats on liquid water.",
    'Continue this JSON object with one field: {"status": "ok"',
    "The café opened in 東京. Describe the scene in English.",
)


def _ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
    )
    if not isinstance(encoded, list):
        encoded = encoded["input_ids"]
    return [int(token) for token in encoded]


@torch.no_grad()
def _measure(
    model,
    ids: list[int],
    vocab: int,
    device: str,
    decoding: transformers.GenerationConfig,
) -> tuple[torch.Tensor, list[int]]:
    sequence = torch.tensor([ids], device=device)
    logits = (
        model(input_ids=sequence, logits_to_keep=1).logits[0, -1, :vocab].float().cpu()
    )
    continuation = model.generate(sequence, generation_config=decoding)[
        0, len(ids) :
    ].tolist()
    return logits, continuation


def load_prompts(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return PROMPTS
    prompts = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
    ):
        raise ValueError("the corpus must contain a nonempty list of prompt strings")
    if len(set(prompts)) != len(prompts):
        raise ValueError("the corpus contains duplicate prompts")
    return tuple(prompts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("merged", type=Path)
    parser.add_argument("--corpus", type=Path, help="JSON array of frozen prompts")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    try:
        prompts = load_prompts(args.corpus)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    from transformers import (
        AutoModelForImageTextToText,
        AutoTokenizer,
        Qwen3_5ForCausalLM,
    )

    meta = json.loads((args.merged / "qwengpu9b.json").read_text(encoding="utf-8"))
    text_vocab = int(meta["chip_token_offset"])
    base_tok = AutoTokenizer.from_pretrained(args.base)
    merged_tok = AutoTokenizer.from_pretrained(args.merged)
    base_ids = [_ids(base_tok, prompt) for prompt in prompts]
    merged_ids = [_ids(merged_tok, prompt) for prompt in prompts]
    if base_ids != merged_ids:
        raise ValueError("base and merged tokenizers differ on fixed prompts")
    pad = base_tok.pad_token_id or base_tok.eos_token_id or 248044

    base_config = json.loads((args.base / "config.json").read_text(encoding="utf-8"))
    base_class = (
        Qwen3_5ForCausalLM
        if base_config.get("model_type") == "qwen3_5_text"
        else AutoModelForImageTextToText
    )
    base = (
        base_class.from_pretrained(args.base, dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )
    # Use the base generation settings for both models, including every EOS ID.
    decoding = transformers.GenerationConfig.from_dict(base.generation_config.to_dict())
    decoding.max_new_tokens = args.max_new_tokens
    decoding.do_sample = False
    decoding.pad_token_id = pad
    base_outputs = [
        _measure(base, ids, text_vocab, args.device, decoding) for ids in base_ids
    ]
    del base
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    merged = (
        Qwen3_5ForCausalLM.from_pretrained(args.merged, dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )
    records = []
    for prompt, ids, (base_logits, base_cont) in zip(
        prompts, merged_ids, base_outputs, strict=True
    ):
        merged_logits, merged_cont = _measure(
            merged, ids, text_vocab, args.device, decoding
        )
        difference = (base_logits - merged_logits).abs()
        base_top = base_logits.topk(2)
        merged_top = merged_logits.topk(2)
        if not torch.isfinite(difference).all():
            raise ValueError("nonfinite text logit difference")
        prefix = next(
            (i for i, (a, b) in enumerate(zip(base_cont, merged_cont)) if a != b),
            min(len(base_cont), len(merged_cont)),
        )
        record = {
            "prompt": prompt,
            "prompt_tokens": len(ids),
            "base_next_top1": int(base_logits.argmax()),
            "merged_next_top1": int(merged_logits.argmax()),
            "next_top1_same": int(base_logits.argmax()) == int(merged_logits.argmax()),
            "base_next_margin": float(base_top.values[0] - base_top.values[1]),
            "merged_next_margin": float(merged_top.values[0] - merged_top.values[1]),
            "next_max_abs_logit_difference": float(difference.max()),
            "next_mean_abs_logit_difference": float(difference.mean()),
            "greedy_identical": base_cont == merged_cont,
            "greedy_prefix_tokens": prefix,
            "base_greedy_ids": base_cont,
            "merged_greedy_ids": merged_cont,
            "merged_chip_tokens": sum(token >= text_vocab for token in merged_cont),
        }
        if args.max_new_tokens == 4:
            record["greedy_four_identical"] = record["greedy_identical"]
        records.append(record)
    report = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "merge": meta,
        "prompts_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        "tokenized_prompts_sha256": hashlib.sha256(
            json.dumps(base_ids).encode()
        ).hexdigest(),
        "prompt_count": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "generation_config": decoding.to_diff_dict(),
        "records": records,
        "next_top1_same": sum(record["next_top1_same"] for record in records),
        "greedy_identical": sum(record["greedy_identical"] for record in records),
        "median_next_max_abs_logit_difference": statistics.median(
            record["next_max_abs_logit_difference"] for record in records
        ),
        "max_next_abs_logit_difference": max(
            record["next_max_abs_logit_difference"] for record in records
        ),
        "merged_chip_tokens": sum(record["merged_chip_tokens"] for record in records),
    }
    if args.max_new_tokens == 4:
        report["greedy_four_identical"] = report["greedy_identical"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "prompt_count",
                    "next_top1_same",
                    "greedy_identical",
                    "merged_chip_tokens",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
