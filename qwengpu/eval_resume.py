"""Verify model-generated state transfers across fresh execution windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import transformers
from chipformer.q35.ir import Target

from qwengpu.compiled_control import (
    CHECKPOINT_BASE,
    CHECKPOINT_BYTES,
    EXPORT_BASE,
    RESUME,
    SNAPSHOT_BITS,
    STATE_BASE,
    STATE_COUNT,
    TICK,
    build,
    build_standalone,
    program_tokens,
    state_token,
)
from qwengpu.control_reference import (
    MAX_STEPS,
    ControlState,
    parse_kernel,
    snapshot_bits,
    step,
)
from qwengpu.eval_control import hex_bits, snapshot
from qwengpu.reference import Instruction, WarpState

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "eval/frozen/resume_gate_v1/divergent_scratch_loop.json"
)


def checkpoint_tokens(bits: list[int]) -> list[int]:
    """Encode the initial snapshot or construct an oracle for verification."""
    return [
        CHECKPOINT_BASE
        + index * 256
        + sum(bit << shift for shift, bit in enumerate(bits[index * 8 : index * 8 + 8]))
        for index in range(CHECKPOINT_BYTES)
    ]


@torch.no_grad()
def execute_windows(
    model,
    rom: list[int],
    initial_checkpoint: list[int],
    device: str,
    max_steps: int,
    offset: int = 0,
    observe=None,
) -> dict:
    """Execute with opaque checkpoint tokens and optionally observe model state."""
    checkpoint = initial_checkpoint
    control = state_token(0, 0)
    trace = []
    transfers = []
    calls = 0
    while len(trace) < max_steps:
        # Each window begins with an empty cache and the same immutable program.
        cache = None

        def feed(tokens: list[int], *, all_logits: bool = False) -> list[int]:
            nonlocal cache, calls
            output = model(
                input_ids=torch.tensor(
                    [[offset + token for token in tokens]], device=device
                ),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=len(tokens) if all_logits else 1,
            )
            cache = output.past_key_values
            calls += 1
            return [
                token - offset for token in output.logits[0].argmax(dim=-1).tolist()
            ]

        inputs = rom + checkpoint + [control, TICK]
        for stage in range(MAX_STEPS):
            (generated,) = feed(inputs)
            if not STATE_BASE <= generated < STATE_BASE + STATE_COUNT:
                raise ValueError(f"invalid generated control token {generated}")
            status = generated - STATE_BASE
            exhausted = bool(status & 512)
            record = {"control": generated}
            if observe is not None:
                record["snapshot"] = observe(cache, stage)
            trace.append(record)
            if status & 256:
                return {
                    "trace": trace,
                    "transfers": transfers,
                    "forward_calls": calls,
                    "outcome": "halted",
                }
            if len(trace) == max_steps:
                break
            if stage < MAX_STEPS - 1:
                if exhausted:
                    raise ValueError("model exhausted before the end of the window")
                inputs = [generated, TICK]
        if len(trace) == max_steps:
            break
        if not exhausted:
            raise ValueError("model did not signal exhaustion at the window boundary")
        # Read every byte in one forward pass. The last request emits the
        # same PC with the local step and status reset for the next window.
        outputs = feed(
            [generated, *range(EXPORT_BASE, RESUME), RESUME], all_logits=True
        )
        checkpoint, control = outputs[1:-1], outputs[-1]
        for index, token in enumerate(checkpoint):
            base = CHECKPOINT_BASE + index * 256
            width = min(8, SNAPSHOT_BITS - index * 8)
            if not base <= token < base + (1 << width):
                raise ValueError(f"invalid generated checkpoint token at byte {index}")
        if not STATE_BASE <= control < STATE_BASE + 16:
            raise ValueError(f"invalid generated restart token {control}")
        transfers.append(
            {"after_step": len(trace), "checkpoint": checkpoint, "resume": control}
        )
    return {
        "trace": trace,
        "transfers": transfers,
        "forward_calls": calls,
        "outcome": "step_limit",
    }


def oracle(
    initial: tuple[int, ...], program: tuple[Instruction, ...], max_steps: int
) -> dict:
    state = ControlState(WarpState(lanes=initial))
    trace = []
    transfers = []
    for index in range(max_steps):
        state = step(state, program)
        stage = index % MAX_STEPS
        halted = state.warp.halted
        exhausted = stage == MAX_STEPS - 1 and not halted
        bits = snapshot_bits(state)
        trace.append(
            {
                "control": state_token(
                    state.warp.pc,
                    stage if halted or exhausted else stage + 1,
                    halted,
                    exhausted,
                ),
                "snapshot": hex_bits(bits),
            }
        )
        if halted:
            break
        if exhausted and index + 1 < max_steps:
            transfers.append(
                {
                    "after_step": index + 1,
                    "checkpoint": checkpoint_tokens(bits),
                    "resume": state_token(state.warp.pc, 0),
                }
            )
    return {
        "trace": trace,
        "transfers": transfers,
        "outcome": "halted" if state.warp.halted else "step_limit",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path)
    parser.add_argument("--kernel", type=Path, default=FIXTURE)
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument(
        "--allow-step-limit", action="store_true", help="accept an exact partial trace"
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    payload = args.kernel.read_bytes()
    initial, program = parse_kernel(json.loads(payload))
    expected = oracle(initial, program, args.max_steps)
    meta = None
    offset = 0
    if args.model:
        meta = json.loads((args.model / "qwengpu9b.json").read_text())
        if meta["layout"] != "qwengpu-resume9b-v1":
            raise ValueError("not a resume checkpoint")
        offset = meta["chip_token_offset"]
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16
        )
        cfg = model.config
        circuit = build(
            Target(
                cfg.linear_num_key_heads,
                cfg.linear_num_value_heads,
                cfg.intermediate_size,
            ),
            resumable=True,
        )
    else:
        model, circuit = build_standalone(resumable=True)
    model.to(args.device).eval()
    start = time.monotonic()
    observed = execute_windows(
        model,
        program_tokens(program),
        checkpoint_tokens(snapshot_bits(ControlState(WarpState(lanes=initial)))),
        args.device,
        args.max_steps,
        offset,
        observe=lambda cache, stage: hex_bits(snapshot(cache, circuit, stage)),
    )
    exact = all(observed[key] == value for key, value in expected.items())
    report = {
        "fixture": args.kernel.name,
        "fixture_sha256": hashlib.sha256(payload).hexdigest(),
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "circuit": circuit.stats(),
        "merge": meta,
        "exact": exact,
        "instruction_count": len(observed["trace"]),
        "handoff_count": len(observed["transfers"]),
        "seconds": round(time.monotonic() - start, 3),
        **observed,
    }
    if not exact:
        report["expected"] = expected
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "exact",
                    "instruction_count",
                    "handoff_count",
                    "outcome",
                    "seconds",
                )
            }
        ),
        flush=True,
    )
    if not exact or (not args.allow_step_limit and observed["outcome"] != "halted"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
