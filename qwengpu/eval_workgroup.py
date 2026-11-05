"""Launch two warps on one model and verify the cooperative execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import transformers
from chipformer.q35.ir import Target

from qwengpu.compiled_control import (
    EXPORT_BASE,
    STATE_BASE,
    STATE_COUNT,
    TICK,
    build,
    build_standalone,
    program_tokens,
    state_token,
)
from qwengpu.control_reference import ControlState, parse_kernel, snapshot_bits, step
from qwengpu.eval_control import hex_bits, snapshot
from qwengpu.eval_resume import check_checkpoint, checkpoint_tokens
from qwengpu.reference import Instruction, WarpState

FIXTURES = Path(__file__).resolve().parents[1] / "eval/frozen/workgroup_gate_v1"
SHARED = slice(8, 40)


@dataclass(frozen=True)
class Workgroup:
    programs: tuple[tuple[Instruction, ...], ...]
    initial: tuple[ControlState, ...]
    expected_outcome: str


def parse_workgroup(data: object) -> Workgroup:
    if not isinstance(data, dict) or not isinstance(data.get("warps"), list):
        raise ValueError("workgroup must contain a warps list")
    if len(data["warps"]) != 2:
        raise ValueError("workgroup requires exactly two warps")
    scratch = data.get("shared", [0] * 32)
    if (
        not isinstance(scratch, list)
        or len(scratch) != 32
        or any(type(value) is not int or not 0 <= value <= 255 for value in scratch)
    ):
        raise ValueError("shared must contain 32 unsigned bytes")
    outcome = data.get("expected_outcome", "halted")
    if outcome not in ("halted", "deadlock", "step_limit"):
        raise ValueError("invalid expected_outcome")
    kernels = [parse_kernel(raw, workgroup=True) for raw in data["warps"]]
    states = tuple(
        ControlState(WarpState(lanes=lanes, scratch=bytes(scratch) + bytes(32)))
        for lanes, _ in kernels
    )
    return Workgroup(tuple(program for _, program in kernels), states, outcome)


@torch.no_grad()
def execute(
    model, circuit, group: Workgroup, device: str, max_steps: int, offset: int = 0
) -> dict:
    rom = [program_tokens(program, workgroup=True) for program in group.programs]
    checkpoints = [
        checkpoint_tokens(snapshot_bits(state, extended=True))
        for state in group.initial
    ]
    shared = checkpoints[0][SHARED]
    control = [state_token(0, 0)] * 2
    status = ["ready"] * 2
    trace = []
    releases = []
    turn = 0
    export = list(range(EXPORT_BASE, EXPORT_BASE + len(checkpoints[0])))

    while len(trace) < max_steps:
        if status == ["halted", "halted"]:
            break
        if "ready" not in status:
            break
        warp = turn if status[turn] == "ready" else 1 - turn
        turn = 1 - warp
        upload = checkpoints[warp][:]
        upload[SHARED] = shared
        # The single causal forward writes the stage snapshot at TICK and
        # reads it at the following export positions in the same sequence.
        inputs = rom[warp] + upload + [control[warp], TICK, *export]
        output = model(
            input_ids=torch.tensor(
                [[offset + token for token in inputs]], device=device
            ),
            use_cache=True,
            logits_to_keep=1 + len(export),
        )
        tokens = [token - offset for token in output.logits[0].argmax(dim=-1).tolist()]
        generated, *checkpoint = tokens
        raw = generated - STATE_BASE
        if not 0 <= raw < STATE_COUNT or raw & 0xF0 or raw >> 8 == 2:
            raise ValueError(f"invalid workgroup control token {generated}")
        check_checkpoint(checkpoint, circuit.snapshot_bits)
        checkpoints[warp] = checkpoint
        shared = checkpoint[SHARED]
        control[warp] = generated
        status[warp] = {0: "ready", 1: "halted", 3: "waiting"}[raw >> 8]
        trace.append(
            {
                "warp": warp,
                "control": generated,
                "checkpoint": checkpoint,
                "snapshot": hex_bits(snapshot(output.past_key_values, circuit, 0)),
            }
        )
        if status == ["waiting", "waiting"]:
            releases.append(len(trace))
            status = ["ready"] * 2

    outcome = (
        "halted"
        if status == ["halted", "halted"]
        else ("deadlock" if "ready" not in status else "step_limit")
    )
    return {"trace": trace, "releases": releases, "outcome": outcome}


def oracle(group: Workgroup, max_steps: int) -> dict:
    states = list(group.initial)
    shared = states[0].warp.scratch
    waiting = set()
    trace = []
    releases = []
    while len(trace) < max_steps:
        ran = False
        for warp, program in enumerate(group.programs):
            state = states[warp]
            if len(trace) == max_steps or state.warp.halted or warp in waiting:
                continue
            ran = True
            op = program[state.warp.pc].op
            state = step(
                replace(state, warp=replace(state.warp, scratch=shared)), program
            )
            states[warp] = state
            shared = state.warp.scratch
            if op == "BARRIER":
                waiting.add(warp)
            flags = 3 if op == "BARRIER" else int(state.warp.halted)
            bits = snapshot_bits(state, extended=True)
            trace.append(
                {
                    "warp": warp,
                    "control": STATE_BASE + state.warp.pc + (flags << 8),
                    "checkpoint": checkpoint_tokens(bits),
                    "snapshot": hex_bits(bits),
                }
            )
            if len(waiting) == 2:
                waiting.clear()
                releases.append(len(trace))
        if not ran:
            break
    if all(state.warp.halted for state in states):
        outcome = "halted"
    elif all(state.warp.halted or warp in waiting for warp, state in enumerate(states)):
        outcome = "deadlock"
    else:
        outcome = "step_limit"
    return {"trace": trace, "releases": releases, "outcome": outcome}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path)
    parser.add_argument("--kernel", type=Path, help="run one workgroup manifest")
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    paths = [args.kernel] if args.kernel else sorted(FIXTURES.glob("*.json"))
    cases = []
    for path in paths:
        payload = path.read_bytes()
        cases.append(
            (
                path.name,
                hashlib.sha256(payload).hexdigest(),
                parse_workgroup(json.loads(payload)),
            )
        )
    if not cases:
        raise ValueError("no workgroup manifests found")

    meta = None
    offset = 0
    if args.model:
        meta = json.loads((args.model / "qwengpu9b.json").read_text())
        if meta["layout"] != "qwengpu-workgroup9b-v1":
            raise ValueError("not a workgroup checkpoint")
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
            workgroup=True,
        )
    else:
        model, circuit = build_standalone(workgroup=True)
    model.to(args.device).eval()
    report = {
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "circuit": circuit.stats(),
        "merge": meta,
        "cases": [],
    }
    for name, digest, group in cases:
        expected = oracle(group, args.max_steps)
        observed = execute(model, circuit, group, args.device, args.max_steps, offset)
        exact = observed == expected and observed["outcome"] == group.expected_outcome
        result = {
            "fixture": name,
            "fixture_sha256": digest,
            "exact": exact,
            "instructions": len(observed["trace"]),
            **observed,
        }
        final_r0 = [list(state.warp.lanes) for state in group.initial]
        shared = list(group.initial[0].warp.scratch[:32])
        for entry in observed["trace"]:
            checkpoint = entry["checkpoint"]
            final_r0[entry["warp"]] = [token & 255 for token in checkpoint[:8]]
            shared = [token & 255 for token in checkpoint[SHARED]]
        result.update(final_r0=final_r0, final_shared=shared)
        if not exact:
            result["expected"] = expected
        report["cases"].append(result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "fixture",
                        "exact",
                        "instructions",
                        "outcome",
                        "releases",
                    )
                }
            ),
            flush=True,
        )
    report["all_exact"] = all(case["exact"] for case in report["cases"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
