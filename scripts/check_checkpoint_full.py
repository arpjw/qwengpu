"""Check full inherited, public matrix, and text evidence for a newer layout."""

from __future__ import annotations

import argparse
import hashlib

from check_matmul16 import FIXTURE
from check_resident_add32_full import cases_by_name, load_report
from check_tilegrid import check
from check_tilepipe_matmul16 import check_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stem", choices=("globalgrid", "compute"), default="globalgrid"
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--text", action="store_true")
    args = parser.parse_args()
    if (args.matrix or args.text) and not args.cuda:
        parser.error("matrix and text checks require --cuda")
    stem = args.stem
    layout = f"qwengpu-resident-{stem}-9b-v1"
    baseline = cases_by_name(load_report("tilegrid_inherited_mps"))
    mps = load_report(f"{stem}_inherited_mps")
    assert len(baseline) == 27 and cases_by_name(mps) == baseline
    print(f"{stem} MPS: all 27 inherited case records match")
    if not args.cuda:
        return
    cuda = load_report(f"{stem}_inherited_9b_cuda_fla")
    assert cuda["merge"]["layout"] == layout
    assert cases_by_name(cuda) == baseline
    print(f"{stem} merged 9B CUDA: all 27 inherited case records match MPS")
    if args.matrix:
        matrix = load_report(f"{stem}_matmul16_9b_cuda_fla")
        assert (
            matrix["input_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        )
        assert matrix["merge"] == cuda["merge"]
        old = load_report("tilegrid_2x2_full_mps")
        # The public runner uses different report headers from eval_tilegrid.
        for field in (
            "matrix",
            "passes",
            "transitions",
            "transport",
            "memory_tokens_by_call",
            "instructions",
            "scheduler_decisions",
            "workgroup_outputs",
            "exact",
        ):
            assert matrix[field] == old[field], field
        # Reuse the independent scalar, boundary, and upload audit with its
        # original report header, after checking every execution field above.
        check(old, load_report("resident_tilepipe_matmul16_mps"))
        print(
            "Public 16x16 multiply: both scalar passes and all eight tile traces match MPS"
        )
    if args.text:
        check_text(cuda["merge"], stem)


if __name__ == "__main__":
    main()
