"""Probe branch-to-branch Hub copy and delete using disposable private refs.

This small integration check leaves the model's main branch unchanged and
deletes both probe branches. It does not upload or validate model weights.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)


PROBE_FILE = "qwengpu-promotion-probe.txt"
STALE_FILE = "qwengpu-promotion-stale.txt"
PROBE_BYTES = b"Qwen-GPU promotion probe\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="existing private model repository")
    args = parser.parse_args()
    api = HfApi()
    before = api.model_info(args.repo_id)
    if before.private is not True:
        parser.error("probe requires a private model repository")
    suffix = uuid.uuid4().hex[:10]
    stage_branch = f"codex-v4-probe-stage-{suffix}"
    target_branch = f"codex-v4-probe-target-{suffix}"
    created: list[str] = []
    try:
        for branch in (stage_branch, target_branch):
            api.create_branch(args.repo_id, branch=branch, revision=before.sha)
            created.append(branch)
        stage = api.create_commit(
            repo_id=args.repo_id, repo_type="model", revision=stage_branch,
            parent_commit=before.sha, commit_message="Probe staged copy",
            operations=[CommitOperationAdd(path_in_repo=PROBE_FILE,
                                           path_or_fileobj=PROBE_BYTES)],
        )
        target = api.create_commit(
            repo_id=args.repo_id, repo_type="model", revision=target_branch,
            parent_commit=before.sha, commit_message="Probe stale target",
            operations=[CommitOperationAdd(path_in_repo=STALE_FILE,
                                           path_or_fileobj=b"remove me\n")],
        )
        promoted = api.create_commit(
            repo_id=args.repo_id, repo_type="model", revision=target_branch,
            parent_commit=target.oid, commit_message="Probe staged promotion",
            operations=[
                CommitOperationCopy(src_path_in_repo=PROBE_FILE,
                                    path_in_repo=PROBE_FILE,
                                    src_revision=stage.oid),
                CommitOperationDelete(path_in_repo=STALE_FILE),
            ],
        )
        files = set(api.list_repo_files(args.repo_id, revision=promoted.oid))
        copied = Path(hf_hub_download(args.repo_id, PROBE_FILE,
                                      revision=promoted.oid)).read_bytes()
        if copied != PROBE_BYTES or STALE_FILE in files:
            raise RuntimeError("staged copy or stale-file deletion failed")
        if api.model_info(args.repo_id).sha != before.sha:
            raise RuntimeError("model main changed during the probe")
    finally:
        for branch in reversed(created):
            api.delete_branch(args.repo_id, branch=branch)

    remaining = {branch.name for branch in api.list_repo_refs(args.repo_id).branches}
    if stage_branch in remaining or target_branch in remaining:
        raise RuntimeError("probe branch cleanup did not verify")
    if api.model_info(args.repo_id).sha != before.sha:
        raise RuntimeError("model main changed after probe cleanup")
    print(json.dumps({
        "repo_id": args.repo_id,
        "main_sha": before.sha,
        "stage_sha": stage.oid,
        "target_before_sha": target.oid,
        "target_after_sha": promoted.oid,
        "source_branch": stage_branch,
        "target_branch": target_branch,
        "copied_small_file": True,
        "deleted_stale_file": True,
        "main_unchanged": True,
        "temporary_branches_deleted": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
