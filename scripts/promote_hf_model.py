"""Prepare or atomically promote a verified Hub staging revision to main.

Upload and clean-restore the staging revision before using --apply. This
helper checks the staged checksum list and file names; it does not replace
the independent byte-level download and GPU qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from huggingface_hub import (
    CommitOperationCopy,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)

if __package__:
    from .check_checkpoint_package import REQUIRED
else:
    from check_checkpoint_package import REQUIRED


def parse_checksums(contents: str) -> set[str]:
    names = set()
    for line in contents.splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid staged SHA256SUMS line")
        if not name or name in {".", "..", "SHA256SUMS"} or Path(name).name != name:
            raise ValueError(f"unsafe staged package file: {name!r}")
        if name in names:
            raise ValueError(f"duplicate staged package file: {name}")
        names.add(name)
    if not REQUIRED <= names or not any(name.endswith(".safetensors") for name in names):
        raise ValueError("staged checksum list lacks required model files")
    return names | {"SHA256SUMS"}


def promotion_operations(stage_files: set[str], main_files: set[str],
                         stage_sha: str):
    stale = sorted(main_files - stage_files - {".gitattributes"})
    operations = [CommitOperationCopy(src_path_in_repo=name,
                                      path_in_repo=name,
                                      src_revision=stage_sha)
                  for name in sorted(stage_files)]
    operations.extend(CommitOperationDelete(path_in_repo=name) for name in stale)
    if len(operations) > 100:
        raise ValueError("promotion exceeds the 100-operation review limit")
    return operations, stale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("staging_revision", help="immutable 40-character Hub commit SHA")
    parser.add_argument("expected_checksum_digest")
    parser.add_argument("--apply", action="store_true",
                        help="create one public main-branch commit after review")
    parser.add_argument("--message", default="Publish qualified Qwen-GPU v4")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.staging_revision):
        parser.error("staging revision must be an immutable commit SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_checksum_digest):
        parser.error("expected checksum digest must be SHA-256")
    token = sys.stdin.readline().strip()
    if not token:
        parser.error("provide a Hugging Face token on standard input")

    api = HfApi(token=token)
    main_before = api.model_info(args.repo_id)
    stage = api.model_info(args.repo_id, revision=args.staging_revision)
    if main_before.private is not False or stage.private is not False:
        parser.error("promotion requires an already public model repository")
    if stage.sha != args.staging_revision:
        parser.error("staging revision did not resolve to the requested commit")
    checksums = Path(hf_hub_download(args.repo_id, "SHA256SUMS",
                                     revision=args.staging_revision, token=token))
    contents = checksums.read_bytes()
    if hashlib.sha256(contents).hexdigest() != args.expected_checksum_digest:
        parser.error("staging checksum digest differs from the qualified package")
    try:
        stage_files = parse_checksums(contents.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    actual_stage = set(api.list_repo_files(args.repo_id, revision=args.staging_revision))
    if actual_stage - {".gitattributes"} != stage_files:
        parser.error("staging file list differs from SHA256SUMS")
    main_files = set(api.list_repo_files(args.repo_id))
    operations, stale = promotion_operations(stage_files, main_files,
                                             args.staging_revision)
    report = {
        "repo_id": args.repo_id,
        "staging_revision": args.staging_revision,
        "main_before": main_before.sha,
        "package_checksum_digest": args.expected_checksum_digest,
        "package_files": len(stage_files),
        "stale_main_files": stale,
        "commit_operations": len(operations),
        "applied": args.apply,
    }
    if args.apply:
        if api.model_info(args.repo_id).sha != main_before.sha:
            raise RuntimeError("main changed after promotion preflight")
        commit = api.create_commit(repo_id=args.repo_id, repo_type="model",
                                   revision="main", parent_commit=main_before.sha,
                                   operations=operations, commit_message=args.message)
        after = api.model_info(args.repo_id, revision=commit.oid)
        if after.private is not False or after.sha != commit.oid:
            raise RuntimeError("promoted revision or public visibility did not verify")
        if set(api.list_repo_files(args.repo_id, revision=commit.oid)) - {".gitattributes"} != stage_files:
            raise RuntimeError("promoted file list differs from staging")
        report["main_after"] = commit.oid
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
