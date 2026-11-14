"""Upload a verified package to an existing Hugging Face model repository.

The destination must already have the requested visibility. This helper
does not publish a private repository or change its access settings.
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

if __package__:
    from .check_checkpoint_package import CHECKSUMS, package_hashes, sha256
else:
    from check_checkpoint_package import CHECKSUMS, package_hashes, sha256


def verify_package(package: Path, pinned_manifest: Path, expected_digest: str) -> set[str]:
    entries = list(package.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise ValueError("package must contain only regular files")
    checksums = package / CHECKSUMS
    if sha256(checksums) != expected_digest:
        raise ValueError("package checksum digest differs")
    if checksums.read_text(encoding="utf-8") != package_hashes(package, pinned_manifest):
        raise ValueError("one or more package files differ from SHA256SUMS")
    return {path.name for path in entries}


def check_visibility(private: bool, public: bool) -> None:
    if private is not (not public):
        expected = "public" if public else "private"
        raise ValueError(f"destination model repository must already be {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--pinned-manifest", type=Path, required=True)
    parser.add_argument("--expected-checksum-digest", required=True)
    parser.add_argument("--public", action="store_true",
                        help="require an already public destination; never changes visibility")
    parser.add_argument("--revision",
                        help="existing staging branch; required and non-main for public uploads")
    parser.add_argument("--message", default="Upload qualified Qwen-GPU checkpoint")
    args = parser.parse_args()
    if not args.package.is_dir():
        parser.error(f"package directory does not exist: {args.package}")
    token = sys.stdin.readline().strip()
    if not token:
        parser.error("provide a Hugging Face token on standard input")
    if args.public and (not args.revision or args.revision in {"main", "refs/heads/main"}):
        parser.error("public upload requires an existing non-main staging branch")
    api = HfApi(token=token)
    main_before = api.model_info(args.repo_id) if args.public else None
    if main_before is not None:
        try:
            check_visibility(main_before.private, True)
        except ValueError as exc:
            parser.error(str(exc))
        branches = {branch.name for branch in api.list_repo_refs(args.repo_id).branches}
        if args.revision not in branches:
            parser.error("public staging branch does not exist")
    before = api.model_info(args.repo_id, revision=args.revision)
    try:
        check_visibility(before.private, args.public)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        package_files = verify_package(args.package, args.pinned_manifest,
                                       args.expected_checksum_digest)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    remote_files = set(api.list_repo_files(args.repo_id, revision=args.revision))
    stale_files = sorted(remote_files - package_files - {".gitattributes"})
    commit = api.upload_folder(
        folder_path=str(args.package),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.message,
        delete_patterns=stale_files or None,
        parent_commit=before.sha,
        revision=args.revision,
    )
    after = api.model_info(args.repo_id, revision=commit.oid)
    check_visibility(after.private, args.public)
    if after.sha != commit.oid:
        raise RuntimeError("uploaded revision did not verify")
    after_files = set(api.list_repo_files(args.repo_id, revision=commit.oid))
    if after_files - {".gitattributes"} != package_files:
        raise RuntimeError("uploaded revision does not match the flat package file list")
    if main_before is not None and api.model_info(args.repo_id).sha != main_before.sha:
        raise RuntimeError("default branch changed during staging")
    print(json.dumps({"repo_id": args.repo_id, "revision": after.sha,
                      "private": after.private, "files": len(package_files),
                      "staging_branch": args.revision if args.public else None,
                      "package_checksum_digest": args.expected_checksum_digest,
                      "removed_stale_files": stale_files}, sort_keys=True))


if __name__ == "__main__":
    main()
