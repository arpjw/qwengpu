"""Upload a qualified model package to an existing private Hugging Face repo."""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--message", default="Upload qualified Qwen-GPU checkpoint")
    args = parser.parse_args()
    if not args.package.is_dir():
        parser.error(f"package directory does not exist: {args.package}")
    token = sys.stdin.readline().strip()
    if not token:
        parser.error("provide a Hugging Face token on standard input")

    api = HfApi(token=token)
    before = api.model_info(args.repo_id)
    if before.private is not True:
        parser.error("destination model repository must be private")
    package_files = {path.name for path in args.package.iterdir() if path.is_file()}
    remote_files = set(api.list_repo_files(args.repo_id))
    stale_files = sorted(remote_files - package_files - {".gitattributes"})
    commit = api.upload_folder(
        folder_path=str(args.package),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.message,
        delete_patterns=stale_files or None,
    )
    after = api.model_info(args.repo_id, revision=commit.oid)
    if after.private is not True or after.sha != commit.oid:
        raise RuntimeError("uploaded revision or private visibility did not verify")
    after_files = set(api.list_repo_files(args.repo_id, revision=commit.oid))
    if after_files - {".gitattributes"} != package_files:
        raise RuntimeError("uploaded revision does not match the flat package file list")
    print(json.dumps({"repo_id": args.repo_id, "revision": after.sha,
                      "private": after.private, "files": len(package_files),
                      "removed_stale_files": stale_files}, sort_keys=True))


if __name__ == "__main__":
    main()
