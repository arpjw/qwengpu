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
    commit = api.upload_folder(
        folder_path=str(args.package),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.message,
    )
    after = api.model_info(args.repo_id, revision=commit.oid)
    if after.private is not True or after.sha != commit.oid:
        raise RuntimeError("uploaded revision or private visibility did not verify")
    print(json.dumps({"repo_id": args.repo_id, "revision": after.sha,
                      "private": after.private}, sort_keys=True))


if __name__ == "__main__":
    main()
