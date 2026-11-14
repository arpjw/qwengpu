# Corrected v4 restore interruption

The first clean restore stopped after verifying weight shard 16. Its
transcript is `restore-interruption.txt`; shard 17 was left partial and
there was no Python traceback. The RunPod CPU pod remained `RUNNING`
at the same address and start time. Its network volume mount reported
ample shared filesystem space through `df`, but a Python bytecode write
on `/workspace` returned `[Errno 122] Disk quota exceeded`. An `scp`
upload to that mount also failed at close. The exact quota accounting
was not exposed by those commands.

The original and corrected package trees were temporary copies. Before
removing them, the corrected package's `SHA256SUMS` and `README.md`
matched the committed digests. The private corrected draft had already
passed its independent asset manifest check. Removing both package
trees made a write probe on `/workspace` succeed. The subsequent
restore used `--resume` on the partial destination, rehashed existing
files, and redownloaded the partial shard. This preserves the separate
clean-download check without duplicating the already verified bytes.

The resumable command is the normal corrected restore command in
`RELEASE.md` with `--resume` added. `--resume` requires an existing
directory, rejects unexpected entries and symlinks, and verifies each
complete file against the release's checksum manifest before skipping it.
