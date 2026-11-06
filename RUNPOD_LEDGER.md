# QwenGPU RunPod ledger

The approved project cap is $74 in total. Check posted pod billing before
provisioning and reserve enough for any still pending estimates. Quotes and
elapsed-time estimates are separate from posted billing. The last refresh
was November 5, 2026 at approximately 4:19 p.m. Pacific.

| Pod | Work | Posted | Estimate when pending |
| --- | --- | ---: | ---: |
| `fwn759qm8qlepr` | Initial build | $2.202258 | |
| `iteq656mks0tq6` | ROM integration | $0.480742 | |
| `f1p3hnt69n3t7n` | Failed writeback pod boot | $0.213247 | |
| `dhifh5im6qr6ql` | Writeback validation | $0.430302 | |
| `6uboyzo7zyt0ii` | Control validation | $0.743543 | |
| `guroiogxqgl6g7` | Long execution handoff | $0.556100 | |
| `yj3zp8mnqzzjft` | Indexed gather and multiply | $0.422576 | |
| `08m7vfhv4m2r26` | Cooperative workgroups | $0.051943 | |
| `ttbmr62niu7yjw` | 4×4 tile engine | $0.150611 | |
| `dq1b7ch0c20s5w` | 8×8 tile engine | $0.268307 | |
| `0an76y516vonqw` | Resident and indexed memory validation | $0.331857 | |
| `4423pqr6lhu0mz` | Resident lane mask validation | $0.188049 | |
| `cdlyro9kdlgyjp` | Resident nested divergence validation | $0.183926 | |
| `3f8xz3lqh43ja8` | Resident data-dependent loop validation | Pending | $0.28 |
| `96stm71qzgxsm2` | Cooperative kernel and launcher validation | Pending | $0.27 |
| `4vqekx2tdst3uz` | Resident warp programs and launcher validation | Pending | $0.31 |
| `mimr8xs3ebeqeb` | Resident warp banks and launcher validation | Pending | $0.34 |

Total posted is about $6.22. With pending estimates, tracked spend is
about $7.42. All project pods are terminated and the live pod list was
empty at the last check. Refresh pending rows before adding another
pod, then record the new pod's quote, timestamps, outcome, and teardown.

The A100 for resident and indexed validation ran from about 2:34:30 p.m.
to 2:49:33 p.m. Pacific at $1.79/hour. Both merged Qwen3.5-9B suites
passed. The pod was terminated after the reports were copied, and
RunPod showed an empty pod list and no posted record for it.

The lane mask A100 ran from about 2:54:15 p.m. to 3:02 p.m. Pacific at
$1.79/hour. The merged Qwen3.5-9B suite passed. The pod was terminated
after copying the report. RunPod showed an empty pod list and no posted
record for this pod at teardown.

The divergent A100 ran from about 3:09:23 p.m. to 3:17:08 p.m. Pacific
at $1.79/hour. The merged Qwen3.5-9B suite passed. The pod was terminated
after copying the report. RunPod showed an empty pod list and no posted
record for this pod at teardown.

The looping A100 ran from about 3:23:51 p.m. to 3:33:05 p.m. Pacific
at $1.79/hour. The merged Qwen3.5-9B suite passed. The pod was terminated
after copying the report. RunPod showed an empty pod list and no posted
record for this pod at teardown.

The cooperative kernel and launcher A100 ran from about 3:36:54 p.m.
to 3:45:48 p.m. Pacific at $1.79/hour. Both merged 9B checks passed.
The pod was terminated after copying the reports. RunPod showed an
empty pod list and no posted record for this pod at teardown.

The resident-ROM A100 ran from about 3:52:59 p.m. to 4:03:29 p.m.
Pacific at $1.79/hour. The merged suite and JSON launcher passed.
The pod was terminated after copying both reports. RunPod showed an
empty pod list and no posted record for this pod at teardown.

The resident-state A100 ran from about 4:08:08 p.m. to 4:19:25 p.m.
Pacific at $1.79/hour. The merged suite and JSON launcher passed.
The pod was terminated after copying both reports. RunPod showed an
empty pod list and no posted record for this pod at teardown.
