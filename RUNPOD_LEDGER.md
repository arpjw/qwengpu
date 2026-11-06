# QwenGPU RunPod ledger

The approved project cap is $74 in total. Check posted pod billing before
provisioning and reserve enough for any still pending estimates. Quotes and
elapsed-time estimates are separate from posted billing. The last refresh
was November 5, 2026 at approximately 8:04 p.m. Pacific.

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
| `3f8xz3lqh43ja8` | Resident data-dependent loop validation | $0.198139 | |
| `96stm71qzgxsm2` | Cooperative kernel and launcher validation | $0.267857 | |
| `4vqekx2tdst3uz` | Resident warp programs and launcher validation | $0.315822 | |
| `mimr8xs3ebeqeb` | Resident warp banks and launcher validation | $0.339075 | |
| `8ntzzjatpnm2q4` | Resident changed-only state validation | $0.344076 | |
| `yr7ft6o0sdf34z` | Model-generated state read plan validation | $0.377327 | |
| `pyh42o27vq2rux` | Selective state and store read validation | $0.375245 | |
| `izc2or1pmn2hyu` | Same-checkpoint transport benchmark | $0.597566 | |
| `ufiwd1ajdkif9d` | Transport selector and store-reload benchmark | $0.393304 | |
| `4usv1zb1tssh0w` | Programmable row-pair extent and launcher validation | $0.411953 | |
| `yvpz8kd8flarzs` | Resident row windows on merged 9B | $0.228612 | |
| `mj8lm4qnjz9kcb` | Composed 16×16 multiply on merged 9B | $0.259879 | |
| `uv3u7eqvj0xh65` | Masked ADD32 on standalone CUDA and merged 9B | $0.175184 | |
| `tzsvs2wb4ztt3w` | Resident high register banks on merged 9B | $0.204695 | $0.04 remaining |
| `pg2ix4z275np5a` | Resident ADD32I and mixed tiled kernel on merged 9B | Pending | $0.27 |
| `rg0clhnajboc7f` | Composed multiply and mixed kernel on one ADD32 merge | Pending | $0.35 |
| `xu7m4p6hee9a0w` | Full resident ADD32 qualification on merged 9B | Pending | $0.64 |
| `pjb7usdcbldw80` | Fresh-cache text comparison for the unified ADD32 merge | Pending | $0.20 |

Total posted is about $10.71. With pending estimates, tracked spend is
about $12.21. All project pods are terminated and the live pod list was
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

The resident-delta A100 ran from about 4:22:53 p.m. to 4:34:20 p.m.
Pacific at $1.79/hour. The merged suite and JSON launcher passed.
The pod was terminated after copying both reports. RunPod showed an
empty pod list and no posted record for this pod at teardown.

The resident-plan A100 ran from about 4:37:43 p.m. to 4:50:17 p.m.
Pacific at $1.79/hour. Both checks passed. The pod was terminated after
copying both reports. RunPod showed an empty pod list and no posted
record for this pod at teardown.

The resident I/O A100 ran from about 4:53:15 p.m. to 5:05:42 p.m.
Pacific at $1.79/hour. Both checks passed. The pod was terminated after
copying both reports. RunPod showed an empty pod list and no posted
record for this pod at teardown.

The benchmark A100 ran from about 5:07:38 p.m. to 5:27:30 p.m.
Pacific at $1.79/hour. All 18 timed launches passed. The pod was
terminated after copying the report and host record. RunPod showed
an empty pod list and no posted record for this pod at teardown.

The selector benchmark A100 ran from about 5:29:46 p.m. to 5:42:50 p.m.
Pacific at $1.79/hour. All launcher checks and 18 timed launches passed.
The pod was terminated after copying the evidence. RunPod showed an
empty pod list and no posted record for this pod at teardown.

The resident grid A100 ran from about 5:48:28 p.m. to 6:02:12 p.m.
Pacific at $1.79/hour. All 15 merged fixtures and all three launcher
transports passed. The elapsed compute estimate is about $0.41,
with $0.42 reserved including disk. The pod was terminated after
the reports were copied. RunPod showed an empty pod list and no
posted record for this pod at teardown.

The resident windows A100 ran from about 6:05:34 p.m. to 6:13:11 p.m.
Pacific at $1.79/hour. Five evaluator cases and the fixed-transport
launcher check passed. Elapsed compute was about $0.23, with $0.24
reserved including disk. The pod was terminated after copying the
reports. RunPod showed an empty pod list and no posted record for
this pod at teardown.

The composed multiply A100 ran from about 6:20:05 p.m. to 6:28:42 p.m.
Pacific at $1.79/hour. Both accumulation passes matched the scalar
reference. Elapsed compute was about $0.26, with $0.27 reserved including
disk. The pod was terminated after the reports were copied. RunPod
showed an empty pod list and no posted record for this pod at teardown.

The ADD32 A100 ran from about 6:36:42 p.m. to 6:42:32 p.m. Pacific
at $1.79/hour. The standalone CUDA and merged 9B cases matched MPS.
Elapsed compute was about $0.17, with $0.18 reserved including disk.
The pod was terminated after copying the reports. RunPod showed an
empty pod list and no posted record for this pod at teardown.

The resident wide A100 ran from about 6:53:38 p.m. to 7:01:18 p.m.
Pacific at $1.79/hour. Three evaluator cases and the fixed and
state-only launcher checks passed. Elapsed compute was about $0.23,
with $0.24 reserved including disk. The pod was terminated after
copying the reports. RunPod showed an empty pod list and no posted
record for this pod at teardown.

The resident ADD32I A100 ran from about 7:06:42 p.m. to 7:15:33 p.m.
Pacific at $1.79/hour. Four evaluator cases and both explicit
launcher checks passed. Elapsed compute was about $0.26, with $0.27
reserved including disk. The pod was terminated after copying the
reports. RunPod showed an empty pod list and no posted record for
this pod at teardown.

The ADD32 composed multiply A100 ran from about 7:19:47 p.m. to
7:31:03 p.m. Pacific at $1.79/hour. Both multiply passes and the
cooperative ADD32I replay passed on one merged directory. Elapsed
compute was about $0.34, with $0.35 reserved including disk. The pod
was terminated after copying the reports. RunPod showed an empty
pod list and no posted record for this pod at teardown.

The full qualification A100 ran from about 7:34:25 p.m. to 7:55:18 p.m.
Pacific at $1.79/hour. All 23 merged cases matched MPS. The pinned
dependency install was slower than on previous pods. Elapsed compute
was about $0.62, with $0.64 reserved including disk. The pod was
terminated after the reports were copied. RunPod showed an empty pod
list and no posted record for this pod at teardown.

The unified text comparison A100 ran from about 7:58:07 p.m. to
8:04:34 p.m. Pacific at $1.79/hour. Both model measurements completed.
Elapsed compute was about $0.19, with $0.20 reserved including disk.
The pod was terminated after copying the reports. RunPod showed an
empty pod list and no posted record for this pod at teardown.
