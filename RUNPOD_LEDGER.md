# QwenGPU RunPod ledger

The approved project cap is $74 in total. Check posted pod billing before
provisioning and reserve enough for any still pending estimates. Quotes and
elapsed-time estimates are separate from posted billing. The last refresh
was November 7, 2026 at 05:16 UTC.

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
| `tzsvs2wb4ztt3w` | Resident high register banks on merged 9B | $0.231115 | |
| `pg2ix4z275np5a` | Resident ADD32I and mixed tiled kernel on merged 9B | $0.266326 | |
| `rg0clhnajboc7f` | Composed multiply and mixed kernel on one ADD32 merge | $0.340125 | |
| `xu7m4p6hee9a0w` | Full resident ADD32 qualification on merged 9B | $0.627930 | |
| `pjb7usdcbldw80` | Fresh-cache text comparison for the unified ADD32 merge | $0.193388 | |
| `icyks8qxc6uuwd` | Direct GPU-to-text cache handoff | $0.262291 | |
| `d77qwi6198u2f1` | Text cache reset with persistent GPU state | $0.253064 | |
| `spczei5clluhi2` | GPU-to-text-to-GPU cache relaunch | $0.202233 | |
| `e7dircqgmin263` | Two K slices with resident C | $0.214874 | |
| `ulujgufwzoo34a` | Full 16×16 multiply with resident C | $0.276951 | |
| `q07e1z7zdcgywv` | Shared32 qualification and retained launch replays | $0.705477 | |
| `7ag8r4iy5e2exr` | Addmem32 qualification and retained multiply | $0.672188 | |
| `30mh7vnuymrpdk` | Addmem32 text, cache handoff, and same-device control | $0.864872 | |
| `bkgbc6rveup6to` | Full-width arithmetic through a text handoff | $0.229964 | |
| `05o4r8o4duazn3` | Branch32 qualification and retained multiply | $1.400939 | |
| `nbqqwl22xu1sa0` | Loop32 qualification and retained multiply | $1.510754 | |
| `v6b971ophoua68` | Preloaded operand banks and inherited suite | $1.474978 | |
| `ev84g71gc7yn2n` | Model-selected K transition and inherited suite | $0.613430 | |
| `e28q01ftnq770h` | Public tilepipe multiply, text and cache integration | $0.566842 | |
| `d3vvbd3rvgwktg` | Merged tilegrid matrix, inherited cases, kernel and text qualification | $1.128544 | |
| `mcla5c112zcnxu` | Merged globalgrid IDs, reduction, and launcher checks | $0.309297 | |
| `lcs8xcxtr8ezrj` | Full globalgrid and compute qualification, larger reductions, and transport benchmark | $0.620611 | $1.88 remaining |

Total posted is about $23.80. Including the remaining reserve for unposted
charges, tracked project spend is about $25.68, leaving about $48.32 under
the approved project cap. This is a project ledger, not an account balance.
The latest QwenGPU task pod was terminated and is absent from the live
pod list. Reconcile its remaining estimate when final charges post.

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

The direct handoff A100 ran from about 8:13:30 p.m. to 8:22:13 p.m.
Pacific at $1.79/hour. The four cooperative replays and twelve text
comparisons completed. Elapsed compute was about $0.26, with $0.27
reserved including disk. The pod was terminated after copying the
reports. RunPod showed an empty pod list and no posted record for
this pod at teardown.

The cache reset A100 ran from about 8:28:15 p.m. to 8:36:39 p.m.
Pacific at $1.79/hour. Four text comparisons and all GPU row checks
passed. Elapsed compute was about $0.25, with $0.26 reserved including
disk. The pod was terminated after copying the reports. RunPod
showed an empty pod list and no posted record for this pod at teardown.

The cache relaunch A100 ran from about 8:43:40 p.m. to 8:50:24 p.m.
Pacific at $1.79/hour. The GPU, text, and retained consumer checks
passed. Elapsed compute was about $0.20, with $0.21 reserved including
disk. The pod was terminated after copying the reports. RunPod
showed an empty pod list and no posted record for this pod at teardown.

The cached tile A100 ran from about 8:58:15 p.m. to 9:05:27 p.m.
Pacific at $1.79/hour. Both K slices and the fresh-cache control
matched MPS. Elapsed compute was about $0.21, with $0.23 reserved
including disk. The pod was terminated after copying the reports.
RunPod showed an empty pod list and no posted record for this pod.

The retained 16×16 multiply A100 ran from about 9:14:15 p.m. to
9:23:28 p.m. Pacific at $1.79/hour. The eight-launch 9B report matched
MPS. Elapsed compute was about $0.28, with $0.30 reserved including
disk. The pod was terminated after copying the reports. RunPod
showed an empty pod list and no posted record for this pod.

The shared32 A100 ran from about 11:36:24 a.m. to 11:59:55 a.m.
Pacific at $1.79/hour. The merged checkpoint passed all 24 cases
and the retained multiply, two-slice tile, and cache relaunch
replays. Elapsed compute was about $0.70, with $0.73 reserved
including disk. The pod was terminated after copying results.
RunPod showed an empty pod list and no posted record for this pod.

The addmem32 A100 ran from about 12:05:41 p.m. to 12:28:06 p.m.
Pacific at $1.79/hour. The merged checkpoint matched all 25 MPS
cases and the retained 16×16 multiply. Elapsed compute was about
$0.67, with $0.70 reserved including disk. The pod was terminated
after the reports were copied. RunPod showed an empty pod list.
Both this pod and the preceding shared32 pod were still pending.

The H100 for addmem32 text ran from about 12:30:19 p.m. to
12:43:16 p.m. Pacific at $3.99/hour. The same-device ADD32 control
matched all measured text fields, and the cache checks passed.
Elapsed compute was about $0.86, with $0.89 reserved including
disk. The pod was terminated after copying all reports. RunPod
showed an empty pod list and no posted record for this pod.

The arithmetic relaunch A100 ran from about 12:47:01 p.m. to
12:54:41 p.m. Pacific at $1.79/hour. The new and legacy relaunches
passed on the pinned addmem32 merge. Elapsed compute was about
$0.23, with $0.25 reserved including disk. The pod was terminated
after copying results. RunPod showed an empty pod list and no
posted record for this pod.

The branch32 H100 NVL ran from about 1:02:51 p.m. to 1:29:07 p.m.
Pacific at $3.19/hour. The merged checkpoint passed all 26 cases
and the retained multiply. The tiled replay needed a longer
monitoring window. Elapsed compute was about $1.40, with $1.42
reserved including disk. The pod was terminated after copying
results. RunPod showed an empty pod list and no posted record for
this pod.

The loop32 H100 ran from about 1:33:59 p.m. to 1:56:38 p.m.
Pacific at $3.99/hour. The merged checkpoint passed all 27 cases
and the retained multiply. Elapsed compute was about $1.51,
with $1.54 reserved including disk. The pod was terminated after
copying results. RunPod showed an empty pod list and no posted
record for this pod.

The tilebanks H100 ran from about 2:04:04 p.m. to 2:26:11 p.m.
Pacific at $3.99/hour. The two preloaded slices and all 27
inherited cases matched MPS. Elapsed compute was about $1.47,
with $1.50 reserved including disk. The pod was terminated
after copying results. RunPod showed an empty pod list and no
posted record for this pod.

The tilepipe A100 ran from about 2:31:46 p.m. to 2:52:12 p.m.
Pacific at $1.79/hour. The one-launch transition and all 27
inherited cases matched MPS. Elapsed compute was about $0.61,
with $0.63 reserved including disk. The pod was terminated after
copying results. RunPod showed an empty pod list and no posted
record for this pod.

The final tilepipe integration A100 ran from about 2:55:32 p.m.
to 3:14:27 p.m. Pacific at $1.79/hour. The public 16×16 multiply,
native text corpus, and cache handoffs completed. Elapsed compute
was about $0.56, with $0.59 reserved including disk. The pod was
terminated after copying results. RunPod showed an empty pod
list and no posted record for this pod.

The tilegrid A100 SXM 80 GB ran from 01:50:18 to 02:27:50 UTC
on November 7 at the live rate of $1.79/hour. Elapsed compute was
about $1.12. The ledger reserves $1.20 including disk until billing
posts. The merged 9B grid, all 27 inherited cases, the assembled,
vector, reduction, and seeded cases, and the text and cache suites
completed. The pod was terminated after copying reports. RunPod
returned HTTP 204 and an empty live pod list.

The globalgrid A100 SXM 80 GB ran from 02:45:43 to 02:56:00 UTC
on November 7 at $1.79/hour. Elapsed compute was about $0.31.
The ledger reserves $0.35 including disk until billing posts.
The IDs, two-pass reduction, and public and legacy launchers
matched. The pod was terminated after copying the reports.
RunPod returned HTTP 204 and an empty live pod list.

The full qualification A100 SXM 80 GB (`lcs8xcxtr8ezrj`) ran on November 7
from 03:54:49 to 05:16:34 UTC at the quoted $1.79/hour. Its approximately
81.74 minutes imply $2.44 of compute. Reserve $2.50 total including disk;
$0.620611 has posted and about $1.88 remains reserved. It qualified
globalgrid and the unified compute checkpoint, including the common
kernels, inherited traces, large matrix, global IDs, native text, retained
cache handoffs, and repaired transport benchmark. All reports, config,
weight index, shard hashes, commands, and pinned rebuild instructions
were copied before termination. RunPod returned HTTP 204 and a fresh list
confirmed this task pod was gone. See `results/compute_runpod_cost.json`.

## 32-slot resident ROM qualification

One A100 SXM 80 GB pod at $2.19/hour ran the 9B merge, native-text gate,
two long programs, 31 compute requests, seven raw memory probes, all 27
inherited traces, the tiled 16×16 kernel, and the native-text RAM handoff.
The run finished in under 46 minutes. Estimated compute was $1.65.
No charge had posted for this pod at teardown. The $1.85 reserve includes
storage and rounding. RunPod returned HTTP 204 on deletion and HTTP 404
on the subsequent lookup. The project retains $42.47 of authorized
headroom with all reserves applied. See `results/rom32_runpod_cost.json`.

## November 13 CPU offload

The Mac's QwenGPU model runs were stopped at Arya's request. CPU3 General
Purpose pod `fszn8193hx2otq` had two vCPUs and 8 GB RAM, quoted and
confirmed at $0.08/hour before disk. It ran unit and compiler checks and
audited the merged GPU results. It was deleted after copying the reports;
the API returned HTTP 204 and the next lookup returned HTTP 404. About
nine minutes of compute cost an estimated $0.012; the invoice had not
posted at teardown. An initial pod `ezu9itqyym6bw0` at the same rate
never provided a working shell and was deleted after about a minute.
Reserve $0.05 for both CPU attempts pending billing. MPS-only validation
remains paused; CPU checks are not MPS qualification.

The ROM32 A100's posted charge later reconciled to $1.672135, including
disk. This replaces its $1.85 pending reserve; historical teardown
estimates above are retained for comparison.

## November 13 integrated integer and writeback runs

The RTX PRO 6000 Blackwell Server pod `4n7zzre8rv6qyg` ran from
22:14:32 UTC November 13 through the verified teardown at 00:48:04 UTC
November 14 at the quoted $2.69/hour. The elapsed compute estimate is
$6.88. RunPod had posted $4.113736 including disk at the immediate
recheck; this is partial, so reserve $7.20 total for the pod. The
checkpoint manifests, exact reports, and adapter checks were copied
before deletion. The API returned HTTP 204 and then HTTP 404.
Conservative project headroom is about $35.40 under the $74 cap,
including existing reserves and the two CPU attempts. This is not a
verified RunPod account balance. See `results/alu32_integrated_runpod_cost.json`.

The follow-up CPU3 General Purpose pod `id25l4k0x4kysb` ran all 46
compiler and regression tests on Linux at `632bfee`. It had two vCPUs,
8 GB RAM and 5 GB ephemeral disk, with a quoted compute rate of
$0.08/hour. RunPod created it at 00:51:05 UTC and confirmed deletion by
00:55:53 UTC on November 14 (HTTP 204 followed by HTTP 404). The compute
estimate is $0.0064, with $0.02 reserved for compute and disk. No charge
had posted on the immediate billing recheck. The transcript and setup
commands are in `results/runpod_cpu/`.

## November 14 stock control generation

Both RTX PRO 6000 Blackwell Server pods were quoted at $2.69/hour.
The first, `vaciis65zskpjl`, repeatedly failed to create its container
and was verified deleted at 01:02:16 UTC. Estimated compute was $0.091.
The replacement, `9txecrsz7qmlqk`, completed compiler, standalone CUDA,
merged CUDA, and text checks. It ran from 01:02:52 to verified deletion
at 01:11:51 UTC, for an estimated $0.402 of compute. Both deletions
returned HTTP 204 followed by HTTP 404. Neither had posted charges at
the immediate recheck. Reserve $0.60 for both, including disk. The
artifact hashes and test reports are in `results/control_generation_cuda/`.

## November 14 resident stock generation

Pod `q1i77hvq53722m` ran the initial and compact resident-generation
builds on one RTX PRO 6000 Blackwell Server GPU at $2.69/hour.
RunPod created it at 01:17:43 UTC and verified deletion by 01:57:00 UTC
(HTTP 204 followed by HTTP 404). Estimated elapsed compute was $1.76.
No charge had posted on the immediate recheck. Reserve $2.00 for
compute and disk. The checkpoint manifests and workload traces are in
`results/resident_generation_cuda/`.

Pod `769wn1axp3j7sg` rebuilt the compact generation checkpoint and
verified native-text handoff, single-workgroup memory and arithmetic, and
eight compute requests. It ran on the RTX PRO 6000 Blackwell Server at
$2.69/hour from 01:58:40 to verified deletion at 02:30:38 UTC on November
14. The API returned HTTP 204 followed by HTTP 404. Estimated compute
was $1.43. No charge had posted on the immediate recheck. Reserve $1.70
for compute and disk.

The two-vCPU CPU3 General Purpose pod `8fwsdvrqtppazx` ran all 51
Linux regression tests at `d615810` with 8 GB RAM and 5 GB ephemeral
disk. Its live rate was $0.08/hour before disk. It ran from 02:53:28
to verified deletion by 03:03:26 UTC on November 14 (HTTP 204 followed
by HTTP 404). Estimated compute was $0.013 and no charge had posted
at teardown. Reserve $0.02 for compute and disk. The transcript and
reproduction command are in `results/resident_generation_v2_cuda/`.

RTX PRO 6000 Blackwell Server pod `tr7tg4b23z1mkf` ran the sequential
v2 standalone and merged checks from 02:33:49 to verified deletion
by 03:24:45 UTC on November 14. The quoted rate was $2.69/hour before
disk. Two full emission attempts hit the host memory limit. The
layer-at-a-time build and the subsequent merged checks completed.
The API returned HTTP 204 and then HTTP 404. Approximate elapsed
compute was $2.28, with $2.65 reserved including disk. No charge had
posted at the immediate recheck. The account's pod list was empty.

CPU3 General Purpose pod `uzzlpp3ajrvfso` ran the v3 compiler replay
on two vCPUs with 8 GB RAM at $0.08/hour before disk. It ran from
03:30:33 to verified deletion by 03:44:58 UTC on November 14.
Deletion returned HTTP 204 followed by HTTP 404. Estimated compute
was $0.019 and no charge had posted. Reserve $0.03 including disk.
