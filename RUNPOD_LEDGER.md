# QwenGPU RunPod ledger

The approved project cap is $74 in total. Check posted pod billing before
provisioning and reserve enough for any still pending estimates. Quotes and
elapsed-time estimates are separate from posted billing. The last refresh
was November 5, 2026 at approximately 2:49 p.m. Pacific.

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
| `ttbmr62niu7yjw` | 4×4 tile engine | Pending | $0.15 |
| `dq1b7ch0c20s5w` | 8×8 tile engine | Pending | $0.27 |
| `0an76y516vonqw` | Resident and indexed memory validation | Pending | $0.45 |

Total posted is about $5.10. With pending estimates, tracked spend is
about $5.97. All project pods are terminated and the live pod list was
empty at the last check. Refresh pending rows before adding another
pod, then record the new pod's quote, timestamps, outcome, and teardown.

The A100 for resident and indexed validation ran from about 2:34:30 p.m.
to 2:49:33 p.m. Pacific at $1.79/hour. Both merged Qwen3.5-9B suites
passed. The pod was terminated after the reports were copied, and
RunPod showed an empty pod list and no posted record for it.
