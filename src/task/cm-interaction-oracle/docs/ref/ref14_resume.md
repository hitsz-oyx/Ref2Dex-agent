# Ref14_1 interrupted-run recovery

2026-10-05: resumed the user-specified session
`01a10c39-7f05-7dd3-98e5-09d8c106eeea` on the existing
`agent/cm-interaction-oracle` route. At takeover, original coordinator PID
1023057 and its simulator workers were absent; GPU6/7 were idle. The source
progress remained `ROUND_EXECUTED`, s263/g1/offset32, rather than reporting a
terminal error. The external interruption cause is unknown. Preserve the
original progress, completed outputs and incomplete offset40 attempt.

Recovery code: `a949b62`, `tools/run/resume_rolling_gt_y.py`. The original
collector, controller, labels and experiment protocol remain hash-identical
to the frozen run at `337d3a6`. Recovery keeps original scientific code identity
and separately records administrative recovery code identity. Before execution,
all frozen inputs and eight cached baseline/repeat workers passed hashes and
argument checks. Existing plans and group results must reconstruct exactly;
completed workers are reused without new simulation. Incomplete own-task
attempts move to `interrupted-attempts/` and are retained, then that worker is
rerun from the last contiguous actual prefix.

The original total cap remains7200s/4GiB, GPU6/7, four workers maximum.
This recovery conservatively charged4549.996s before new work (original last
checkpoint2865.251s plus the timestamp gap, including interruption downtime).
This is stricter than cumulative active compute; it does not grant a new budget.
The recovery manifest and `resume_progress.json` are current operational records;
`progress.json` is the immutable pre-recovery record. A cap failure or incomplete
cohort cannot become negative evidence about the controller.

At takeover, only s263/g0 had completed: baseline4/7, rolling5/7, no harmed
baseline successes, env36 rescued. This partial group does not decide the fixed
32-anchor gate. Final paired statistics and raw trace audit must wait for all
four groups. No predictor, noisy-Y scientific intervention, PPO or new replay
cohort is authorized by this recovery.

Ref14_2 runs alongside as CPU packaging, temporal diagnostics, existing-flow
geometry auditing and synthetic evaluator checks. These tools do not modify
active scientific inputs and do not spend additional physics/training budget.
