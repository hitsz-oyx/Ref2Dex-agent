# 平移抬升与旋转稳定的控制职责复盘

HF15 Decision, current session. Mission/claim unchanged, C3OPEN.
HF13 freezes all hand targets and loses clearance; HF14 preserves finger
feedback but freezes translation AND rotation. Base-finger variant reduces
clearance loss6.435pp but signed height−7.985mm; fit-best fixed remainsbase.
Post-hoc3cm retention−.644pp with widecrosszeroCI does not rescue original
UNPROMISING or establish useful recovery. All execution/force/geometry audits
pass, so another same-controller parameter/seed sweep is unwarranted.

Choice: make control responsibilities explicit. Keep expert wrist XYZ/lift and
finger feedback; anchor only native rotationDOFs3..5 at observed values.
Compare base/cup feedback variants and existing6experts under actualfullH10.
This preserves the task-progress channel removed by full-wrist freeze; test
whether stabilizing rotation can reduce loss while allowing lift. It does not
claim rotation was the proved source of HF14's risk reduction.

Same corrected raw-force/geometry schema, private9slot allocations/cohorts,
propensity merging, complete labels, and fit-selected variant/held gain/risk
gates asHF14. Engineering460/private10460, science461–472/private10461–10472,
96env650ticks;<=60min/8GiB for opportunity including engineering. Family2slots:
1actual opportunity; only if PROMISING,2new plan-conditioned physicalCm fit
and direct-control mechanism Probe, separately<=60min/8GiB. No old2+8model
reuse, no stable-grasp/PPO conclusion from engineering or local consequence.

For positive opportunity, fit actualH10 height/clearance/contact trajectories,
joint-retention and clearance-loss heads using pre-action history and explicit
feedback-program identity/rotation-anchor. Compare state-only, action-shuffled,
always-base and fit-best-fixed; fixed calibration uncertainty/abstain fromcal
only, then independent fresh randomized direct-control/reobserve data. Actual
MPC truncation/replanning must receive its own measurement, not presumed from
committedH10 forecast labels. Policy training/full stable-grasp evaluation only
after genuine local utility; task reward/greedy command changes audited.

If support missing, labelUNCLEAR with exact support, no posthoc extension.
If gate fails, close wrist-anchoring candidate branch, do not continue freezing
additional DOFs individually. Review candidate-generation/control abstraction
at a higher level before further collection. HF13/HF14 failures remain intact;
the three candidate probes test distinct control channels, not new thresholds.
Only own sim/output/PIDs, admittedGPU0/1, immutable checkpoints/assets/motions;
no new identity, external write, mission/claim or resource-boundary change.
