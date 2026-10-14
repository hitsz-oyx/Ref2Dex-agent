---
schema: ref2dex.probe.v2
probe_id: P-20261014-wam-residual-basin
experiment_id: P-20261014-wam-residual-basin
date: 2026-10-14
task: trajectory-policy
branch: main
git_commit: 96186ec3
claim_id: C3
hypothesis_family: HF-wam-residual-ppo
probe_index_in_family: 3
seed_pool: probe
seeds: [292, 296]
decision_changed_if_positive: prioritize targeted finger exploration or a warm-start bias on the actual startup-contact states
decision_changed_if_negative: do not close residual RL from four directions; prioritize actual visited-state coverage and controller geometry
status: UNCLEAR
run_id: wam-ref2-execution-basin-20261014-r2
---

# Frozen finger authority screen after the corrected PPO budget/course probe

Serve Objective A and decide whether a few explicit finger corrections on
the exact frozen WAM+old Chunk-R pair can form startup hold45. Corrected
PPO reached100 updates under the contact course without a strict frame0
hold. Its teacher-contact starts improve, but self-visited contact poses
remain different and initial thumb-pitch exploration is only .0056 rad.
This is a Decision Probe, not a convergence or RL-impossibility claim.

Reuse the saved measured prefix/provenance from contact B2 final.pt at100.
The learned residual actor mean is bypassed; candidate0 is exact zero
residual on the frozen WAM+R. No gradient update, expert rollout intervention,
World input, R swap or model-weight change occurs. The launcher reuses the
audited native runner, captures native PD/command, and independently asserts
the complete recorded residual schedule. final.pt is only an unchanged copy
of the parent network/training state, not a learned intervention policy.

| candidate | latent correction during controls40:160 |
| --- | --- |
| base | zero |
| thumb_pitch_minus | coordinate11=-4 |
| thumb_pitch_plus | coordinate11=+4 |
| finger_close | coordinates6:12=+2 |

Execute clip(base+.15*delta) on12 active native coordinates. Corrections to
thumb pitch are ±.165 rad before clipping; finger-close adds .24 rad to four
fingers, .1725 rad thumb yaw and .0825 rad thumb pitch. These bounded native
commands intentionally test directions beyond the initial Gaussian's usual
finger samples. Wrist residuals remain zero. The intervention ends at160;
all copies then continue the frozen base to the full542 endpoint.

Decision Note: spend one four-env forward-only native run, GPU6 if free,
734 batched RPCs including expanded zero-residual diagnostics, expected3–5
minutes and<30 MiB output. Native hard cap900s plus120s server readiness.
Abort on the existing context/native/GPU safety checks or a residual schedule
mismatch. If a finger candidate has hold45 while base does not, retain a
PROMISING authority/exploration direction and test it on actual visited
states before another long PPO block. Otherwise record UNCLEAR for these
four directions; this sparse screen cannot exclude other wrist/finger/time
corrections. No external authorization boundary is expanded.

Limitations/future evidence: one reference, one copy per candidate, GPU
physics variation, potential clipping, a fixed intervention window and no
claim of learned-policy success or controlled placement. The parent actor
is deliberately bypassed, so this is an exact-base intervention screen,
not an A/B evaluation of the100-update learned residual. Matched repeats
and broader feasible-direction scans are deferred unless a positive signal
changes the next decision.

The first attempt r1 stopped before any candidate evaluation: a generic
training gate rejected an extra phase0 diagnostic copy (env2,30.60 mm),
while the actual baseline-control copy (env0) reached0.77 mm. This frozen
probe only executes frame0; retain all diagnostic traces but require contact
only for phase0/env0, its actual baseline control. Training runs retain their
full phase gate. This is a scoped harness repair, not removal of a failed
candidate result. The original FAILED run is retained.

r2 ran at `96186ec3` with its phase0-only gate and passed both recorded
phase0 copies. The later `04db1415` refinement limits future forward-only
gates to the actual baseline-control copy; it does not change any candidate
action, PPO gate or the completed r2 trace.

## Completed frozen screen

GPU6,152.1s after readiness,734 RPCs,18820 MiB. No parameter changed and
no PPO step occurred. All542 recorded commands match clip(base+.15*delta),
and the complete40:160 residual schedule independently matches the declared
matrix. See frozen_audit.json in the run output; related contracts20 pass.

| candidate | maximum strict held | highest center rise | mean active-DOF saturation |
| --- | ---: | ---: | ---: |
| base | 0 | 32.86 mm | 0% |
| thumb_pitch_minus | 0 | 19.28 mm | .108% |
| thumb_pitch_plus | 0 | 51.21 mm | .431% |
| finger_close | 0 | 4.70 mm | .938% |

Saturation averages include controls outside the120-control intervention
window; inspect eval.npz for per-time/per-joint authority. These copies did
not form a stable grasp; center rise alone does not imply off-table held.
The runtime helper labeled this run UNPROMISING. The card remains UNCLEAR
for the broader basin hypothesis because four fixed directions and one copy
each do not cover wrist correction, timing or actual-state exploration.

No repeat or training was started. Next priority is actual visited-contact
state coverage and wrist/finger coordination, rather than simply closing
all fingers or spending more updates on unchanged teacher starts.
