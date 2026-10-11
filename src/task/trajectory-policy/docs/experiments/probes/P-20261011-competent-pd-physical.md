---
schema: ref2dex.probe.v2
probe_id: P-20261011-competent-pd-physical
experiment_id: P-20261011-competent-pd-physical
date: 2026-10-11
task: trajectory-policy
branch: main
git_commit: eaeb81f
claim_id: C3
hypothesis_family: HF-trajectory-policy-competent-pd-physical
probe_index_in_family: 1
seed_pool: probe
seeds: [182]
decision_changed_if_positive: retain learned actual-PD R and advance to a separately specified high-level joint BC
decision_changed_if_negative: distinguish source-sequence replay calibration from learned-R execution failure before high-level training
status: PROMISING
run_id: competent-pd-execution-20261011-r1
model_sha256: 414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f
---

# Does the competent PD retargeter's finite error preserve real task behavior?

## Decision Note / rationale

The original [offline card](P-20261011-competent-pd-retargeter.md) remains UNCLEAR
with a failed5mm/.05rad PD screen. Tau reduces validation MSE59%, same-tick
shuffle raises error5.535x, competent teacher qualifies32/32, and independent
actual-PD/label/model audits pass. The wrist affine diagnostic does not repair
the gap. A12.5mm PD position error is not itself a physical grasp observation.

This is a new frozen Decision: does this exact finite model still preserve the
physical task? Positive retains R for joint BC, negative diagnoses R/control
distribution, failed calibration remains UNCLEAR. No thresholds in the original
offline card change, no model selection on physical/test results, and no new fit.

## Fixed protocol and gates

Exactly the retained tau.pt with the above SHA, source teacher r2 and its held-out
envs24:32. Seed182,24GPU-pipeline envs,542 controls:8 live official teachers,
8 exact source-PD replay,8 R receiving those source actual future hand plans plus
its own live measured state every control. Pad the unavailable terminal hand
suffix with the recorded final hand; this tail distribution difference is a
limitation. Never feed R future q/object, teacher obs/action, clock or force.
Known source future tau/PD are privileged controls, not a deployed policy.

Use the original physical gates: each calibration arm and R require >=6/8 hold45
and controlled-placement successes; R median holding >=80% of source-PD control
and clipping<1%. PROMISING only if all pass. Failed calibration is UNCLEAR;
passing calibration with failed R is UNPROMISING for this checkpoint/contract.
Record all measured states, features, model output, intended/applied controls,
requested/actual PD, clipping and independent replay audit. No exact hidden-state
fork or causal policy-improvement claim.

## Resource and provenance boundaries

One idle GPU0, <=300s physics plus<=120s audit, <=256MiB outputs, entirely inside
the original card's remaining20GPUmin/1GiB budget. No new GPU/time allowance.
Stop on reset/input drift, nonfinite, GPU conflict, missing PD capture or timeout.
Stay main, no push, no external environment edits, no Campaign edits.
If the consumer runner changes, verify the source producer's code hash against
its recorded Git commit while keeping data/config/native/runtime inputs exact;
record that code-version difference, never rewrite source provenance.

## Results

Native execution `eaeb81f`, seed182,542 controls completed. Teacher8/8,
source-PD8/8 and GT-tau/R8/8 all passed hold45 and controlled placement.
Median held436.5/435/436 frames, clipping0/0/0.0461%. All frozen physical gates
pass: PROMISING for this R checkpoint with oracle actual hand trajectory.
Independent audit `3093877` passes with exact0 error for actual PD, R measured
state, tau, predictions and requested targets. Retained outputs:
[execution](../../../../../../outputs/trajectory-policy/competent-pd-execution-20261011-r1/result.json),
[audit](../../../../../../outputs/trajectory-policy/competent-pd-execution-20261011-r1/audit.json),
[behavior](../../../../../../outputs/trajectory-policy/competent-pd-execution-20261011-r1/behavior.png).
Mean goal endpoint error teacher129.26mm,source-PD145.84mm,R171.01mm: these
are controlled-placement successes, not precise goal attainment.
Next is the [separate joint BC Probe](P-20261011-competent-joint-bc.md).

## Limitations / future evidence

Oracle trajectory control is not H/G->tau competence, precise goal tracking or
Cm utility. Source same-wave holdout and separate evolving PhysX rows limit
attribution. Teacher terminal position error is143.98mm relative to G; success
here concerns geometric grasp and controlled placement, not exact target pose.
