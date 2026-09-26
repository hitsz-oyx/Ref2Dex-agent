# P-20260925-duck-contact-phase-audit

- Classification: Decision Probe using the existing randomized duck
  wrist-z intervention records, no new GPU collection.
- Cm: off; this is a feasibility audit for a later contact-stage model.

## Question and decision

The pooled plus-minus wrist-z action raised H10 contact-supported lift
by 10.10 mm but reduced contact fraction by 1.38 pp. Is the loss
concentrated in early contact, leaving a later phase where upward
action has useful lift **without** a contact penalty? The simplest
candidate phase variable is global intervention step, available before
the action. Split the fifteen scheduled intervention steps into fixed
bins: early50–90, middle100–140 and late150–190. For each bin, among
actually eligible pre-contact rows, estimate the randomized plus-minus
effect on H10 contact-supported object dz and H10 contact fraction.
Use environment-cluster bootstrap intervals and report sample counts.

Only if the **late** bin has >=100 treated rows, >=10 mm supported-lift
point effect and nonnegative contact-fraction point effect should a new
independent-seed randomized collection test this phase gate. Otherwise
stop simple time gating of the duck z action and search for a different
candidate action family or better contact representation. This is a
post hoc subgroup question on the original seed210 and cannot overturn
the original failed overall gate or justify online deployment.

CPU only, <2 minutes and <5 MB output. Stop on missing data, assignment
imbalance, nonfinite outcomes or incomplete H10 followup.

## Results

The existing 64-env seed210 randomized record passed dose, assignment,
finiteness and complete-H10 checks. Environment-cluster 500-resample
analysis, stratified by the fixed five intervention steps per bin:

| Phase | Treated rows | Plus-minus H10 contact-supported dz | Plus-minus H10 contact fraction |
| --- | ---: | ---: | ---: |
| Early50–90 | 286 | +6.07 mm | −1.54 pp |
| Middle100–140 | 255 | +12.48 mm | −1.00 pp |
| Late150–190 | 248 | +12.29 mm, 95% CI [8.46,16.28] | −1.58 pp, 95% CI [−3.82,−0.03] |

The late phase has enough rows and positive lift, but contact remains
negative. The predeclared joint gate **failed**. Result:
`UNPROMISING` for a simple time-gated duck wrist-z selector. This
post hoc subgroup audit does not change the original pooled result
or imply Cm policy utility. Stop this local option family without a
new GPU run.

Artifact: `outputs/CmResidual/agent_duck_contact_phase_audit_20260925/report.json`.
