---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-reference-wrist-goal
experiment_id: P-20261013-goal-point-wam-reference-wrist-goal
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-goal-point-wam-reference-goal
probe_index_in_family: 1
seed_pool: probe
seeds: [295, 296]
decision_changed_if_positive: collect new candidate visited-state expert futures before a physical goal-conditioning comparison
decision_changed_if_negative: stop this fixed sparse-wrist continuation recipe and distinguish reference redundancy from state and inverse-control coverage
status: RUNNING
run_id: reference-wrist-ab-20261013-r1
---

# Sparse planned wrist conditioning matched Probe

Decision experiment supporting the goal-conditioned robot trajectory subproblem
of C3. Does adding sparse reference hand intent improve the current WAM's
near-term prediction at actual actor takeover states, beyond ordinary continued
training? The cheapest discriminating test reuses existing admitted data and
frozen measured-state contexts. No new data cleaning or physics collection.

## Fixed design and reference admission

User permits limited reference demonstration and continued autonomous goal
exploration. Existing planned wrist export has audited raw fields, reconstruction
and rigid-transform checks <2e-5. Its original engineering-only packet remains
unchanged. This experiment admits it as optional learned conditioning only,
not as literal robot commands or feasible robot anatomical targets. Human wrist
and Inspire hand-base differ; the trainable adapter learns their relationship.
Input never includes measured future hand/object states or future robot actions.
Existing expert future labels remain supervision in both arms.

- Start from ref2_8 selected stage2 step4500, same qualified frozen Chunk-R.
- Preserve measured q/dq/previous applied command, planned object goals and the
  old control adapter. Freeze old control adapter, encoders and World/R.
- Three planned human right-wrist poses +8/+16/+24: position and rotation6D in
  the **actual current object** frame, 27D. Fixed reference clock and terminal
  clamping, no inferred phase alignment. Only right-hand query gets residual;
  human replay inputs masked to exact zero.
- Same 27->128->128 GELU adapter with scalar zero gate in both arms, identical
  initialization. A receives all-zero wrist input (learnable constant residual);
  B receives planned wrist poses. Equal parameter capacity, data and draw order.
  Both also update the existing decoder. Zero gate must reproduce parent exactly.
- Fixed pools reused from ref2_8: 1104 nominal TRAIN, 195 recovery TRAIN, 192
  human TRAIN. Batch100 = nominal30/recovery50/human20, seeds295/296 reused
  for continuity, not independent replication. 1000 updates each, AdamW
  decoder1e-5, adapter1e-4, wd.01, clip1. Objective unchanged: trajectory scaled
  squared error + control loss + .1 world effect loss. No sampling alteration.
- Cache frozen contexts on GPU once; no caches on disk. Evaluate initial,
  step500 and fixed final1000; no best-checkpoint selection or budget extension.
- Primary: mean first8 hand EPE across the **three exact C takeover states**
  t32/t40/t48, B at least10% below A. These are TRAIN diagnostics, not heldout.
  All three exact and contact16 panels must be <=1.05*A+.1mm first8; their
  first-PD XYZ p95 <=1.1*A+.1mm. Nominal/startup first-PD same guard, clipping
  <=A+.01; each human all24 <=1.05*A+.1mm. All gates required for PROMISING.
- Report all24 and h1 too. B evaluation with blank and reference+16-frame input
  diagnoses sensitivity; same future label does not become the correct target
  for an alternative intent, so these are not counterfactual accuracy claims.

## Resources, stops, artifacts

One idle local GPU2, <=20min run, <=1GB new artifacts, global<=6 GPUs and400GB.
Stop on nonfinite, freeze/input mismatch, foreign GPU conflict, timeout, signal,
or storage budget. Use two-update small engineering smoke first. Source revision
recorded before launch; checkpoints saved only once per arm. Monitor utilization,
GPU RAM and ETA. No R repair, no remote job mutation, no deployment admission.

Entrypoint `tools/run/probe_reference_wrist.py`; focused tests cover coordinate
invariance, zero-gate parent equivalence, human masking and connected gradients.
Run manifests, panel keys, predictions/targets, both final overlays, sensitivity
metrics and resource progress remain in
`outputs/goal-point-wam/reference-wrist-ab-20261013-r1/`.

## Limitations / future evidence

Only three genuine C takeover queries; remaining192 recovery examples are expert
continuation states. None are new current-ref2_8 actor visits, and the parent has
already trained on this recovery set. Nominal roles24..27 are development roles,
not clean heldout proof. Small human panel16/source only guards gross forgetting.
Single seed, fixed motion and reset, repeated-window Probe; no physical grasp
claim and no refutation of richer goals or training from scratch. Planned wrist
may duplicate existing reference-object/time information for this one motion.
If PROMISING, next obtain actual current candidate visited-state coverage and
check the frozen inverse controller separately before any scale-up. Otherwise
stop this fixed recipe and preserve artifacts for attribution.
