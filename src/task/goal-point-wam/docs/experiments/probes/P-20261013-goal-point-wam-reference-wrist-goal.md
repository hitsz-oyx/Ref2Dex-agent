---
schema: ref2dex.probe.v2
probe_id: P-20261013-goal-point-wam-reference-wrist-goal
experiment_id: P-20261013-goal-point-wam-reference-wrist-goal
date: 2026-10-13
task: goal-point-wam
branch: main
git_commit: ea6b91ef
claim_id: C3
hypothesis_family: HF-goal-point-wam-reference-goal
probe_index_in_family: 1
seed_pool: probe
seeds: [295, 296]
decision_changed_if_positive: collect new candidate visited-state expert futures before a physical goal-conditioning comparison
decision_changed_if_negative: stop this fixed sparse-wrist continuation recipe and distinguish reference redundancy from state and inverse-control coverage
status: UNPROMISING
run_id: reference-wrist-ab-20261013-r1
---

# Sparse planned wrist conditioning matched Probe

Result: UNPROMISING for this fixed sparse-wrist continuation recipe; exact-C
first8 mean A57.0823/B57.6666mm, +1.024% rather than the required10% reduction.
Decision: preserve current parent/R, do not promote or extend this candidate;
prioritize current-actor visited-state coverage and interaction semantics.

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

## Engineering admission

Two-update smoke r1 and deterministic r2 both completed. Final r2 takes20.99s;
A/B initial nominal/recovery/human draws, predictions and losses are exactly
equal. Both adapters receive gradients; frozen modules and R differences0.
Main execution revision `ea6b91ef`, deterministic CUDA kernels and fixed
CUBLAS workspace. All artifacts preserved; no scientific result from smoke.

## Completed outcome and scope

GPU2, execution code `ea6b91ef`, exactly1000 updates each,329.03s total;
GPU RAM2055MiB, observed training utilization29–52%. All own GPU use released.
Equal first-batch draws/predictions/losses verified for all three pools. Frozen
geometry/goal/raw-hand/control-adapter/World/R and buffers max difference0.
Both new adapters and decoders changed, and adapter gradients were observed.
Final B gate -0.0000328954 is small; adapter state max change.00685. Input
sensitivity exists, but does not establish useful semantic dependence.

| First8 hand EPE, mm | Parent | A: blank wrist input | B: planned wrist |
|---|---:|---:|---:|
| Exact C t32, TRAIN | 34.965 | 34.434 | 35.343 |
| Exact C t40, TRAIN | 51.996 | 50.226 | 50.528 |
| Exact C t48, TRAIN | 85.562 | 86.587 | 87.129 |
| Exact C mean, primary | 57.508 | 57.082 | 57.667 |
| t32 contact16, TRAIN | 25.284 | 25.455 | 25.360 |
| t40 contact16, TRAIN | 32.878 | 33.357 | 33.164 |
| t48 contact16, TRAIN | 37.697 | 37.799 | 37.769 |
| Nominal development | 15.388 | 15.396 | 15.391 |
| Startup development | 11.702 | 11.754 | 11.716 |

Each contact16 contains one genuine C takeover query plus15 expert continuation
queries; it is not16 autonomous visits and overlaps the exact-state panel.
The small contact16 improvements do not transfer to the exact-query mean.
No10% primary improvement. t32 contact first-PD XYZ p95 A5.906/B9.140mm also
fails its guard; other predeclared guards pass. Nominal first-PD p95
21.680/21.886mm, startup1.537/1.195mm. Both startup values exceed parent.697mm;
a favorable B-vs-A metric alone would not justify promotion over the parent.
Human all24 guards pass; full per-source values remain in result.json.

B correct/blank/+16-frame wrist inputs produce exact-state first8 means
57.667/57.748/57.739mm. Mean all24 valid prediction changes by about.80–1.21mm
under blanking and.72–1.20mm under shifting at exact C states; all human outputs
are exactly unchanged. Small sensitivity is not evidence of meaningful task
conditioning or statistical significance, especially with the small learned
gate and batching sensitivity below. No future labels enter the wrist path.

Independent saved-array audit passes all trajectory metric recomputations
(max difference0.0000301mm),
identical targets/masks across variants, and human mask invariance. No strict
physical or inverse-controller claim follows from this output arithmetic.
One numerical limitation was found and retained: the same physical takeover
query encoded individually versus in contact16 has roughly1–2mm mean valid
prediction differences (initial t32=2.376mm all24), although corresponding
first8 EPE scalar differences are smaller. These are separately computed
BF16/TF32 contexts; same-batch A/B initial outputs were exactly equal. We have
not isolated which kernel/encoder causes the batch-size sensitivity, so do
not assert it is harmless general numerical noise. As a sensitivity check,
use the first row of each contact16: A57.1437/B57.6415mm, +.871%, also failing
10% improvement. Thus the fixed-budget no-promotion decision is unchanged;
small positive/negative differences should not be overinterpreted.

Artifacts include both final overlays, targets and predictions, the paired
initial-batch receipt, independent_audit.json and comparison.png. These are
local offline Probe artifacts, not deployable checkpoint admission. No new
physics execution, new expert collection or data cleaning was performed.

## Next decision

Stop this particular low-LR sparse-wrist residual extension. It did not buy
near-term accuracy on even the three existing C states. This is not evidence
against reference conditioning generally: one fixed demonstration already
supplies temporal object poses, and wrist-only geometry leaves finger/contact
intent unspecified. Parent data cover old C states rather than the current
actor's failure distribution. Next useful evidence is actual current-actor
expert corrections at approach/contact/hold, with the fixed admitted R and a
reference-conditioned baseline. Revisit explicit contact/palm/finger intent
only with matched data and actual visited-state metrics; no larger training
or automatic deletion of static-object windows is justified by this result.
