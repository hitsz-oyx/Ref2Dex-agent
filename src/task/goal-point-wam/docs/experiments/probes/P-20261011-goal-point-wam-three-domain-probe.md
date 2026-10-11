---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-point-wam-three-domain-probe
experiment_id: P-20261011-goal-point-wam-three-domain-probe
date: 2026-10-11
task: goal-point-wam
branch: cm-pointflow-effect-pretrain
git_commit: 5f12ae31cc725a9f5d660ba476456a77c47d94b9
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 2
seed_pool: probe
seeds: [247]
decision_changed_if_positive: continue this no-video route to a matched multi-seed validation after fixing the held-out panel contract
decision_changed_if_negative: stop unbounded training and diagnose source/representation coverage before adding video or robot policy components
status: UNCLEAR
run_id: three-domain-probe-20261011-r2
---

# Goal-Point-WAM three-domain no-video Probe

This Decision Probe asks whether the approved scene/action/world wiring can make
useful progress on the frozen OakInk2/GRAB/ARCTIC corpus before any video input
or robot policy work is added. It is a single-seed short Probe, not a formal
architecture or generalization Validation.

The cheapest discriminating run was 600 AdamW updates with the fixed balanced
validation panel, followed by small train and held-out panels. A positive result
would justify preparing a matched validation; a negative or ambiguous result
would first require a source/representation diagnosis. The run keeps the
30 Hz/history-4/horizon-24 dual-hand-11-point contract, train-only shared
normalization, and `measured_dynamic_hand_main` supervision. Video and RGB are
disabled.

## Protocol

- Config: `src/task/goal-point-wam/configs/three_domain_no_video_probe.json`.
- Seed `247`, 600 updates, microbatch 2, fresh AdamW, one local GPU.
- Data: frozen `pointworld-main3-sharedstats-20261009` main-three manifest;
  no source or normalization files were changed.
- Device: local physical GPU2 (`CUDA_VISIBLE_DEVICES=2`); GPU0/1 had unrelated
  processes and were left untouched.
- Best checkpoint selection used the fixed 96-window balanced validation panel
  (`32` windows per source). The diagnostic train/test panels sampled `96`
  windows per source with independent fixed seeds.

## Result

The run completed at step 600 in 327.4 seconds. The best checkpoint was step
400. Peak observed GPU memory was about 12 GiB of 24 GiB; sampled utilization
ranged from 0% to 94%. Losses, gradients, checkpoints and metrics remained
finite. The goal replacement audit changed world translation/rotation by at
most `7.18e-7`, consistent with the goal-blind world branch at this numerical
precision.

Errors below are trajectory EPE averaged over all 24 future steps / world point
EPE at h24, in millimetres. The trajectory column is not endpoint-only EPE:

| Panel | OakInk2 | GRAB | ARCTIC |
| --- | ---: | ---: | ---: |
| balanced validation, best step 400 | 35.2 / 8.2 | 141.4 / 101.8 | 71.6 / 57.3 |
| train diagnostic, 96/source | 35.2 / 10.0 | 110.4 / 194.6 | 61.8 / 66.2 |
| held-out diagnostic, 96/source | 43.0 / 8.3 (test) | 132.0 / 201.6 (test) | 73.0 / 104.2 (val) |

The manifest contains zero ARCTIC test windows, so the ARCTIC held-out number is
explicitly a validation diagnostic and must not be called a test result. OakInk2
is stable between validation and test. GRAB has high world error even on the
small train diagnostic and remains high on test; this does not look like a pure
memorization gap. The source panels are small and not a matched full-split
evaluation, so the overall Probe conclusion is **UNCLEAR**, rather than a
claim that the architecture is overfit or that it generalizes.

The static no-motion baseline on the same diagnostic panels is 6.5 / 201.5 /
104.1 mm world EPE for OakInk2 test, GRAB test and ARCTIC validation. The
learned world branch is 8.3 / 201.6 / 104.2 mm, so no source shows a clear
world-motion gain in this short run. The contract audit found full action and
hand validity in all 512 sampled train windows; the main distribution difference
is scale and scene structure: GRAB's median h24 object translation is 105 mm
with 534 scene points and one object, versus OakInk2's 3.5 mm, 3094 points and
six objects, and ARCTIC's 36.0 mm, 1046 points and two objects. This points to a
source/representation or training-budget diagnosis, rather than evidence of a
pure train/test memorization gap.

Following the approved stop condition, no unbounded extra epochs were started.
The next decision is a source/representation and held-out-contract diagnosis;
video, robot policy, and formal multi-seed Validation remain out of scope.

## Artifacts

- [Run result](../../../../../../outputs/goal-point-wam/three-domain-probe-20261011-r2/result.json)
- [Best validation panel](../../../../../../outputs/goal-point-wam/three-domain-probe-20261011-r2/validation_best_panel.json)
- [Held-out panel](../../../../../../outputs/goal-point-wam/three-domain-probe-20261011-r2/heldout_panel.json)
- [Train diagnostic panel](../../../../../../outputs/goal-point-wam/three-domain-probe-20261011-r2/train_panel.json)
- [Static baseline panel](../../../../../../outputs/goal-point-wam/three-domain-probe-20261011-r2/static_baseline_panel.json)
- [Source contract panel](../../../../../../outputs/goal-point-wam/three-domain-probe-20261011-r2/source_contract_panel.json)
- [Best checkpoint](../../../../../../outputs/goal-point-wam/three-domain-probe-20261011-r2/best.pt)
- [Engineering smoke card](P-20261011-goal-point-wam-three-domain-smoke.md)

## Limitations and future evidence

### Implementation scope against user ref1 (reviewed 2026-10-11)

The retained producer script hash matches the current training entry. It creates
the WAM from random initialization; there is no PW-pretrained initialization
path. Thus this run did not test ref1's central reuse of learned PW scene weights.
The action adapter constructs 528 semantic action tokens, but `forward` discards
them and supplies only 24 hand/keypoint-averaged per-time tokens to the world
decoder. This differs from ref1 and the local design's full executed-action
token memory. It can lose semantic point assignment and has not been isolated
as the cause of the reported error. The small trained-adapter audit is recorded
in `tmp/goal-point-wam-ref1-review-20261011/action-summary-audit.json`; it uses
CPU for a tiny token-contract calculation, not training or batch model inference.

The goal is a current/future-step24 semantic hand endpoint proxy (44 point tokens
plus a null token), rather than the general unordered start/goal-cloud interface.
Robot H328, frozen R/PD, video and point-flow remain deferred by the approved
three-domain phase boundary; rigid SE(3) is a permitted intermediate in ref1.
The scene token cap samples by point-list index rather than FPS/interaction-region
selection. This is a declared representation simplification, not an established
explanation for GRAB/ARCTIC error.

The shared scene/trajectory/world wiring and goal isolation remain retained
engineering evidence, and the numerical results remain unchanged. The absence
of benefit in this run cannot refute the pretrained/full-action-token ref1
architecture. Align those implementation choices before using a longer run to
judge that architecture; world-supervision benefit to the trajectory branch also
requires a matched world-loss-off control rather than joint loss alone.

This is one seed and small fixed panels, not a full source test evaluation or
matched baseline. ARCTIC has no test windows in the frozen manifest. Full
source-held-out evaluation, a source-balanced diagnosis, and multi-seed
Validation are deferred until the GRAB/ARCTIC contract and representation
issues are understood.
