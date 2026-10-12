---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref24-chunk-R
experiment_id: P-20261012-goal-point-wam-ref24-chunk-R
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 33
seed_pool: probe
seeds: [295, 296, 292]
decision_changed_if_positive: proceed to explicit PTv3 hand anchors then gated frozen predicted-tau world consistency
decision_changed_if_negative: stop WAM architecture changes and address GT chunk execution first
status: UNCLEAR
run_id: wam-v2-chunk-fit-20261012-r1
---

# ref2_4 A: GT trajectory to independent native action chunk

## Decision Note and staged design

User explicitly requests sequential WAM revision: A chunk-R, B PTv3 explicit
hand-token readout, C teacher-forced world followed by frozen predicted-tau
consistency. Current object endpoint32-point/memory goal stays unchanged.
No simultaneous from-scratch changes or new branch. This card only runs A;
B/C require A's independent GT physics gate, and C also requires B's offline
non-regression. Implementation and parameter choices follow the supplied
ref2_4 design under repository autonomous research authority.

A serves manipulation's execution prerequisite: can a trajectory-conditioned
native-action chunk Transformer execute privileged GT tau while WAM is absent?
The existing learned-native Transformer from2026-10-10 fits failure-rich
trajectories and never qualified. Reuse its128/heads4/FFN512/two-decoder-layer/
8-learned-query architecture, replacing linear tau projection with33->128->128
MLP. State87->MLP token plus24 temporal tau tokens form decoder memory.
No PD target prediction; labels are12 independent actually applied native
commands[t:t+7]. Passive6 columns zero, native clipping recorded. Replan
at every physical tick and execute only first action; never open-loop eight.

Cheapest decisive run: nominal competent audited32-episode source split16
TRAIN/8 development-selection/8 report; actual audited role3 t8/16/32 recovery
259 windows are TRAIN. Pure actor failed futures are never opened as labels.
TRAIN-only state/tau/action normalizers. Batch256=96 nominalstartup(t<16)+
96 laternominal+64 true recovery, Adam3e-4/clip1/L1 standardized all8x12,
seed295 model/296 draws, fixed3000 updates. Select lowest development L1
at fixed250-update reports and save ONE final selected packet; no sweep.
Same motion/reset means reports are development, not independent Validation.

Physical Decision is preauthorized even if strict offline accuracy is unmet:
finite bounded native output and correct labels first, then one128-control
GT-only group (teacher, original qualified PD-R, NEW chunk-R), seed292.
Each R consumes the SAME source-env24 privileged future in its OWN measured
current object frame. No WAM, copied hidden state, teacher rescue or stale
native action8 caching. Calibration: teacher/oldR eachhold45 andclip<1%; new
R musthold>=max(45,80% oldRheld) andclip<1%. Partial128 only establishes
local grasp execution, not full placement/generalization or Cm utility.
If failed with valid calibration, do not start B/C under ref2_4.

Resources: GPU2 idle preflight, fit<=600s/physical<=600s/total<=20GPUmin,
new WAM-v2 stageA<=80MB/global400GB. Prior ref2 artifacts804.52MB remain
within their850MB stage limit; new architecture outputs use their own explicitly
bounded wam-v2 directories. Stop on nonfinite, wrong action/control alignment,
foreign process, reset, timeout or output boundary. Existing code/weights and
old WAM route stay preserved. Code/version identities suffice; no extra
historical hash-calibration audit introduced for this new explicit entrypoint.

## Subsequent interface design (not yet enabled)

B gathers current2x11 PTv3 hand features from full packed point output before
scene<=256 sampling; invalid hand points mask tozero. Add these to temporal/
keypoint/raw observed-hand query residual, keeping raw adapter. Test that
point sampling cannot remove hand identity and that only observed hand tokens
reach trajectory queries. Compare h1/first8/nominal/recovery/goal sensitivity
against old WAM before new actor physics.

C Stage1 learns W(H,GT tau)->actual object E and A(H,G)->GT tau. Stage2 freezes
executed-action adapter and world decoder, detaches world scene/object context;
construct9D[p0+tau,tau,tau[t]-tau[t-1]] with first step tau1. World effect
loss must reach actor ONLY through predicted tau. R parameters are fixed for
predicted action supervision. Valid actual expert futures/effects only; pure
current-PD rows have no true future/effect labels. Test world parameter/scene
shortcut gradients0 and predicted tau/actor gradient nonzero. No RL/3PoinTr.

## Entry points / evidence

- `src/task/trajectory-policy/src/trajectory_policy/chunk_retargeter.py`
- `src/task/trajectory-policy/tools/run/fit_chunk_retargeter.py`
- `src/task/trajectory-policy/tools/run/execute_chunk_retargeter.py`

Inputs: qualified `competent-pd-teacher-20261011-r2`, original calibrated R,
Probe31's audited actual takeover trajectories. Current failed pure student
futures and original low-quality PPO corpus are excluded from training.

## Limitations / future evidence

Oracle GT future is privileged; success cannot establish WAM competence.
Multi-control inverse can be nonunique. Old/new R differ in representation,
objective and initialization, so this is a new route qualification, not a
causal architecture ablation. One nominal128 trace cannot validate placement,
held-out disturbances/motions or final Cm-on/off. B/C remain conditional.

## Results

Pending engineering smoke, fixed chunk fit and independent GT physics.
