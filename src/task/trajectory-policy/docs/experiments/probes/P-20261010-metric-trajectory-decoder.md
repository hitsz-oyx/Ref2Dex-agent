---
schema: ref2dex.probe.v2
probe_id: P-20261010-metric-trajectory-decoder
experiment_id: P-20261010-metric-trajectory-decoder
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: 7e7e909
claim_id: C3
hypothesis_family: HF-trajectory-policy-action-space
probe_index_in_family: 4
seed_pool: probe
seeds: [297]
decision_changed_if_positive: freeze executable D and initialize measured-history actor
decision_changed_if_negative: stop48D reconstruction search and use less compressed trajectory baseline
status: UNPROMISING
run_id: metric-trajectory-decoder-20261010-r1
---

# Does a pose/FF metric retain the executable trajectory in D48?

## Motivation / Decision Note

Previous goal turn made progress: unconstrained linear-node optimum excluded
close reconstruction, then coordinate-PCA improved average geometry but failed
startup/FF screen. These three D probes have not advanced policy utility;
they excluded promoting two concrete initializers and isolated temporal/metric
limits. Keep Mission's self-trained policy and eventual matched Cm utility,
but cap this representation detour: one last fixed48D fit. If it fails, use a
less compressed D to enter H->c learning rather than more48D parameter search.

The candidate changes metric/coordinate design, not root claim. Learned fixed
basis still decodes standalone actor c with measured q/object; no online mother
actor, clock, phase, tactile, true future q/object or command labels. Existing
hand-derived geometry/R reused read-only, oldTask experiments remain paused.

## Protocol

Same543hand-derived24future windows, mean+48fixed temporal modes. Coordinates
are physical object-frame XYZ offset, relative SO3 rotation vector and absolute
six independent finger angles, scaled by[.01m,.1rad,.1rad]. Unlike earlier
unconstrained logit/tanh coordinates, no artificial boundary saturation variance.
Decoded XYZ clipped at1m per local coordinate; fingers native-clipped/coupled;
SO3 Exp always produces a rotation, with nearest native Euler branch.

Fit a single SPD metric: pose weights1 forfirst8 and.25 fortail16; add the
first8 rows of F=I+.1D for six wrist channels, D future-only central differences
at1/30. Fingers use pose weights only. Whiten with Cholesky, one GPU SVD, keep48,
unwhiten decoder and store dual encoder; standardize latent by component score
std floored1e-4. Fixed metric, no scale/dimension/seed/coefficient sweep.
XYZ metric equals nominal wrist FF target error; rotation-vector derivative is
only a surrogate for native Euler FF. Actual hand FK and native Euler FF must
pass independent coverage checks. This is not full contact-dynamics optimization.

Same fixed offline screen: first8 hand3D RMS<=5mm, palm maximum<=10mm,
worst-window wrist FF XYZ3D RMS<=10mm, finite/bounds/coupling and dual consistency.
All68deployment windows are in-sample representation coverage, not Validation
or a learned H->c result. Only if passes, one16env/542step/seed297 native wave:
GT,dense FK,metric48,metric48 repeat4rows each randomized. Replan8 using own live
q/object; explicit oracle hand-derived future geometry supplies c labels.
Frozen R consumes tau and real-time feedback each step. No reference reward.

Physical screen unchanged: GT and dense each>=3/4 >=433held+terminal;
each metric arm>=3/4 longheld+terminal and clipping<1%. Positive PROMISING,
both fail with calibration UNPROMISING, other cases UNCLEAR. Offline miss means
no native run and no physical efficacy claim. Saved c/basis/plans/input897,
velocity and actual native/PD/outcomes audit required. No post-hoc screen change.

## Resources / stop

One idleGPU4, <=8GPUmin/512MiB; fit+coverage<=120s, conditional one native wave
<=300s. GPU SVD/FK/controller/GPUPhysX, CPU only transforms/file/statistics and
tiny contract smoke. Monitor utilization/memory/ETA on native startup/128steps.
No neural policy training, WM, extra data/seed or automatic simulation retry.
Stop on source/input drift, nonfinite, unsafe occupancy or unexpected reset.
Staymain/no new branch/push, preserve all old outputs. No new external authority.
GPU2 preflight rejected71%/6677MiB before model/output creation; GPU3 also busy.
Root relocates the unchanged single-GPU design to idleGPU4 within global budget;
no foreign process touched, no started experiment to restart, no extra run/seed.

## Results

Implementation `03c19e4`, actual fit/native code `7e7e909`; seed297 onGPU4.
Offline screen passed: first8 hand3D RMS0.91855mm, palm maximum4.38554mm,
worst-window wrist FF XYZ3D RMS2.80546mm; full24 handRMS3.57701mm.
This is in-sample nominal-geometry coverage, not learned policy performance.

One native16env/542control wave completed55.168s. GPU memory~7537MiB,
utilization33--47% at later monitors; CPU-exchange/GPUPhysX cost within cap.
No owned GPU process left running. Actual physical screen unchanged:

| Role | >=433held and terminal | Terminal held | Median max held | Clipping |
| --- | --- | --- | --- | --- |
| Original GT tau | 4/4 | 4/4 | 478.5 | 0% |
| Dense hand-derived FK tau | 4/4 | 4/4 | 482.5 | 0.09225% |
| Metric48 | 2/4 | 2/4 | 387 | 0% |
| Metric48 repeat rows | 0/4 | 0/4 | 291.5 | 0.04613% |

Both metric arms missed the >=3/4 gate with calibrated controls: local
**UNPROMISING** for this projection/live-calibration/frozen-R combination.
All eight metric rows first acquired stable grasp; six later lost it, with
last stable ticks206/282/349/352/356/356. These are not eight acquisition
failures. Repeat arm is different native rows, not an independent seed.
Longheld/terminal is a grasp metric, not completion of original placing task.

Independent source/plan/input/command replay passed: actual encoder c error0,
translation0, rotation-matrix1.01e-6, finger1.19e-8, future-only velocity9.54e-7,
897features2.86e-6, command2.38e-7 and applied PD0. Audit r1 failed on missing
`encoder_c` accumulator key, after that additional check was introduced;
`4be7d90` fixes only audit accumulation. Preserved failed record, replay r2
completed; no simulation rerun or original result modification.

Independent read-only review under AGENTS14 found no whitening-direction,
dual-encoder, actual-c or Euler-branch implementation error. Weighted decoder
orthogonality residual2.90e-8; encoder/metric-adjoint difference7.82e-8;
maximum adjacent future Euler coordinate change0.1934rad. However native
compressed plans, anchored in their own measured q/object, have first8 hand
3D RMS8.017mm and maximum43.026mm against dense geometry across68x8 queries;
first query RMS2.941mm/max7.749mm. XYZ FF error `delta_q+.1delta_v` RMS3.233mm,
first query1.366mm. This is a different input domain from offline geometry-q
and fixed initial object frame, so offline0.919mm cannot characterize executed
trajectory accuracy. Near loss, prefix RMS5.18--12.19mm; correlation alone
does not identify the cause of dropping.

Artifacts:

- Fit: `outputs/trajectory-policy/metric-trajectory-decoder-20261010-r1/{manifest.json,result.json,basis.npz,coverage.npz}`.
- Native: `outputs/trajectory-policy/metric-trajectory-decoder-execution-20261010-r1/{manifest.json,result.json,trajectory.npz,plans.npz,monitor.jsonl}`.
- Replay: `outputs/trajectory-policy/metric-trajectory-decoder-execution-audit-20261010-r2/{manifest.json,audit.json,behavior.png}`.
- Preserved failed replay: `outputs/trajectory-policy/metric-trajectory-decoder-execution-audit-20261010-r1/manifest.json`.

## Follow-up Decision Note

Honor the predeclared cap: stop48D reconstruction search, retaining all failures.
Use a direct288D trajectory representation (24x[XYZ,relative SO3,six fingers])
without a PCA projection or reference-action base. First replay its encode/decode
against all68x4 successful dense calibration plans, checking literal native
Euler q, FK and future-only FF, not merely orientation equivalence. This is
bounded GPU engineering identity, no new rollout or scientific claim. If identity
passes, next research decision is measured-history actor initialization and
real-reward RL, not another compression sweep. More action coordinates may
make RL harder; that tradeoff remains unresolved. No abandonment of all48D
policies/trajectory RL, no new external permission/resource requirement.

## Limitations / future evidence

Single-motion in-sample geometry, oracle c labels and frozen warmstarted R.
Projection isn't control-optimal c, covariance fitting isn't learned actor.
Positive opens actual H->c learning, not a claim of policy/Cm benefit. Need
measured-history actor initialization and RL, native-action comparison, then
matched Cm-on/off policy training; physical frozen selector alone insufficient.
