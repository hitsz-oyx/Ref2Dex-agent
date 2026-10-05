# cm-interaction-oracle

Task for action-conditioned physical consequence E/I and task-relative action
quality, on the original branch `agent/cm-interaction-oracle`. The execution
route is paused while ref12 proceeds on this same branch. Task, branch and
experiment identities remain separate.

New execution and audit tools live here. Existing reusable spatial models in
`ObjectInteractionCmv2` and native Inspire geometry in `CmResidual` are reused.
Historical scripts/cards remain at their recorded paths during gradual migration.
New cards live under `docs/experiments/`; outputs use
`outputs/cm-interaction-oracle/<run_id>/`. Mission, campaign, state and seed
ownership remain in the repository-wide docs.

Current user-directed route (ref12): oracle actual hand flow → environment-OOF
predicted E/I → continuation Y. Compare shared current H, directflow, predicted
E/I, GT E/I, hybrid and State-predicted E/I controls. Paired-data engineering
and native execution prediction stay paused. Post-treatment oracle inputs do
not establish attainable flow candidates or deployable planning; final matched
trained-policy Cm utility remains open.

Ref5 point-flow G and surface-I probes are completed; absolute RTG gains were
weak or outlier-dependent. Task ref1's cross-fitted three-class advantage
labels stopped at G0. Task ref2 then explicitly motivated a DIFFERENT
matched ranking probe using the same continuous labels, without V/threshold
re-tuning: H59.71% versus current-action59.62% held-out conditional pair
accuracy, weak action permutation effect; UNPROMISING for this frozen contract.
Actual between-V test Spearman0.540 contradicts using Pearson0.978 as evidence
of strong sorting stability. Neither discrete G0 nor ranking G1 was rescued.

Historical action/outcome contrast limitations are retained; user defers paired
collection while ref12 tests the factual prediction chain on existing data.
Retain relative-outcome and physical E/I candidates; no core Cm refutation or
policy utility claim.
Do not select future-active/agreed test labels to rescue a failed gate.
Neither offline oracle correlation nor prediction loss demonstrates the final
matched Cm-on/off trained-policy utility.

Run tools: `tools/run/probe_relative_action_critic.py`; audit:
`tools/audit/audit_relative_action.py`. Protocol and completed evidence:
[relative action Probe](docs/experiments/probes/P-20261004-recap-relative-action.md).

Ranking run: `tools/run/probe_action_ranking.py`; independent raw-score audit:
`tools/audit/audit_action_ranking.py`.
[Ranking Probe](docs/experiments/probes/P-20261004-relative-action-ranking.md)
records the conditional negative branch and deferred true candidate tests.

Ref3's real randomized intervention Gate has now executed:320 assigned
four-step residuals in336 complete deterministic self-trained airplane
episodes, with full32-step windows and effective dose confirmed. Environment-
grouped H/direct/predicted-Cm/GT comparison fails its seven gates; no selector
or policy training launched. Retain the weak adjusted object-rotation response
and localized GT contact-retention information. All decisions were pre-lift;
early-hold/drop was subsequently tested under ref4; global Cm utility remains open.
[Randomized intervention Probe](docs/experiments/probes/P-20261004-randomized-action-intervention.md)
records the protocol, limits, independent review and root decision.
Collection: `tools/run/run_intervention_collection.sh` and
`tools/run/collect_interventions.py`; fitting: `tools/run/probe_interventions.py`;
audits: `tools/audit/audit_interventions.py`,
`tools/audit/audit_collection_provenance.py`,
`tools/audit/diagnose_randomized_arm_effects.py`.

User-selected ref4's early-hold A/B gate is complete:494 randomized full
windows from1,008 episodes. A UNPROMISING: no registered retention-I control
contrast. B UNCLEAR: GT I improves test error49.6%, but only two sufficiently
supported primary strata. Independent engineering review and factual
recomputation pass; small pair strata inflate the original macro ranking.
C, selector and policy training were not activated. Preserve GT I prognosis;
close the current four-step residual/I8 contract without seed or fit retries.
[Early-hold Probe](docs/experiments/probes/P-20261004-early-hold-intervention.md).
Gate tool: `tools/run/probe_early_hold.py`; saved-data/statistics audit:
`tools/audit/audit_early_hold.py`. Next control work must address the retention
action/operator timescale rather than another predictor on this failed A gate.

Ref5's physics-only duration Probe is complete:1,006 full randomized windows
from2,016 episodes, K4/8/16 jointly assigned across the same seven arms.
All dose/support checks pass; primary contact tail0.1645 and maximum K16
short-contact magnitude6.62pp fail the retained-response gate. UNPROMISING
for this K≤16 feedback-residual extension. Measured wristx+ hand displacement
atstep16 grows4.51→10.47→21.92mm, while retention-I response remains weak.
Some opposing baseline commands are consistent with compensation; no unique
cancellation explanation or global I/Cm refutation. No learned model fitted.
[Duration Probe](docs/experiments/probes/P-20261005-early-hold-duration.md);
physics analysis: `tools/audit/probe_duration_response.py`.

Ref6 fixed-K8 amplitude1/2/4 Probe is complete:863 full randomized windows
from1,680 episodes. Mechanical authority and localized thumb force/proximity
response are clear exploratory signals (I8 family tail0.0005), while the
registered task-aligned threshold gate is UNPROMISING (short-contact tail
0.8895, alpha4 max effect3.68pp). No conditional Stage2 or model fitting.
Preserve action→local-I evidence; do not equate it with useful load-bearing
control or Cm policy utility. All dose/provenance/independent reconstruction
checks pass, including numerical distance sensitivity.
[Amplitude Probe](docs/experiments/probes/P-20261005-amplitude-authority.md);
analysis:`tools/audit/probe_amplitude_authority.py`; descriptive direction
facets:`tools/audit/plot_amplitude_direction_curves.py`.

Ref7 adds `--intervention-set per-finger-range`:six isolated native finger
DOFs both signs, zero and matched composite controls (15arms). Includes yaw;
`--finger-range-fraction` defaults0.05. The completed Probe fixes0.20 to match
old alpha4, K8 and amplitude1. Explicit v4 additionally records native q
(q0:3 metres, q3:18 radians), true tip positions and measured hand-base poses.
Legacy seven-arm defaults/schema remain reproducible.

854 complete windows:registered own-I+task/useful gates UNPROMISING, while
late-contact response is present and yaw-minus/middle-minus task clues remain.
No formal physical benefit or Cm utility is claimed. Full per-joint PD/mimic
and measured six-driver/five-tip matrices are in the
[per-finger card](docs/experiments/probes/P-20261005-per-finger-control.md).
Audit:`tools/audit/audit_finger_amplitudes.py`; fixed-protocol statistics:
`tools/audit/probe_per_finger_control.py`.

Collector invocation pattern (unique run ID and bounded seeds/resources):
```bash
bash src/task/cm-interaction-oracle/tools/run/run_intervention_collection.sh \
  <run_id> <num_envs> <waves> <assignment_seed> <sim_seed> <wall_seconds> \
  early-hold "8" "1" per-finger-range .20
```
Range fractions standardize driver PD targets, not realized tip travel.


### Ref8 GT consequence sufficiency

[Experiment card](docs/experiments/probes/P-20261005-gt-consequence-sufficiency.md)
reuses the ref7 dataset: GT prognosis PROMISING, complete sufficiency UNCLEAR.
Six matched main scorers and three declared signed-force extensions share an
environment holdout; noisy arm contrasts transfer across halves/held arms.
Remaining-action confidence bounds and fit/representation sensitivity prevent
a causal sufficiency claim. Per-finger amplitude tables remain in the card.

Frozen fit entry (unique output directory; check current GPU ownership first):
```bash
CUDA_VISIBLE_DEVICES=6 TMPDIR="$PWD/tmp" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  timeout 600 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/cm-interaction-oracle/tools/run/probe_gt_consequence_sufficiency.py \
  --dataset outputs/cm-interaction-oracle/per-finger-control-s227/interventions.pt \
  --run-dir outputs/cm-interaction-oracle/<new-unique-run-id>
```
`tools/audit/audit_gt_consequence_sufficiency.py` takes the same dataset/run-dir
arguments for CPU file/statistical replay and a standalone plot; it refuses
to replace an existing replay artifact. No Cm/selector/policy fitting occurs.


### Ref9 conditional consequence prediction

[Experiment card](docs/experiments/probes/P-20261005-conditional-consequence.md):
UNCLEAR. Strict environment OOF, source-fold preprocessing and two action
shuffle controls preserve ref8 data/representation. I predictions use actions,
but this fixed fit does not preserve sufficient arm contrasts or stable extra
task value over direct Ha. Mean/persistence outperform the predictors;
independent abnormal-result review and GPU replay find no engineering error.
Do not extend epochs or enter selector/PPO from these results.

Fixed fit entry, after checking GPU ownership (use a new output directory):
```bash
CUDA_VISIBLE_DEVICES=6 TMPDIR="$PWD/tmp" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  timeout 600 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/cm-interaction-oracle/tools/run/probe_conditional_consequence.py \
  --dataset outputs/cm-interaction-oracle/per-finger-control-s227/interventions.pt \
  --oracle-run outputs/cm-interaction-oracle/gt-consequence-s231-r2 \
  --run-dir outputs/cm-interaction-oracle/<new-unique-run-id>
```
`tools/audit/audit_conditional_consequence.py` uses the same arguments for
GPU saved-weight replay and CPU statistics, without fitting. The report tool
`tools/audit/report_conditional_consequence.py --run-dir <completed-run>`
exports all14 signed arm vectors/units and a standalone plot. Both tools
refuse to replace existing artifacts. Per-finger physical magnitudes remain
in the card, including actual joints and true tips, separate from PD targets.

## Ref10: physical action representation and innovation

[Geometric innovation card](docs/experiments/probes/P-20261005-geometric-innovation.md)
converts current native q + intended PD commands to corresponding hand-surface
flow, with explicit actor-root FK and persistence/constant-velocity innovations.
Fixed train-only PCA/ridge controls compare categorical, joint and geometric
actions. Geometry-only input is not the original OI-CmV2 spatial architecture.

Both this screen and the [support factorial](docs/experiments/probes/P-20261005-geometric-support.md)
are UNPROMISING for their fixed contract. Independent review identifies
unconstrained state×action extrapolation, not future leakage/solver errors:
Flow I12.44 falls to1.14 after removing products and physical rescaling,
but still loses to additive Joint1.04. P_Flow does not beat the state control;
do not enter selector/PPO or infer that spatial/point-flow Cm has failed.
All14signed vectors, per-finger commanded/measured/nominal amplitudes and
independent review evidence are preserved in the cards and output folders.

After checking GPU ownership, reuse the bounded entry with a NEW output folder:
```bash
CUDA_VISIBLE_DEVICES=6 TMPDIR="$PWD/tmp" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  timeout 600 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/cm-interaction-oracle/tools/run/probe_geometric_consequence.py \
  --dataset outputs/cm-interaction-oracle/per-finger-control-s227/interventions.pt \
  --oracle-run outputs/cm-interaction-oracle/gt-consequence-s231-r2 \
  --run-dir outputs/cm-interaction-oracle/<new-unique-run-id>
```
`tools/audit/audit_geometric_consequence.py` takes `--dataset` and `--run-dir`
for saved-weight GPU replay, without fitting. `tools/run/probe_geometric_support.py`
takes `--dataset`, `--source-run` and a new `--run-dir` for the predeclared six-fit
diagnostic. `tools/audit/report_geometric_consequence.py --run-dir <main-run>
--support-run <support-run>` exports all arm vectors and a standalone figure.
All entries preserve existing artifacts. Next modeling work should use local
surface/contact structure and an explicit nominal-versus-realized execution
contract, without extending this exposed-test PCA/ridge fitting route.

### Ref10 continuation: reduced OI-CmV2 spatial Probe

The [spatial consequence card](docs/experiments/probes/P-20261005-spatial-consequence.md)
tests actual V13 local interaction/token/fusion modules, reduced width32,
with compact E12/I14 persistence innovations and strict environment OOF.
All28matched consequence/task fits complete: UNPROMISING for this fixed
nominal-action contract. Flow I.8891 vs State.8837/Joint.8670; predicted
consequences show no unique downstream gain. Saved-weight replay is exact.
This is a from-scratch reduced adaptation, not original GRAB/MANO training
or a refutation of spatial Cm. Future work must distinguish nominal endpoint
from predictable realized execution before repeating geometry fits.

Entry: `tools/run/probe_spatial_consequence.py`, with `--dataset`,
`--oracle-run` and a NEW `--run-dir`. Audit: `tools/audit/audit_spatial_consequence.py`,
with `--dataset` and completed `--run-dir`; GPU inference only, refuses to
overwrite its replay report. Full signed vectors, standalone heatmap and
PD/q/true-tip magnitude tables accompany the card and saved run.


### Ref10 continuation: executed geometry versus causal forecast

[Execution geometry card](docs/experiments/probes/P-20261005-execution-geometry.md)
separates post-treatment actual-motion diagnostics from source/OOF execution
forecasts. Original absolute ±pi wrist projection was incorrect: the URDF
wrist is continuous. Preserve that run, replay eight unchanged execution
weights, reuse valid oracle/control weights and repair only three affected
spatial fits. Corrected endpoint spatial contract is UNPROMISING at the fixed
budget; finger execution predictability is retained as a local positive signal.
No-fit FK decomposition attributes most endpoint error to wrist, while actual
joint oracle beats actual spatial oracle. No selector or policy fitting.

`tools/run/repair_execution_geometry.py --dataset <original> --original-run
<execution-geometry-s247> --run-dir <new-unique-folder>` preserves continuous
angles and refuses overwrite. Main uses GPU6≤600s within the experiment cap;
verify ownership first. `tools/audit/audit_execution_geometry.py --dataset
<original> --run-dir <completed-run>` replays both original and repaired formats,
without fitting. `tools/audit/decompose_execution_endpoint.py` takes those same
arguments for explicitly post-treatment wrist/finger diagnosis; add
`--causal-contracts` for separate decision-time substitutes. Reports refuse
overwrite. RMSE is XYZ-component average, with Euclidean counterpart disclosed.
Current root/point identities/object frame and source-only normalizers are
preserved. Slim repair artifacts require the original immutable run alongside.

### Ref10 continuation: action fidelity and anatomical innovations

[Spatial action fidelity](docs/experiments/probes/P-20261005-spatial-action-fidelity.md)
uses frozen stages and24source-only linear decoders. Known finger motion is
recoverable from raw intrinsic flow but weakly recovered from spatial latents.
The14arm lookup already nearly solves these labels: this is neither continuous
physical generalization nor an isolated radius-loss diagnosis. LocalFlow mean
is a proxy that omits actual edge geometry. Independent review narrows scope.

[Relative finger innovation](docs/experiments/probes/P-20261005-relative-finger-innovation.md)
then tests seven matched heads: full anatomical mean/RMS flow vs joint input,
ordinary vs candidate-centered residuals, nominal vs causal forecast, shuffled
control. Four immutable State nuisance fits supply source environment-OOF and
full-test baselines. All common inputs are shared. Fixed300updates produce
UNPROMISING: predicted flow I.9124 vsState.8837/matched predictedJoint.9077;
contrast corr.1565/sign56.57%, test shuffle penalty2.75% with CI crossing zero.
Exact input/weight/candidate replay passes. Stop this fixed contract without
epoch/seed retries; global physical action and Cm utility hypotheses remain open.

Both run entries take `--dataset`, `--forecast-run <corrected execution run>`
and a NEW `--run-dir`: `tools/run/probe_spatial_action_fidelity.py` and
`tools/run/probe_relative_finger_innovation.py`. Check GPU ownership before
using GPU6; freeze each protocol and stay within its resource cap. Corresponding
`tools/audit/audit_spatial_action_fidelity.py` and
`tools/audit/audit_relative_finger_innovation.py` take the same arguments for
saved-weight/input GPU replay, signed-vector exports and figures without
fitting. Reports refuse overwrite. Strict nested consequence OOF and matched
trained-policy Cm-on/off are still required before claims of policy utility.

The closing [contact innovation Probe](docs/experiments/probes/P-20261005-contact-innovation.md)
tests bounded surface-relative scalars and two nuisance baselines with ten
closed-form ridge fits. UNPROMISING for this finite basis/scale/estimator:
PhysicsContact I.98857 vs matchedState.98882/priorState.88370. Exact root and
independent matrix replay passes; all action columns meet the scale floor, so
equal ridge penalties do not establish equal effective shrinkage. Preserve
the result and pause the execution route per ref11. Tools are
`tools/run/probe_contact_innovation.py` and `tools/audit/audit_contact_innovation.py`,
both taking dataset, corrected forecast run and unique/completed output folder.
No follow-on execution or contrast-stability fit was launched.

### Ref11: measured endpoint and temporal oracle hand flow

[Oracle hand-flow Probe](docs/experiments/probes/P-20261005-oracle-hand-flow.md)
compares current State, full measured endpoint flow and two trajectory chunks,
with trained/frozen flow-shuffle controls. All correspondences retained in720
raw action slots; no execution forecast, PD/arm input, V13 pooling or Y scorer.
Frame0 object/root is fixed; actual wrist motion is included through native
FK at0/4/8 with independent measured base/tip agreement checks. Current-only
H preprocessing excludes previous controller actions and absolute world offsets.

Entry `tools/run/probe_oracle_hand_flow.py --dataset <ref7 interventions>
--split-run <gt-consequence-s231-r2> --run-dir <new-unique-folder>` (optional
oneupdate `--smoke`). Check idleGPU6 ownership and protocol resource cap first.
Actual flow is a post-treatment oracle input. Exact observable duplicate-state
support is audited separately; donor-label substitution does not create true
same-state candidate outcomes. Prospective planning and final policy utility
remain later requirements.


Completed main: Chunk E0.48968 vs State0.52895 (7.42% gain, CIpositive);
I0.55603 vs0.60486 (8.07%, CIcrosseszero). E passes; the fixed joint E/I
gate is UNPROMISING. Both trained/frozen flow shuffles support action sensitivity,
but Chunk-vsEndpoint intervals crosszero and same-state pairs are absent.
Retain the E signal and I uncertainty; no claim of temporal superiority or
prospective flow-space ranking. New current-state preprocessing also improves
State, so old-vs-new MSE differences cannot be attributed solely to flow.
All input/FK/weights/bootstrap GPU replay differences are0;69Task tests pass.

`tools/audit/audit_oracle_hand_flow.py` takes the same dataset/split-run and
completed run-dir for GPU geometry/input/weight reconstruction, CPU statistics
and standalone plot, without fitting; it refuses to overwrite an existing replay.
Read-only inspection found only cold-state serialization: warm PhysX caches
are not restorable. User defers paired-data collection and related parallel
matched-history engineering/smoke. Independent saved-head/statistical review
passes and is archived with provenance. Current prediction Probe is complete;
same-state candidate evidence remains deferred. Native execution prediction stays paused.


### Ref12: oracle flow consequence task chain

[Oracle flow task protocol](docs/experiments/probes/P-20261005-oracle-flow-task.md)
compares shared H, direct raw flow, OOF predicted E/I, GT E/I and hybrid task
readouts, plus a State-predicted E/I control. Source E/I preprocessing and
models are fitted per environment fold; test uses immutable ref11 full-source
weights. Y starts atstep9. Paired-state work remains explicitly deferred.

`tools/run/probe_oracle_flow_task.py --dataset <ref7 interventions>
--oracle-flow-run <oracle-hand-flow-s255> --run-dir <new-unique-folder>`
(optional `--smoke`) runs on idleGPU6 within the card's bounded resources.
Raw actualflow is post-treatment oracle information; chain gain/R does not
establish prospective planning or unique Cm benefit versus directflow.


Completed ref12: directflow primaryY MSE0.45788 vsH0.63951 (28.40% gain,
CIpositive). OOF predictedE/I0.54712 (14.45%, CIcrosszero); GT0.45440.
Oracle retained gainR0.4991 CI[-.0639,1.2790], fixedchainUNPROMISING.
Hybrid0.43694 improves4.57% over directflow with CIcrosszero, unique Cm
contributionUNCLEAR. Preserve directflow task information without planner claims.
GPU fullgeometry/foldsource preprocessing/OOF/weight/statistical replay errors0.
`tools/audit/audit_oracle_flow_task.py` takes the same dataset/oracle-flow-run
and completedrun-dir to rebuild inputs/folds/weights onGPU and export a plot,
without fitting; it refuses to overwrite existing replay artifacts.

Independent read-only review reconstructs all8raw Y labels and source-only
fold stats; saved-head normalizedCPU error≤2.17e-6 and bootstrap/R/gates agree.
Reports/provenance are archived beside the main artifacts.72Task tests and
scoped repository verification pass. This closes the current fixed-fit Probe.
