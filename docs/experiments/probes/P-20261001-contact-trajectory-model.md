# P-20261001-contact-trajectory-model

HF10 slot1/2, Decision Probe. Improve the physical role of Cm after HF09's
local mean-lift signal and unresolved retained progress/drop risk.

Question: can task-relevant physical trajectories and explicit base-relative
numeric-action effects be learned/calibrated from the recorded real sim data?
Decision: adequate physical heads permit a separately fixed fresh randomized
five-selector mechanism test; inadequate heads require representation/data
review. This is model preparation, not utility evidence from recycled holdout.

Frozen inputs:1984 uniform six-arm records (ranking-r1 seeds331–333) plus3941
targeted actual-state records (targeted-r1 seeds341–346). Expert/actor/RMS and
old ranker checkpoint remain read-only. Initial-frame split follows old ranker
mapping; new frames receive deterministic stratified split seed9831; no group
crosses fit/cal. Old held targets have been used in HF09 and are only audited,
not called an independent new scientific test. Normalization of encoder inputs
retains the old fit-only scales; new height-target scales use new fit only.

Outputs:10 signed Δz(mm),10contact logits, joint last3contact logit, release
logit (22). release includes pre-existing or within-window acquired lift>=3cm
followed by height<2cm or6lost-contact steps after lift. It is an all-window
event; do not condition evaluation on future acquired lift. Base trajectory
head plus action-effect head minus its numeric-base evaluation gives exact
zero base effect. State-only sixslot heads share history/baseintent/phase but
no numeric candidate input; shuffled independently corrupts fit-row actions.
No future actions/labels/context enter prediction.

Warm encoder initialization from matching HF09 variants/members, new physical
heads initialized with private seeds9861–9863;3members/variant,1000Adam steps,
lr.001,batch64, initial-frame bootstrap9961–9963. Loss: normalized signed-height
smooth-L1, perstep and joint-contact BCE, all-window release BCE. Source mix
balanced uniform/targeted within each bootstrap to preserve candidate support;
no PPO/V/actor training. Report training norms and all actual update counts.

Decode retained score=min(last3predictedΔz).clamp_min(0)×joint_contact_probability.
Candidate relative-score mean minus ensemble std must exceed calibration
median factual retained-score error, floor.5mm. Contact guard joint-probability
decrease<=.02. Relative release-probability mean+std<=0; unsupported release
class abstains entirely. These are heuristics, not certified effect bounds.
Class support fit>=20events/cal>=5events; factual joint-contact calibration
MAE<=.20 and release Brier<=constant fit-event-rate predictor. Fit/calibration
only; no threshold sweep. Report factual height/retained errors and coverage.

Slot1 label remains UNCLEAR for policy utility regardless of training loss;
prepare fresh control test only if heads are finite/supported and calibration
screen passes. Fixed single idleGPU4, model/batch inference GPU first,≤60min/
8GiB including retries. Unique owned outputs, original artifacts untouched.

## Run r1 terminal

run_status COMPLETED, code fbf3002;51.11s GPU4. All9models completed1000updates
and warmed8encoder keys each, original inputs/NN hashes verified and GPU
released.20 targeted tests at launch. Fit3277windows/444episodes/89release
events, cal1297/186/13, historical engineering audit1351/192/39; grouped
partitions preserve first episodes and initial frames.

Original physical preparation gate FAILED; policy label UNCLEAR. Cm cal
height RMSE24.35mm, retained MAE3.416mm (state-only3.757/shuffle3.673), joint
contact MAE.239>.20, release Brier.00953<constant.01022. Old held is engineering
only and cannot promote utility. Checkpoint f7c0b0f95d2344947bd872e0ef518bc0fa49ebc592a9c154bb101f55ebdd22e8,
owned output `src/task/CmResidual/research/contact_consequence/output/P-20261001-contact-trajectory-model-r1/`.

Terminal diagnostic:9.25%cal histories exceed10old std; lower-norm contact
MAE still.232. Largest973std occurs in rare finger velocity spikes, not a
global feature unit mismatch. Joint Brier.157 improves overconstant.223, but
ECE.096 indicates overconfidence (predicted.991bin observed.908). More
updates or a normalization-only rerun are not justified. Old MAE failure
stays; subsequent probability reliability/signed-support design is separately
fixed in [decision](../../decisions/D-20261001-supported-height-calibration.md).
NN training remains closed; only scalar calibration and prospective actual
control may follow the new setup gate, with total family budget2/2.
Full result/audit: [results](P-20261001-contact-trajectory-model-results.json).
