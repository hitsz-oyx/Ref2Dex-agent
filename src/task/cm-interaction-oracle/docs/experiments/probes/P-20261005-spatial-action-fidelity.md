---
schema: ref2dex.probe.v2
probe_id: P-20261005-spatial-action-fidelity
experiment_id: P-20261005-spatial-action-fidelity
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: pending
claim_id: C3
hypothesis_family: HF-spatial-action-fidelity
probe_index_in_family: 1
seed_pool: probe
seeds: [245, 249]
decision_changed_if_positive: distinguish preserved physical action information from spatial selection or compression before changing consequence learning
decision_changed_if_negative: stop attributing weak consequence contrast solely to downstream training and inspect geometric locality or representation
status: UNCLEAR
run_id: spatial-action-fidelity-s249
---

# Where does continuous finger action information survive?

Result: UNCLEAR: frozen information-fidelity Decision Probe, run pending.
Decision: Decode physical finger perturbation from frozen stages before another Cm fit.

## Motivation and root Decision Note

MissionB/ref10 requires useful action-conditioned physical consequences.
Corrected execution Probe found predictable fingers (Ha error79.81% lower
thanH), wrist-dominated full endpoint error, and weak spatial I contrast.
RealizedJoint oracle outperformed RealizedSurface at the same fixed budget.
Question: is causal finger geometry lost by local selection/spatial fusion,
or retained but weakly mapped to E/I? Cheapest discriminator freezes the
existing encoders and fits small linear physical-action decoders, never new
consequence or task models. Alternative of more epochs/width is deferred;
new simulation cannot isolate this representation link as cheaply.

Use nominal PD and corrected source/OOF forecast inputs, each with identical
seed245 initialization versus its already-trained spatial encoder. Extract
all15 recipient candidates for all854 states, original688/166 environment
split. For every stage subtract the same state's arm0 feature. Decode
candidate fingerq12-minus-arm0 / .32rad, using generated intended PD q for
nominal, causal predicted q for forecast. These are **known input-motion
labels**, not actual unexecuted future consequences. No q8, future pose,
force, E/I or Y labels enter new decoding. Existing encoders were trained on
source E/I; that source-supervised history is explicit. Outer-test encoder
labels were never fitted. Forecast train geometry uses source environment
OOF, test full-source execution fit, as saved in corrected run.

This measures linear recoverability on the existing14arm grid. It can recover
arm identity instead of a general physical law; does not certify unseen
amplitude interpolation, actual execution accuracy, E/I or policy utility.
Failure of a linear/PCA decoder alone cannot prove information absent.

## Frozen stages and matched decoder contract

RawObject: corresponding point-flow candidate-minus-zero in current object
frame. RawHandBase: hold all candidates' wrist at recipient arm0 wrist,
then express endpoint differences in that baseline hand-base frame. This
removes common wrist rigid motion using only causal q/current root. Actual
network inputs still use unchanged total object-frame flow.

LocalFlow: uniform mean physical hand flow over network-selected valid
neighbors per object point, before learned edge weights. Contact: all64
object-point contact features. Tokens: four token features, normalized
anchors, normals and logmass. Fused: frozen32dim final spatial feature.
Output: frozen26dim consequence head. Compare candidate-minus-zero features
at every stage. Raw stages shared across initialized/trained paths. No
H/state concat in decoder: recipient subtraction isolates geometric action.

All stages use source-only PCA at most32 dimensions, source-only score scale
floor1e-6 and centeredfloat64 ridgealpha32/unpenalized intercept. Train
decoders on all15candidate motions of source688states, test166states;
candidate labels are available from inputs, so additional generated examples
do not consume new simulator labels. Fit PCA/scales only on source candidate
rows. Same feature-width bound and regularizer; no alpha/width/seed scan.
Candidate0 participates in training; metrics use14nonzero candidates to avoid
trivial-zero dilution. Report gainvszero, per-axis/per-arm MSE, sign for target
absolute normalized motion≥.02, and gain on current-radius-supported fingers
as a diagnostic only. Current support is computed from pre-action geometry,
never actual future contact or test-label selection.

## Predeclared decision rules, cost and stops

Raw motion recoverable: RawObject decoder gainvszero≥.80. If raw is weak but
RawHandBase improves≥.10 and reaches.80, prioritize intrinsic/shared-wrist
geometry. If raw strong and LocalFlow gain<.50, inspect selection/locality.
If raw and LocalFlow strong but trained Fused gain<.50 with≥.20 deficit from
raw, prioritize learned spatial compression. If trained Fused gain≥.80 and
retains≥.80 of raw gain, prioritize consequence/action-contrast objective
instead of claiming missing physical input. Initialization-versus-trained
comparison separates architectural from training-associated attenuation.
Otherwise UNCLEAR for this diagnostic. No downstream or PPO gate is changed.
These rules select the next smallest research intervention; not a formal
architectural conclusion. Environment bootstrap249/2000 draws reports fixed
decoder uncertainty, never refits. Numerical replay/source-boundary checks
precede interpretation; anomalous or route-closing negatives get independent
review under the user's standing instruction.

One idle GPU6, main≤600s, engineering smoke≤120s, total new artifacts≤80MiB.
Model/FK/PCA/ridge/inference GPU; CPU bootstrap/JSON and tiny synthetic tests
because CUDA startup dominates them. Stop on source/hash drift, leakage,
nonfinite, original candidate replay>1e-5, GPU conflict or artifact cap.
No external authorization, claim change or identity change. Preserve failures.
Smoke subset is engineering-only UNCLEAR and cannot establish these gates.

## Inputs and deferred evidence

Original dataset `outputs/cm-interaction-oracle/per-finger-control-s227/interventions.pt`;
nominal source `spatial-consequence-s245`, corrected forecast
`execution-geometry-continuous-fix-s247`, original execution assets referenced
by that run. Bind original dataset/source manifests/diagnostics/code/URDF and
mesh hashes. Output `outputs/cm-interaction-oracle/spatial-action-fidelity-s249/`
stores frozen input references, PCA/decoder weights, candidate predictions,
feature hashes and replay summaries, not duplicated full latent tensors.

Independent/new split, nonlinear decoders, unseen dose and actual motion
validation are deferred until this diagnostic changes a decision. Strict
consequence OOF and matched trained-policy Cm-on/off remain required for the
Mission; this information screen does not replace them.
