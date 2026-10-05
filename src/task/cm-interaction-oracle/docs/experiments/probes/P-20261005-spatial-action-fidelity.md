---
schema: ref2dex.probe.v2
probe_id: P-20261005-spatial-action-fidelity
experiment_id: P-20261005-spatial-action-fidelity
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: b06f09e
claim_id: C3
hypothesis_family: HF-spatial-action-fidelity
probe_index_in_family: 1
seed_pool: probe
seeds: [245, 249]
decision_changed_if_positive: distinguish preserved physical action information from spatial selection or compression before changing consequence learning
decision_changed_if_negative: stop attributing weak consequence contrast solely to downstream training and inspect geometric locality or representation
status: UNPROMISING
run_id: spatial-action-fidelity-s249
---

# Where does continuous finger action information survive?

Result: UNPROMISING for linear action recovery from the fixed local/fused paths; interpretation restricted to the existing arm dictionary.
Decision: Preserve full per-finger motion and test a centered action-innovation objective; do not infer that the 2cm radius caused consequence failure.

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

## Results and root scope correction

GPU6 main11.67s, peak708MiB; smoke3.09s engineering-only UNCLEAR.
Root full frozen-feature, savedPCA and decoder replay8.46s: all errors0,
source normalizers0, normal-equation relative residual≤3.89e−16.
24linear fits only; no Cm/task/PPO fitting or new simulation. Main+smoke
about56MiB; 59Task tests pass.

| Stage | Nominal gainvszero | Forecast gainvszero | Nominal currently supported | Forecast currently supported |
| --- | ---: | ---: | ---: | ---: |
| RawObject | 0.8893 | 0.8853 | 0.9258 | 0.9224 |
| RawHandBase | 1.0000 | 0.9987 | 1.0000 | 0.9982 |
| Initialized_LocalFlow | 0.4050 | 0.4694 | 0.5036 | 0.5416 |
| Initialized_Contact | 0.2951 | 0.3503 | 0.3646 | 0.4147 |
| Initialized_Tokens | 0.3393 | 0.4262 | 0.4220 | 0.4745 |
| Initialized_Fused | 0.3121 | 0.3787 | 0.4055 | 0.4747 |
| Initialized_Output | 0.2638 | 0.4047 | 0.3318 | 0.4482 |
| Trained_LocalFlow | 0.4050 | 0.4694 | 0.5036 | 0.5416 |
| Trained_Contact | 0.3872 | 0.4641 | 0.4726 | 0.5320 |
| Trained_Tokens | 0.3682 | 0.4635 | 0.4630 | 0.5356 |
| Trained_Fused | 0.3306 | 0.3436 | 0.4122 | 0.4275 |
| Trained_Output | 0.2794 | 0.2877 | 0.3472 | 0.4625 |

RawObject gains .8893(CI.8300–.9320) and .8853(CI.8102–.9395);
RawHandBase≈1/.9987. Trained Fused only .3306/.3436. Current pre-action
support is index77.71%,middle84.94%,pinky33.13%,ring80.72%,thumb70.48%.
Supported-finger Fused .4122/.4275 remains limited. Both registered branch
choices are LOCAL_SELECTION, meaning inspect the local summary contract.
**This does not identify radius filtering as the cause:** LocalFlow uniformly
averages selected flow and omits actual edge relative positions, normals,
distances, indices and individual flow. PCA32 and linear recoverability can
also discard information; no information-theoretic absence claim.

Independent CPU review: decoder replay0, normal-equation residual≤1.30e−15;
metrics/bootstrap within1.79e−7, axis/arm reductions within1.17e−6. Known
nominal targets have rank6 and nearly fixed arm means. Arm-mean lookup itself
achieves nominal≈1.000/forecast.99597. Thus perfect RawHandBase recovery can
be fixed14arm identification, not a validated continuous physical law. CPU/GPU
divide-by.32 difference≤1.19e−7 and early strict tolerance failures were
verified numerical precision effects, not a new scientific anomaly.

V13 consumes points/normals/flow without explicit hand link/finger ID; it was
designed for global object rigid effect. I14 here includes anatomy-ordered
five netforces and five proximities. This suggests testing per-finger motion
with explicit identity, but H already contains ordered current state, so the
entire adapter is not proven nonidentifiable. Root next smallest intervention
combines full intrinsic per-finger motion with source/OOF state-mean subtraction
and an action-centered residual, with matched joint, ordinary-residual and
shuffled controls to separate representation from learning objective. No
larger network, radius sweep or unseen-dose claim from this diagnostic.

Reports in main folder: result.json, diagnostic.pt (compact PCA scores, decoder
weights/targets/predictions), engineering_replay.json, action_fidelity.png,
independent_review.md/json and review_provenance.json. Full stage tensors are
regenerated, not duplicated. Field metrics do not establish E/I or task value.
