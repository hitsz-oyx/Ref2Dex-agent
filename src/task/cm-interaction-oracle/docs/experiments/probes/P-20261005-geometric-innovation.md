---
schema: ref2dex.probe.v2
probe_id: P-20261005-geometric-innovation
experiment_id: P-20261005-geometric-innovation
date: 2026-10-05
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: pending
claim_id: C3
hypothesis_family: HF-geometric-innovation
probe_index_in_family: 1
seed_pool: probe
seeds: [241, 242]
decision_changed_if_positive: prioritize intended geometric consequence modeling before candidate mechanism and matched trained-policy utility
decision_changed_if_negative: distinguish baseline or continuous-action gains from geometry gains without closing the spatial Cm hypothesis
status: UNCLEAR
run_id: geometric-innovation-s241
---

# Ref10: does intended surface motion generalize beyond action categories?

Result: UNCLEAR: protocol frozen before held-out predictions.
Decision: Use existing ref7 packet for a minimal representation discriminator; no new simulation, epoch/width/seed sweep or selector.

## Motivation and root Decision Note

This Decision serves Mission C3, GT→predictable consequence→trained-policy
utility. Ref9 memorized training data yet lost to train-mean/persistence on
held environments; its categorical PCA/MLP failure does not test the original
OI-CmV2 geometry architecture. Ref10 explicitly authorizes intended FK point
flow and physics innovation. Question: does geometric action inductive bias
improve held-environment I/action contrasts over both categorical and native
continuous controls, independently of a stronger baseline/fitting contract?

Cheapest discriminator uses the same854 ref7 windows, ref8 train688/test166,
125/31 environments, E12/I14 step8 and eight Y9..32 heads. All failures stay.
Root chooses a fixed exact ridge on physical surface motion before a larger
spatial network: this isolates representation cheaply and makes optimization
error irrelevant. This is a geometric-input Probe, NOT the original spatial
OI-CmV2 network, NOT ref5 point-flow output E, NOT policy utility. A negative
here cannot close the core Cm/point-network hypothesis. A positive warrants
a separately frozen spatial/candidate mechanism experiment; not immediate PPO.

Resource: idle GPU6, timeout600s, outputs≤100MiB; tiny synthetic FK engineering
tests on CPU because GPU startup dominates. Statistical OLS/bootstrap on CPU;
all real FK/PCA/predictor/scorer fits and inference on GPU. Stop on source/data
drift, nonfinite, FK mismatch>0.1mm, environment/OOF leakage or resource conflict.
User requires independent review for anomalous results: reviewer checks native
PD/FK contract now and actual artifacts after the run. No external authorization
needed; Mission, claims and resource boundaries unchanged.

## Decision-time physical action contract

Native q=history[-1,:18] includes actual measured followers; base_action is
the decision-time source actor action. Exact DExplore clamping/scales/mimic
coupling generates all15 PD target candidates; all15 have preassignment
headroom. FK transforms include measured actor hand_root, preserving the wrist
chain. Compare FK current handbase/true tips to before measurements. Sampling
is area-based then deterministic24points per finger,120 total, covering the
proximal/intermediate/distal regions represented in I. Seed241; saved local
points/link IDs guarantee correspondence across candidates.

F_total(a)=P(FK(target(base+delta_a)))-P(FK(current_q)). Baseline F_total(0)
is generally NONZERO and shared as state by every arm. Geometric action is
delta_F(a)=F_total(a)-F_total(0). Express point/vector in decision object frame;
include signed delta_F dot current surface normal. No actual future q, tips,
actions, PD targets or feedback sequence enters inputs. Nominal endpoints are
NOT an executed eight-step sweep and are NOT multiplied by8. Decision-time
base action is fixed in representation; actual feedback base_t varies over8
steps and remains unmodeled. World-axis before state remains available for E.

Baseline E: constant world linear/angular velocity for8control_dt, short
rotation representative, zero velocity innovation. Baseline I: exact current
force-log/proximity/global-proxy features. Raw target minus this baseline is
a statistical innovation, not an identified causal treatment effect.

## Frozen matched fitting and OOF contract

Shared H104 retains ref8 exact full-fit PCA basis + current physical72;
OOF fits ref9 helper's PCA only on fold training environments. Shared geometry
state=(current120surface positions/normals + baseline nominal point flow),
train-only PCA16. Action32slots: zeros State; standardized14onehot Arm;
standardized18PD_target(candidate)-PD_target(base) Joint, padding zeros;
Flow PCA32 of120point incremental xyz +120normal components. Fit geometry
PCA on raw metre coordinates; no pre-PCA per-point whitening. All projected
scores normalized on fit rows only. State-action products=H's first16 ×32
action slots; same664 design columns for every predictor. Fixed α32 exact
centered float64 ridge, unpenalized intercept. Target residual scaled by
fit-only raw E/I std (floor.001). RawState/RawArm same x, ridge, target scale
predict raw E/I, isolating baseline subtraction from representation changes.

FlowShuffled assigns a permuted arm within wave×motion×phase-quarter,
separately fit/hold/test, seed242. It REGENERATES flow at the recipient state's
q/base/root, rather than moving a donor's state-dependent geometric feature.
Also test frozen Flow with permuted test arms. PCA fitted on factual fit rows
and reused for candidate/shuffle; held data never changes it. No tuning.

Three environment OOF folds seed241 inside outer train; fold H/geometry/action
PCA/norms/target scales fitted only other-fold environments. Save source/hold,
permutation, weights, normalizers, predictions. Downstream train always OOF,
test full-fit. Same α32 scorer: common H plus decision-known physics baseline
for ALL arms, matched action/product26consequence slots. H/Arm/Joint/Flow,
GT, P_State, P_Flow, Flow_P_Flow. Main comparison P_Flow or Flow_P_Flow against
direct Flow, not merely weak H or previous overfit Ha. Report all task heads.

## Predeclared decisions and evidence boundaries

A: Flow I gain over State≥5% and environment-bootstrap lower95>0;
also Flow I gain over trained FlowShuffled≥3% and frozen Flow test-action
shuffle error penalty≥3%. These latter two are point-estimate Probe screens;
their confidence intervals must also be reported, not silently upgraded.
B: same-state15candidate contrasts against ref8 current-adjusted factual GT
fulltest rank14, raw I corr≥.5/sign≥.65/gain-vs-zero≥10%, AND arm-centered
I gain-vs-zero>0. Also report seven within-pair +/− contrasts to cancel shared
zero offset. Unlike ref9 this shared-reference sensitivity is PREDECLARED.
Geometry-specific: Flow I gain≥3% over BOTH Arm and Joint. Report paired CI,
do not interpret marginal significant comparisons as definitive model ranking.
Representation PROMISING only A+B+geometry-specific; otherwise UNPROMISING
with rank support (or UNCLEAR without it). This is a fixed linear-feature
screen, not a closure of nonlinear spatial modeling. C unique transfer:
P_Flow or Flow_P_Flow ≥5% primary gain over direct Flow with lower95>0,
and physical-failure gain≥−2% (point-estimate safety screen, not certified safety).
No selector unless separate mechanism evidence, C and further validation.
Report E12/I14/joint MSE vs TrainMean, Persistence, SimpleDynamics, RawArm;
OOF quality, train/test gap, bootstrap CIs and same-budget GT retained R.
Bootstrap242,2000 environment resamples, fixed fitted models, not refitting CI.

## Results and review

Pending. Preserve nominal per-finger surface displacement and measured ref7
perturbation audit in this card, with explicit units/reference distinctions.

## Artifacts and limitations / future evidence

Outputs: `outputs/cm-interaction-oracle/geometric-innovation-s241/`.
Labels/persistence/GT definition unchanged. Single split and fixed α; ridge
effective capacity differs across representations despite identical slots.
Per-finger equal sampling and PCA can discard local contact information.
Different real feedback trajectories/unknown material and load not modeled.
Spatial token network, actual execution prediction, held-sign/finger transfer,
matched trained-policy Cm-on/off and multi-seed Validation deferred until
they can change the next research decision. Original OI-CmV2 remains open.
