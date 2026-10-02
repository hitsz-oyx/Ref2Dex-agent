# Fresh measured-return gradient control variate Decision

Probe, not Validation. Distinguish whether fixed physical Cm derivatives
reduce COMPLETE parameter-gradient noise more than a direct task-Q derivative,
action-removed physical dynamics and a common state-only baseline. If this
cheap Decision fails, do not launch actual corrected policy training or scan
the control-variate coefficient. Positive permits a separate actual matched
learning comparison, not a policy-benefit or algorithm-novelty claim.

Model source: completed option-model-policy-r1 fit/models.pt and the SAME
actor_initial_parameters from its actors.pt. Predictors/V/Q are fixed after
1500updates each on603/604; no fresh return is fitted or used for coefficients.
Report all prior6000model and3000offline-actor steps separately; the trained
actor means are NOT used here. New actor is exact common3555scratchinit with
zero final head, current152 input, raw mean0. P0 before one PRElift-8 decision.

Fresh native panel618,768env/202ticks, original spaced scene and physical105.
Independent per-env XY offsets seed+11000; four balanced randomized arms
seed+16000. Arm0P0, arms1/2/3 IDENTICAL Gaussian raw N(0,I) option behavior:
private CPU seed+18000 draw768x12, use respective environment row, no
antithetic copies or replay pairing. Zero action before decision, fixed raw
held[decision,stop+30), native tanh/scales/bounds/couplings unchanged. ALL576
stochastic episodes included,192eachmotion;192P0 episodes retained as behavior
reference only. No successful-only filtering or future input to derivatives.

Current152 and FIT-only SDK normalization unchanged. Known future21 computed
from public plan atdecision+8; no actual future physics enters c or b.
Shared c(x)=fixed direct-Q(x,raw0). b_cm is derivative wrt raw option of
V(Cm8step(x,tanh(raw)),tanh(raw)); b_off removes ONLY the physical model
action path but retains V's option input; b_directQ is derivative ofQ(x,tanh(raw)).
All gradients evaluated at raw0, detached, coefficient1, covarianceI fixed.
Actual return R is physical105bool. Let epsilon be actual raw option and J
the current actor-mean parameter Jacobian. Compare
g0=J.T@((R-c)*epsilon) and
gb=J.T@((R-c-b.dot(epsilon))*epsilon+b).
Gaussian first/second moments give conditional zero-mean correction; omitting
the +b term biases it. No same-episode baseline/coefficient fit, PPO clipping,
GAE/bootstrapped return or reward normalization. Raw Gaussian is not clipped;
native tanh belongs to the return mapping. Actor parameters remain fixed.

Primary metric: trace of unbiased empirical covariance of COMPLETE actor
parameter-gradient vectors across576episodes. State mixture included, all
parameter dimensions included; advantage/score/output-gradient variance cannot
replace it. At zero final head, hidden-parameter Jacobians are EXACTzero;
independently verify ALL block gradients and use only the nonzero final-head
columns for numerical efficiency, not as a proxy. Retain full Torch gradients.

PROMISING iff Cm trace <=90% EACH of common baseline/off/directQ AND every
paired95% bootstrap interval for Cm-minus-control trace has upper<0. One
predeclared1000bootstrap, private CPU3557, stratified192eachmotion, identical
indices for all four estimators. Exploratory episode intervals only; native
job independence/conditional variance and formal unbiasedness are not
empirically proven by them. No threshold/coefficient/sigma/model/seed scans.

Audit ALL native current/P0/PD/options/full-mesh labels/private RNG; independently
rebuild currentSDK and known plan; NumPy critic forward AND reverse-mode raw
derivatives <=2e-5, every actual Torch actor-parameter gradient <=2e-5 against
independent zero-head NumPy Jacobian. Rebuild full variance/bootstrap/gates
after independently checking gradients; exact statistics use recorded float32
vectors. No optimizer steps, learning benefit or conditional-noise isolation
claimed. Tiny CPU moment smoke before scientific launch, no research data.

Whole<=300s/512MiB, one freshly idle GPU for native/inference/gradients;
CPU independent files/NumPy audit/statistical reconstruction. Original675
inputs and prior fit/models/raw outcomes protected/read-only. New Gaussian
collection and complete-gradient data retained regardless of outcome.

## Execution closure

f0be3ca/r1 COMPLETED,768fresh trajectories/576stochastic gradient episodes,
zero new optimizer steps. Full parameter-gradient trace baseline8.2513/
Cm32.5816/off11.9448/directQ14.0500; both mandatory gates fail, UNPROMISING.
Allnative/input/critic-reverse/complete-actor-gradient/bootstrap audits pass.
206.507s/391985146bytes, protected inputs unchanged, all own PIDs terminal.
[Complete result](../../research/20261002-physical-gradient-control-results.md).
Do not launch corrected policy training or vary the control coefficient.
