# Force-aware action-conditioned non-gravity impulse information

Decision Probe: can a weight-normalized CURRENT force representation and a
physical prior expose useful action-conditioned impulse prediction? Positive
permits model gradients into actor; negative closes this exact impulse recipe.
No actor updates or claim of Cm utility here. This differs from the closed
critic-only objective and the closed70-input Gaussian-moment field.

FIT all20training547--566, allthree stochastic behaviors. FreshTEST575 ONLY,
same actual u20policies with stochastic requests,768env/202ticks. No old final
568/569 or response-field TEST570 in fitting/selection. Current eligibility:
original105reference ticks, currentPRE rootrise>=.03m and fullmeshclearance>=.02m,
tick>=1so current raw forces are observed. No future-success/contact filter.
Require FIT>=10000eligible transitions and TEST>=1024/32episodes, elseUNCLEAR
and stop, no alternate seed/window. ALLeligible motions/behaviors remain included.

Three normalized world non-gravity impulse coordinates:
`Y=(v_next-v_current)/(abs(g)*dt)-g/abs(g)`,dt1/30s, native g=(0,0,-9.81).
Mass cancels in label normalization but actual per-env weight scales forces.
Native object linear damping.01means Ycontains damping, NOT exact pairwise
hand-contact impulse, contact truth, force closure or task value. Approximate
force/damping persistence prior `P=F_current/(m*abs(g))-.01*v_current/abs(g)`.
Current forces are previous POSTtrace: object3 and five hand bodies15channels.
Never force after action as input. Preserve raw force/weight and clock provenance.

103inputs: existing normalized current70;18weight-normalized current forces;
3physicalprior;12executed independent target-minus-currentq/absPDscale. Extra
21channels get FIT-only mean/std(floor.001),clip10; original70uses frozen base
normalization/clip10. Executed action12unchanged. Residual Y-P gets FIT-only
mean/std(floor.001), invert predictions as P+mean+std*network_output.

Three identical103->64ReLU->64ReLU->3networks, seed2191, zero last layers,
same private batch seed2192, Adam3e-4,2000updates EACH,4096samples with replacement,
global gradient norm clip10. Cm receives allfeatures; state-only zeros12actions;
global-action replaces70currentstate features by FIT motion/known-phase means
while retaining LIVEcurrentforces/prior/actualactions. Known four phases:
beforelift, lift..stop-75, stop-74..stop, afterstop; emptyFITcells0. This strong
global control is more than an action-only mean. No model/step/temperature selection.

Primary TEST mean squared Yprediction error averagedthreecoordinates, same
actual rows for allmodels plus physicalprior. Require Cm<=.99*EVERYthreecontrol
errors AND upper exploratory95%paired(Cm-control) interval<0forALLcontrols.
2000bootstrap resamples192common-stream episode indices, retaining allthree
replicas and their eligible tick counts jointly, seed2193. Zero-eligible
episodes stay in denominator bookkeeping. One learner sequence/environment
panel is Probe, not formal independent training Validation. Fixed gates. The1%operational threshold is a cheap permission-to-invest
screen for small std.05request perturbations, not a publication effect size;
actual matched policy utility remains indispensable.

Unique committed run; source/raw/cfg/base/nativeSDK/taskasset SHA pinned. Native
audit permits ONLY575/u20/stochastic while all original native/PD/mesh/causal
checks remain unchanged. Check label algebra/no-future-force, full predictions,
FIT normalization/template/priors, common-cluster statistics independently.
Meaningful CPU conservation/causality smoke before GPU launch. Single freshidle
GPU4for native/modelfit/inference; CPU file/stats and independent NumPy audits.
<=1800s/1GiB new data; stop ownedchild only on drift/nonfinite/contention/budget.
Failure ends exactmodel/features/target/step/seed recipe without local scans.
