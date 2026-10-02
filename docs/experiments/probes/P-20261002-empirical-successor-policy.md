# Actual policy training with a learned observed-successor law

Decision Probe, not Validation. Frozen FIT603/604 from option-model-policy-r1
ONLY,1536actual episodes. Exclude618/oracle diagnostic and all EVALdata from
models/actor fitting, support construction or model selection. Existing
SDK80current+futureFITnormalizers/P0context70norm fixed, current152 and future
152 unchanged. Public knownfuture21 analytic,131unknown physical observations.

Atomic support is ALL1536FIT complete unknown-future131tuples, no successful-only,
contact/return-selected, motion-filtered or smallest-error subsets. They are
complete observed NORMALIZED feature tuples, not guaranteed world states for
the new query. At value composition replace knownfuture21 by the public query
plan/clock; no donor futureplan or future information is deployed.

Conditional probabilities: shared query/key encoder164->64ReLU->64ReLU->16
linear, inputcurrent152+tanh(rawoption)12. Query and every donor's ACTUAL
current/option are embedded by SAMEnetwork. Logits=-squared latent distance/16.
Softmax overALL1536support atoms, exclude own trajectory at BOTH physical-model
and actor fitting. Physical-action-removed control zeros BOTHquery and donor
action12; Vstill retains actual proposed option, an action-aware strong control.
Same parameter count/init3560, all learned model weights used, no explicit
motion index or extra known class feature/mask. Public plan remains in152for
all actors and directQ. No task return enters these physical-model losses.

Fixed physical kernel logscore for observed target j versus donor k:
`-sum((future_j[DYNAMIC]-future_k[DYNAMIC])^2)/(2*.5^2)`.
Train negative logsumexp(logweight+kernel_logscore), same own-episode exclusion.
This kernel observation score trains an approximate atomic law; do not call it
calibrated transition density or exact Gaussian-mixture value integration.
Support atoms remain observed centers; no added Gaussian physical-state noise.
1500Adam3e-4/weight_decay1e-4/gradclip10updates EACH, shared privateCPU3564batch256.
No anchor encoder initialization or model selection from labels/TEST.

Existing fixed shared V(actualfuture152,tanhoption12) and directQ(current152,
tanhoption12), each1500valid603/604steps, are loaded and frozen without refit.
Cm actor score=SUM learnedweight*V(complete observed physical atom + public
future21,proposedtanhoption). Off actor uses action-removed weights and SAMEV.
No mean successor fed to the learning value objective. For diagnostics only,
record V(weighted mean physical tuple) on fixedFIT actual actions; it cannot
choose a checkpoint or substitute actual policy outcomes. No actual mean-control
actor trained now; a mechanistic claim would require it after utility passes.

Cm/on and off actors current152->64ReLU->64ReLU->12tanh raw mean[-1,1]. EXACT
initialparameters from original common3555scratchactor,1000steps each, identical
privateCPU3556batch256,Adam3e-4/clip10; objective -expectedprob+.05*mean(raw^2).
Freeze physical models and Vbefore actors; probability and explicitV action
paths differentiate through actor. SAME initialization/data/steps/reg as the
retained direct-Q actor whose1000updates are reused, never repeated. Finger,
wrist and native P0goal/couplings/bounds/scales unchanged. Sources'6000model
and3000actor steps separately reported, including unused older models/actors;
new costs3000physical/2000actor steps, not free pretraining/sample efficiency.

New EVAL623/624,768env/202ticks each, randomized independent placements/arms
P0/Cm/off/directQ,384episodes perarm pooled/128motion/192seed. Deploy ACTORS
ONLY, before decision P0; at PRElift-8current152->one raw option held until
stop+30exclusive. No support/retrieval, model/value, actualfuture or task return
at deployment. DirectQ new deployment weights EXACTsourcecheckpoint.

PROMISING iff Cm>=5pp EACHof P0/off/directQ, noninferior toALLthree on EACH
seed, and motion1 no worse thanP0 by>5pp. ALLmotions included. No gate/seed/
support/kernel/width/temperature/steps/penalty/rawscale/horizon/model scan to
rescue failure. Forecast/loss/mean-composition diagnostics cannot rescue it.

Full FIT raw current/futureSDK/knownplan/normalization independently verified;
full final encoders and learned probabilities/kernel observation score at ALL
FITactual actions independently rebuilt with own masks, rowsums and action-off
input. Vsum and Vmean outputs compared with separate NumPy composition<=2e-5.
Actor final fullFIT NumPy <=2e-5, controlweights bitwise source and commoninit
exact; no independent optimizer replay claimed. All native current/P0/options/
PD/fullmesh105/privateRNG audits bothEVALpanels; allactual actor observations
and recorded forwards checked. Tiny CPU law/gradient seam beforelaunch.

Whole<=1200s/1GiB, freshly idle oneGPU for fit/native, CPU independent files/
geometry/NumPy; original675inputs/sourceFIT/V/Q/checkpoints protectedread-only.
Bounded owned process guards; preserve failures, no overwritten checkpoints,
original/external writes, unknown kills, remote push or delegated sessions.
