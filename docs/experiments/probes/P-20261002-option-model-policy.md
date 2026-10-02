# Short successor Cm in actual offline option-policy training

Decision Probe, not Validation. Does an action-conditioned short physical
successor improve an ACTUALLY trained option actor over action-removed dynamics,
direct task-Q policy training and unchanged self-trained P0? No same-state
replay assumption, retrospective maximum, physical-motion reward or old failed
model is used. Distinctive methodology/journal readiness remain unestablished.

Fresh FIT603/604, each768env/202ticks, original spaced scene/envSpacing5.
Four balanced randomized arms: P0, duplicate P0, standard Gaussian12 joint
option, antithetic option. Private independent per-env XY +/-1cm draw uses
seed+11000 (768draws, NOT copied placements). Private arm permutation+16000;
option draw192x12 CPU seed+18000 mapped by within-motion/arm sorted indices.
All arms P0 before PRElift_start-8. Raw option held from that decision through
PREstop+30 exclusive, then P0. Native tanh/scales(.02XYZ,.10rotation,.15finger),
bounds/couplings/PD remain unchanged. source_e260/u00 construction only, no
source actor or old request-policy actions. All actually observed states used,
no successful-only/contact/future-state selection. Old601/engineering/TEST
data excluded from FIT/model selection. No extra FIT panel after a poor result.

At decision t, current152=normalized P0context70 + knownalive1 + remaining
taskticks/202 + SDKextra80. The80observations are EXACTLY the previously audited
measured body positions15/relative rotations30/net forces18/prior barrier
increments2/relative-position flow15, using only POST[t-1]/POST[t-2]. Net forces
and SDK origins are not pairwise contact truth/force closure/contact points.
8-step successor is ACTUAL PRE(t+8), computed from POST[t+7]/POST[t+6]. Known
next reference18/phase1/alive1/remaining1 must be independently rebuilt from
the public plan/clock, not future physics. Both decisions are BEFORE the first
105criterion tick; knownalive=1. Target actual terminal physical105 bool/gamma1
under continuation with the SAME held option, explicitly conditioned on it.

FIT-only SDKextra mean/std uses pooled current+8-step observed extras, floor
.001/clamp10. P0normalization fixed. Model target is normalized physical
successor residual on131unknown dimensions (current0:51 and72:152), with
FIT-only residual mean/std floor.001.21known plan/history/clock dimensions
are analytic, not learned. Two matched164->64ReLU->64ReLU->131 linear predictors,
common init3551/zero final layer: Cm takes state152+tanh(rawoption)12; dynamics_off
zeros option12.1500updates each, common private CPU batch3552/256 transferred
GPU, Adam lr3e-4/weight_decay1e-4,clipnorm10,MSE standardized residuals.

Shared continuation V(actual future152,tanhoption12) and direct Q(actual
current152,tanhoption12): same164->64ReLU->64ReLU->1sigmoid, commoninit3553,
1500updates EACH, identical batch3554/256, Adam3e-4/weight_decay1e-4,clip10,
BCE on terminal105. Physical predictors/value/Q then frozen. Composition
uses analytic known future21, learned physical residual131, predicted physical
features clipped[-10,10], and the SAME option in continuation V. This is an
8-step option-conditional value model, not an unconditioned future-policy value.

Three actors cm/dynamics_off/direct_q: current152->64ReLU->64ReLU->12,
raw mean=tanh(output) bounded[-1,1], common scratchinit3555/zero final layer.
Each1000actual offline policy-optimization steps, common batch3556/256, Adam
3e-4,clip10; objective -estimatedterminalreturn +.05*mean(rawmean^2).
Same physical data, actor inputs/init/steps, observed-state preprocessing and
coverage regularization. Cm-off removes only the physical model action path;
continuation V retains its option input, making it an action-aware control.
Direct-Q actor is an additional strong model-free comparison; parameter and
pretraining-compute differences reported. No fabricated policy gradients,
label-imitation targets, deploy-time model selector or optimizer reruns.
Final evaluation deploys trained ACTORS ONLY, no future information or Cm/V/Q
at inference. P0 before decision/current152 acquired at decision; predicted
raw request held with unchanged native execution. No rollout outcome selection.

Fresh EVAL611/612,768env/202ticks each, four randomized armsP0/cm/dynamics_off/
direct_q with independently drawn placements. Each trained arm384episodes
pooled/128eachmotion,192perseed. Primary PROMISING iff cm success >=5pp EACH
of THREEcontrols, each evaluation seed noninferior to ALLthree, and primary
motion1 no worse than P0 by>5pp. All gates mandatory; allmotions included.
No same-start pairing, bootstrap oracle, partial-motion rescue, new seeds,
width/update/horizon/action-scale/penalty/threshold/checkpoint scans after fail.
All fits/actors kept whether good or bad; forecast/loss/proxyreturn are
diagnostic only and cannot replace the actual primary policy outcome.

Native independent reset/actualcontext/P0/options/PD/complete-mesh105 audit
ALLfour panels. Evaluation additionally reconstructs current SDKextra, actor
normalization, ALL recorded actor forwards via independent NumPy <=2e-5,
causal decision/start/stop/held requests. Model/actor final full FIT NumPy
forward checks, exact analytic plan and initialization/update-count provenance;
no independent optimizer replay claimed. Tiny engineering seam verifies Cm
input intervention actually changes actor gradients while action-removed
dynamics deletes that path. Original numerical tolerances retained.

One freshly idle GPU per native/fit phase; CPU files/statistics/independent
NumPy geometry/model reconstruction. Whole<=1200s/2GiB, protected sources and
old675inputs, original/external data read-only, no overwrite or unknown kills.
Four3072native trajectories count as pretraining1536+evaluation1536;6000model
and3000actor optimizer steps reported separately. Positive triggers matched
Validation/novelty review, negative closes this exact option-model-policy
recipe. Generic model-based offline RL is established; no publication claim
without real positive utility and further validation.
