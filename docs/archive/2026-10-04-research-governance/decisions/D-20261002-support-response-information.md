# Decision: conditional executable support response before model-assisted RL

Question: on states visited by our frozen reference-target actor, does an
action-conditioned short-horizon physical model predict support better than
state-only and per-motion/per-action average responses? This is the next cheap
decision before matched policy training, not another baseline optimization.
Initializer596c5a0has motion1physical10556/64; freeze checkpoint15ed218f...
Broad physical-support prediction is established methodology, not a novelty claim.

Choose a new prospective task variant: object initial XY independently uniform
within+/-10mm, no Z/orientation/velocity/property changes. Apply exactly once
before our first physical tick, record before/after, then no object state writes.
Random offset generator eval_seed+11000 is separate from assignment+12000.
Every panel has768env,256per each of3motions. Assign8executable primitives
equally32per motion BEFORE physics: unchanged, curl-.30/-.15/+.15rad,
wrist target X-/+10mm, wrist target Y-/+10mm. Canonical null commands and native
finger bounds/coupling retained. Intervention starts at current physical progress
first-reference-lift-start-8 and persists under the frozen actor/reference plan;
its commanded native action is recomputed each tick, not held open-loop.

Fit cohorts519--524 (4608rows), test525/526 (1536rows), all202ticks retained.
One decision row per environment contains current69features and3primitive
parameters, with no future physical observation or outcome in input. Features:
native q3:18,dq18,object position minus native translation q0:3,object quaternion,
object linear velocity minus dq0:3,object angular velocity,current force proxies,
planned q minus current q,planned phase fraction,current root rise from actual
frame0 and current fullmesh clearance. These differences are coordinate features,
not an assertion that a DOF coordinate is an exact physical wrist landmark.

Primary physical target: root rise>=30mm and fullmesh tabletop clearance>=20mm
at ALL last10ticks of the next30physical ticks, progress decision+21..+30.
This is a short-horizon physical forecast under specified continuation, not
long-term action value. Physical105 is independently reported as a secondary;
no future force proxy defines either label. All three motions remain reported.

Two identical72-input,64/64ReLU scalar-logit MLPs: Cm receives normalized current
69features and primitive scaled by[.01,.01,.30]; state-only receives zero in
three action slots. FIT-only mean/std (.001floor,clip10); same initialization741,
same mini-batch sequence,1000Adamupdates,.001lr,batch256,gradclip10,finalonly.
Privileged global-motion/action control uses Beta(1,1)smoothed FIT probabilities.
No model or temperature selection. GPU fits/inference; CPU geometric/label audit.

Preselected motion1 primary gates: eachclass>=10%in EACH testseed, Cm logloss
>=10%lower than BOTH controls pooled, and lower than both in eachtestseed.
Report pooled/allmotions too. Failure stops these exact models without additional
updates, selected arms, or another seed. Success permits a matched policy-training
design with this same actor initializer and fixed task metric; it is not utility,
formal Validation, generalization or journal readiness. One admittedGPU,
wholeprobe<=1800s/2GiB, no external authorization needed.
