# P-20261001-randomized-task-selection

Decision Probe, frozen before fresh acquisition. Hypothesis: conditional response
selection from the simple factual control improves retained task success beyond
the native actor and state-independent/random controls. This model is a baseline;
a positive Probe is a route signal, not a journal contribution by itself.

## Frozen models and local decision

Use ONLY factual1000-update checkpoints611/612/613from
P-20261001-direct-randomized-response-fit-r1. At current state/reference command,
average their physical WORLD one-step residual predictions equally for candidate
arms0zero,1x+,2x-,3y+,4y-,5z+,6z-. Candidate offsets are exactly0.01normalized
wrist translation, same PD scale as previous probes. Score by predicted vertical
object residual, choose maximum with lowest-index tie break. No test outcomes,
reference motion IDs or future physics enter the score. Canonicalization and
fit-only scales remain frozen.

Global control uses seven arm-specific IPW FIT means of the observed WORLD
vertical residual, using all3072fit rows and their known original propensities.
This differs explicitly from the previous three-axis contrast-mean control:
action selection needs all seven factual candidate scores. Never calculate it
from new test outcomes. Random control chooses uniformly among valid candidates
with a separate per-environment CPU generator (seed21000+evaluation_seed+env).
Native-actor control always chooseszero. Mask candidates that would clip wrist
commands;zero remains valid. Same candidate validity rule in every policy.

## Fresh trial, correction budget, and outcomes

Actors286/287 × new initial seeds496/497;four768-env panels,256each motion.
At reset independently IID assign one of four policies,probability1/4, using
separateCPU Generator(11000+evaluation_seed),before first physics. Keep all
environments in the intention-to-treat analysis, even without a qualifying event.
Do not rebalance groups. Record assignments and actual policy/correction counts.

At even current ticks>=10, qualify both independent force proxies ANDfull sampled
hand1538/object1024gap<=20mm. Each environment gets at most10decision events,
spaced>=6physical steps (5Hz at30Hz), within its first native episode. A zero
selection still consumes a decision event. One selected wrist offset applies
for that single physics step; the actor remains closed loop. Compute the same
geometry and all candidate model scores for every policy group, including unused
control scores, to match inference/geometry access. Frozen actor/RMS parameters
and RNG are protected during controller/geometry initialization; no model fitting.

Follow each environment to its own first native termination, at most700steps,
with the same early-termination-disabled/frame0setup as the existing paired
physical evaluator. Finished environments may reset for simulator maintenance,
but never contribute later episodes or controller corrections. All768first
episodes must complete in every panel; no favorable complete subset is retained.

Primary retained success: >=45consecutive steps (1.5s) at>=3cm above initial
object height ANDboth force proxies, then no detected drop before first-episode
termination. Use existing HoldTracker contract unchanged. Drop-after-success
means falling below2cm or losing the conjunction of the two contact proxies for6steps after45-step
success. Report stable success, retained success, drop-after-success, maximum
hold duration, mean lift, correction count and inference wall cost separately.
These proxy metrics do not establish attributed finger-object contact/hardware.

## Fixed scoring and decision

No optimization seed selection: the controller is the specified equal-weight
ensemble, not a claimed independent three-policy replication. For each of the
six evaluation-seed/motion strata, average observed complete-episode outcomes
within assigned policy across both actors; weight strata equally. Report
conditional-model minus each control in percentage points. Use2000descriptive
bootstrap replicates,seed12026,resampling shared environment IDs across actors
inside each stratum, retaining their policy assignments. Every policy must have
>=32assigned complete episodes per motion in each panel; otherwiseUNCLEAR.

PROMISING requires retained-success point improvement>=3pp ANDone-sided95%
LOWER bound>0against EACH actor,random,globalcontrol, plus one-sided95%UPPER
drop-after-success increase<=2ppagainst the actor. ElseUNPROMISING. Report
all actor/seed/motion subgroups without replacing pooled gate. First-step lift
or a favorable video cannot upgrade failed retained success. Positive justifies
independent tasks/objects and a novel-method proposal; negative ends this exact
one-step vertical-selection controller without changing its dose/budget/gate.

Resources:GPU4sequential,<=3600s/2GiB, at most2of our GPUs concurrently. Freeze
controller sources, seven global scores and all model/physical input SHA before
any new physics. No PPO, no external write, no checkpoint replacement.
