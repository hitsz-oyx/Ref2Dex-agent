# P-20261001-hold-plateau-curriculum

Decision/Blocker Probe, fixed by
`docs/archive/2026-10-04-research-governance/decisions/D-20261001-hold-plateau-curriculum.md`. H1: a small explicitly
task-specific reset curriculum and phase reward can produce a frame0 holding
substrate. H0: this exact continuation remains unusable. The result selects
whether to invest in a model-utility screen or reassess feasibility, not whether
a novel method works.

Freeze before physics: actor286e420, generated references from
`P-20261001-hold-plateau-reference-r1`, training seed721,300 NEW epochs,96env,
horizon16,4 PPO passes, minibatch512, learning rate1e-5. New optimizer; model,
critic and both observation normalizers copied from source. Final endpoint only.
Expected new steps460800; checkpoint epoch300/frame460800. Save only final
`GRAB.pth`; no intermediate seed/epoch selection. Include raw reset counts and
training phase-held fractions, but do not use them as success claims.

Reset probability=.5 through80, .5*(200-epoch)/120 for80<epoch<200,0 thereafter.
Otherwise frame0. Phase start is first inserted stationary row73/51/63. Use native
loader reference joint/object pose, and loader zero velocity. Reset hook changes
training initialization only. Keep original reference reward plus10*bounded
contact-supported3cm lift in the phase; no extra reward after intended release.
Source normalizers adapt during PPO; they are frozen during evaluation.

Test seed500/501,192env each,64per motion, frame0, no early termination/kappa.
Complete every first episode. Same frozen75 consecutive phase physical steps
>=3cm over own initial native height with both force proxies;45-step secondary.
PROMISING iff75 rate>=10%overall and>=5%per motion, elseUNPROMISING. No source
control superiority claim, no validation confidence claim, no Cm policy claim.

One GPU<=3000s, training<=1800s, output<=1GiB. CPU labels/file/hash operations;
GPU inference/training/native simulation. No source checkpoint or data writes.

Prelaunch exploration audit: source fixed sigma is exp(-2.9)=0.055m for
additive wrist translations, larger than the earlier10mm interventions. This
recipe freezes translation standard deviation0.005m before any PPO interaction;
all other sigma entries retain exp(-2.9), and deterministic source means/critic
are copied exactly. Record this deliberate fixed exploration change separately
from source initialization hashes. No sigma fitting or outcome-based tuning.
