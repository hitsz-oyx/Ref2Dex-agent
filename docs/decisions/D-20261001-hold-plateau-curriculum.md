# Decision: establish one holding baseline before more model utility

Question: can a small task-specific PPO continuation make the explicit synthetic
hold task a usable substrate, or is this task variant still unsuitable?

Evidence: 768 complete actor-only first episodes,5 retained75 (0.65%), all frozen
gates fail; reference/velocity/progress/endpoint audits pass. Previous greedy
one-step model corrections failed retention despite short-height effects. More
updates to those predictors would not resolve the current baseline deficiency.
This route review ends the repeated prediction/control refinements.

Action: ONE baseline Probe. Initialize from fixed self-trained actor286 (lowest
source seed, not the better held actor287). Restore model and its two observation
normalizers, start a fresh PPO optimizer/epoch counter. Train seed721 for exactly
300 new epochs,96 envs,16-step rollouts (460800 new native interactions),4 PPO
passes, minibatch512, constant1e-5. Only the final300-epoch model is evaluated.
Reference reward stays; add10 times contact-supported bounded lift ONLY during
the90-row plateau. No Cm/auxiliary model, action correction, or selected checkpoint.

Training starts: 50% at the first stationary plateau row through epoch80, linearly
reduce plateau-start probability to0 at epoch200, then100 further epochs from
frame0. Plateau starts use copied native reference robot/object pose and zero
velocity; they are privileged training initialization and never test evidence.
No altered masses, friction, collision masks or object resets based on outcomes.
Native early termination/adaptive kappa are off throughout. Three motions remain
balanced. Evaluate final actor from frame0 on fresh500/501,192env each, using the
same actual-progress75-step scorer and pooled>=10%/each-motion>=5% feasibility
gates. Training rewards and elevated reset episodes never count as task success.

Cost/stop: one freshly admitted idle GPU, whole Probe<=3000s, train<=1800s,
storage<=1GiB, no video. Stop on drift, nonfinite states, incomplete evaluation,
wrong step budget or source modification. Preserve a failed run without restart
or learning-rate/duration tuning. If PROMISING, freeze the resulting baseline and
design a separate matched model-utility Probe. If UNPROMISING, end this specific
continuation and reassess task/actuation feasibility at a higher level. No external
authorization required by the user's autonomous request or campaign limits.

Prelaunch exploration audit: source fixed sigma is exp(-2.9)=0.055m for
additive wrist translations, larger than the earlier10mm interventions. This
recipe freezes translation standard deviation0.005m before any PPO interaction;
all other sigma entries retain exp(-2.9), and deterministic source means/critic
are copied exactly. Record this deliberate fixed exploration change separately
from source initialization hashes. No sigma fitting or outcome-based tuning.
