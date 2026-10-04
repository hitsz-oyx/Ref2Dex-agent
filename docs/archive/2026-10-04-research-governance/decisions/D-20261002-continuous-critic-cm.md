# Different integration: continuous policy learning with a causal dynamics critic

Static correction and event-reflex routes have no useful fixed gate result:
first learned macro heads choose identical tested actions; natural event control
fails; independent positive-finger corrections all fail motion2. Stop those
families. This does not establish that feedback policy optimization in the full
independent action space has no potential.

Choose an ACTUAL continuous policy-learning comparison, not another static arm,
predictive-loss gate, or controller-only paper claim. Freeze the successful
scratch reference-target actor as a base and learn bounded target residuals on
ALL202native ticks. Twelve independent native coordinates: wrist0..5, finger
parents6/8/10/12/14/15. The six dependent child commands are reconstructed and
not sampled as independent Gaussian policy dimensions.

Policy head70->64ReLU->64ReLU->12mean, zero final weights/bias; trainable12logstd
initial log(.05). Identical scratch seed762and base/statistics for all learned
policies. Sample Gaussian REQUESTS, transform tanh and fixed scales
[.02m]*3+[.10rad]*3+[.15rad]*6, add to the evolving base target, project wrist
into native executable q+/-PDscale and fingers into dependent/native limits.
PPO likelihood is over the saved Gaussian requests, not a density over clipped
many-to-one native targets. Mean-only final evaluation uses the same projection.
Source official checkpoint is bootstrap only and never acts or initializes heads.

Three matched policies, separate state-only V critics and identical initial
parameters: Cm critic auxiliary predicts one REAL next-physics object transition
given current normalized70features and executed independent target-minus-q12;
state-only auxiliary zeros those12action features but predicts the same target;
no-aux critic runs identical auxiliary forward/backward work with zero weight.
Separate actor and critic encoders: auxiliary gradients never directly update
the actor. This is established auxiliary-task PPO, not methodological novelty.
No old failed frozen probability bank or macros are reused as the new model.

Physical auxiliary target is next-minus-current object worldXYZ/.005m and
linear velocity/.05m/s, six coordinates. Executed target and current state are
available BEFORE that physics step; next_q, future measured hand flow, success
labels, and later adaptive actions never enter the input. Critic V is STATE ONLY;
action-dependent predictions must not become an uncorrected actor baseline.
The only task reward is actual ORIGINALphysical105 at trajectory end, broadcast
as Monte Carlo return with gamma1. No predicted/model reward, bonus or surrogate
success criterion. State normalization reuses frozen base FIT statistics,
clipped10. Gaussian request logstd clips[-5,0]in probability computation.

Before any native learning run: implement and test the request probability,
PPO clipped gradient, critic input, aux-to-actor gradient isolation and native
projection; audit complete physical labels/provenance and actual actions. A
small bounded engineering smoke proves wiring only, not policy usefulness.

Prospective design proposal (must become a committed complete runner/card before
launch):20fresh panels547--566,768env/panel,4preassigned groups64/motion:
unchanged base and3learned policies,192trajectories/variant/panel. Equal per-policy
interactions, four PPO epochs per panel, batch1024, Adam3e-4, ratio clip.2,
value weight.5, entropy.001, dynamics weight.05(Cm/state-only), global gradclip.5.
Final-only568/569(two fresh seeds outside training),1536trajectories. Primary
all-motion Cm gains>=5ppover BOTHlearned controls and is no worse than unchanged
pooled; each seed no worse than BOTHlearned controls. One optimization seed is
Probe only. Failure ends exact recipe without coefficient/update/head/seed scans.
Base19392teacher rows and all training/model compute reported separately.

One freshly admitted GPU, whole run<=3600s/6GiB, within campaign; no physics,
external project or checkpoint overwrite changes. Native asset/target/PD/raw
current-state/likelihood/physical105 audits mandatory. Preserve all panels,
failures and final-only checkpoints. Generic critic auxiliary success would
only justify deeper method/Validation work, not the journal claim. Journal
objective and Cmcausalutility remain active and unproved.
