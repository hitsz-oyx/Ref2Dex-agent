# Decision: test policy learning rather than another surrogate gate

The nominal forecasting primary, corrected-gradient screen and four-bin position
policy fail; the new post-lift disturbance recipe has too little recovery headroom.
All labels remain UNPROMISING and both finite calibration families end. Their
failure does not answer the mission's matched trained-policy utility question.
Cm still provides descriptive conditional forecast information, especially versus
state-only, but useful forecast loss and gradient second moments are insufficient
for policy utility. No theorem or methodological priority is claimed.

Choose a DIFFERENT integration: frozen short-physical predictions as features for
a policy learned from REAL physical105 rewards. Stop gradient-control-variate
integration, direct greedy model selection and further forecast fitting. This
one bounded learning comparison can answer the mission more directly than another
new information metric. Reusing the frozen forecaster as a feature does not
upgrade its failed primary; fresh policy-training and evaluation cohorts are
required. It is established feature transfer, not a journal-ready method.

Frozen task: original initial-XY+/-10mm variant and eight WORLD-axis executable
primitives,202native ticks, same scratch reference-target initializer15ed218f...,
physical105 from stop-74..stop+30. No new force, object pose/orientation change,
source actor action/weight use, teacher fallback or altered task outcome.

Three matched macro policies: Cm physical30 probability vector over all8actions;
state-only physical30 vector (same zero-action predictor for each action); and
privileged FIT motion/action mean vector. The extra8features are probabilities
mapped2p-1, no logit temperature or fitted rescaling. Current69features use the
same FIT-only mean/std, clipped10. All77-input64/64ReLU/8-logit heads share seed752,
same weights, zero final weights and bias[2,0,...,0], so the initial policy is
identical. Cm/state inference uses the same8queries; the global control executes
the same predictor work and substitutes its FIT-only probabilities, so compute
differences do not silently define the comparison. All physical predictor weights
are frozen fromr2; no forecasts or actor checkpoints are selected by evaluation.

Prospective training529--540: twelve fresh768-environment randomized panels,
all3motions, balanced32per primitive/motion, same private placement and assignment
laws asr2. Shared off-policy data for ALL policies, known behavior probability1/8;
state occupancy before the one macro decision is independent of every trained
head. After EACH fresh panel, exactly ONE full-batch policy-gradient update per
head: actual physical105 return, importance pi(a)/(1/8) detached in the score
loss, common frozen FIT per-motion mean physical105 baseline, constant across
actions. An action-dependent baseline without an exact correction is forbidden.
No model reward, greedy
ranking, PPO clipping, repeated fitting on a panel or control-variate correction.
Adam0.01, gradclip10, no entropy term,12updates, final-only. Models start before
the first panel; each update is conditioned on fresh independent assignments.
Data, updates, optimizer, baseline, interactions and actor inference are matched.
Record pretraining4608trajectories and all extra model compute separately.

Prospective evaluation541/542: each768env, balanced64per motion for four
preassigned groups: unchanged initializer, trained Cm, trained state-only and
trained global. At the existing decision tick, each trained head selects its
deterministic argmax once, and the frozen baseline actor executes that primitive
thereafter. This is a TRAINED macro policy, with reference-conditioned base
manipulation; it is not general continuous RL, reference-free grasp or hardware.
Retain all1536trajectories. Primary all-motion physical105: Cm exceeds BOTH
trained controls by>=5percentage points pooled and is no worse in EACH seed.
Report initializer and all motions, never replace the primary with motion2.
Single optimization seed remains a Probe, not Validation; no hidden head/LR/seed
sweep. Positive result permits independent multi-training-seed Validation;
failure stops this exact feature-transfer learner without extra updates.

Implementation remains to be made and audited before launch. One admitted GPU,
<=1800s/3GiB, within campaign. Save current-state features, pre-update heads,
exact probabilities/propensities, gradients, updated heads, full native physical
traces and hashes; independently reconstruct rewards and evaluated actions. No
external approval boundary is crossed. Distinctive method and generalization
remain separate journal-readiness requirements even if this Probe passes.
