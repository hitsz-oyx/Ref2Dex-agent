# P-20261001-recovery-option-learning

HF11 COMPLETED / UNPROMISING, scientific budget1/1closed.
[Frozen design](../../decisions/D-20261001-recovery-option-learning.md).

Question: can the physical recovery signal become a trainable actual option
policy? HF10 retained-height gate stays UNPROMISING; local release differences
motivate this independent learning mechanism. Final C3 remains OPEN.

Frozen trajectory SHA027202015c32ba783aa1bbef5a0a3c501643bfa904e0b460cdf1877193971355,
six self-trained experts/base4, airplane3motions, actual2+8/cooldown6. Trainable
GRU32/shared6option scoring and MC value heads; categorical probability is
softmax(learned logits + log prior). On frozen physical22features/Cm choice,
off zero22features/base choice; both priors .90recommended/.02other5, same
initial entropy and new model initialization9381. Both compute same frozenCm.
Only new option policy/value parameters are updated; old actors/Cm/V unchanged.

PPO .2clip,lr3e−4,gamma.99,20epochs/rollout,minibatch128,entropy.01,value.5,
gradclip1. MC targets are complete first-episode discounted native rewards
rescaled .01 (fixed same on/off); no reward from Cm predictions or old V.
Full history/current independent12channel actions/pre-action context, fixed
normalization with student inputs clamped±10; physical NN inputs unchanged.

Engineering smoke: seed380,one96env complete rollout perarm,2epochs. It checks
finite genuine option logprob/ratio updates, selected cached2step execution,
complete episode MC targets and frozen model hashes; ENGINEERING_SMOKE only.
Synthetic policy and native fake-loop tests14passed on CPU (tiny unit-test
exception); real neural inference/optimization/simulation use admitted GPU.

After smoke pass: seed381,four96env complete rollouts/arm,20epochs each;
update only between rollouts. Equal on/off rollout episode budgets required.
Private sampling/training RNG7381, native simulator RNG separate.
Evaluate independent391–394,96env full first episodes perarm,deterministic
actual option argmax. Initial frozen-prior controllers also evaluate391.
Stable holds>=3cm/contact proxy45ticks then no subsequent2cm/6lostcontact
through episode end; report acquisition/release/five-step hold separately.
No incomplete episode trimming, checkpoint selection or intermediate tuning.

Positive screen requires actual learning and improved stable success vs off,
release noninferiority<=2pp and learned decisions beyond frozen prior baseline.
Single training seed remains Probe. No positive signal: close1/1, do not add
rollouts/epochs/seeds/reward variants. Positive: independently predeclare
multi-training-seed Validation. Total smoke+science<=60min/8GiB,<=2idle GPUs;
parent admission falls back only among declared GPUs, never kills unknown jobs.

GPU engineering completed, code be58aee: on/off each96complete episodes,
51746effective nativeenvsteps,22optimizer updates; actual option probability
ratio error0, policy parameters changed, frozen Cm/experts unchanged. Raw
inputs checked unchanged and own native PIDs exited. First attempt r1 failed
before simulation at a public batch-initialization compatibility bug; preserve
r1 and charge its cost into r2 and scientific parent. No science outcome used
to modify method. r2smoke artifact includes full manifest/results. Per-decision
Cm computation is identical on/off; actual decision counts/total compute can
differ and will be reported. Science slot1/1 now launching fixed protocol.

Terminal: matched4rollouts/arm,each207123effective first-episode steps and
840updates; all rollout episode-length arrays matched. Actual batched simulator
steps counted separately in results. Initial weights identical; all inputs,
model/decision hashes unchanged and own native processes exited/GPU released.
Total including successful/failed engineering1849.32s/110.74MB.

Independent four-seed evaluation: stable-success on17/384 vs off13/384
(+1.042pp, eval-seed descriptive t95[−.872,+2.956]pp). Acquisition143vs137;
release-after-acquisition127/143vs126/137 is descriptive, conditions on a
post-treatment outcome and is not a common-prestate causal risk comparison.
On/off stable-ever58vs46,followed-by-drop41vs33. Only17vs13retain success
through episode end. No formal utility or stable-grasp claim.

Critical failed learning gate: on5065/off5014evaluation choices have ZERO
raw option-ID argmax changes from their frozen priors. Learned probabilities
changed, but greedy deployed physical actions did not. Trained-on6/96 vs
frozen-on3/96 atseed391 therefore cannot demonstrate learning; labels can
vary without changed choices under nonrepeatable simulator/observations.

GPU mechanical audit(.960s): fixed log prior advantage3.80666; maximum
learned competitor score advantage1.28299on/1.61185off, so no learned choice
can win against it. Removing prior only in offline arithmetic changes
4535/5065on and4871/5014offargmaxes; that is NOT evidence those alternative
actions improve outcomes. Runtime actual choices replay exactly; weights and
records unchanged. Root introduced the fixed prior, and it prevented this
learned scoring head from changing deployed decisions. Do not solve this
by adding epochs or interpreting17vs13asRL benefit.

Native reward audit: compute_humanoid_reward is rb*ro*rig*rcg tracking
reference body/object/interaction/contact and energy. It does not explicitly
optimize45tickhold/drop prevention. Next independent route should move Cm
knowledge into trainable actor parameters and align common reward with contact
supported retention, then first verify actual learned control. Existing gate
UNPROMISING and budget1/1remain closed; C3OPEN.
