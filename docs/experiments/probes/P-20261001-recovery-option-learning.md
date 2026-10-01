# P-20261001-recovery-option-learning

HF11, scientific budget1/1 running after terminal engineering pass.
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
