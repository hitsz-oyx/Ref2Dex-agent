# Fixed actual continuous PPO comparison

Decision Probe: does an action-conditioned physical auxiliary critic improve
actual continuous feedback learning over matched state-only auxiliary and
no-auxiliary critics? A positive gate justifies method development/Validation;
a failure stops this exact recipe without coefficient/head/update/seed scans.
Cheapest discriminating test is one optimization seed762, equal short native
interaction budgets, final-only fresh evaluation. Standard auxiliary PPO itself
is established; no novelty or journal-readiness claim follows from this test.

Design authority: D-20261002-continuous-critic-cm.md. Engineering-only native
smoke fea9add on fresh567 completed70.580s,768trajectories,zero updates; independent
GPU-request/projection/likelihood/critic/next-physical-target/mesh/PD replay passes.
Do NOT train on567, do NOT initialize from any updated smoke or official actor.

Freeze base P0 and its FIT mean/std, own synthetic three references, native
original physics/XY±10mm/reset/202ticks and originalphysical105. Four private
balanced groups64/motion per768 panel, unchanged base/Cm/state-only/none. Private
assignment seedpanel+16000, placement+11000, same three Gaussian standardized
noise streams panel+17000; groups are independently randomized physical
environments, NOT paired counterfactuals. Effective independent coordinates
0..5,6,8,10,12,14,15. Request Gaussianstd.05initial, tanh scales.02m/.10rad/.15rad,
native executable projections/coupling. Save ACTUAL normalized70inputs and noise
along with full raw state/targets/native actions for independent replay.

All20training panels547--566,768each,192episodes per learned variant/panel,
3840episodes per variant. Actor and critic scratch seed762identical initially;
70->64ReLU->64ReLU->12 actor plus trainable logstd, zero actor final layer.
State-only value and dynamics encoder separate from actor; physical dynamics
head sees64latent plus12causal executed target-minus-currentq normalized by
nativePDscale(Cm) or zeros(state-only/none). True next objectXYZ delta/.005m and
linear-velocity delta/.05m/s targets. No future hand state or model reward input.

Actualterminalphysical105bool broadcast over202steps, gamma1. Frozen behavior V
subtracted; normalize all38784advantages per variant using population std+1e-8.
Four epochs, private CUDA permutations seed762+20000+update shared in shape
across variants, batch1024including final smaller batch:152steps/variant/panel.
Loss: clipped PPO(.2), entropy.001, valueMSE.5, dynamicsMSE.05Cm/state and0none.
One Adam over actor+critic parameters, lr3e-4, betas.9/.999, eps1e-8, foreachFalse;
jointglobalgradientclip.5. Same auxiliary forward/backward compute for all.
Carry only own optimizer/model states, no checkpoint selection or retries.

Mandatory independent CPU NumPy raw-state, normalization, native target/PD,
Gaussian request/logprob, complete actor/critic forward, one-step target and
full-meshphysical105 audit of EVERY22native panel. Before moving to next panel,
audit FIRST predetermined1024minibatch of EACHvariant: all actor/critic gradients,
joint norm and Adam parameter update. This is60audited first minibatches; other
9060optimizer steps are retained/schedule checked but NOT independently replayed.
Finite-gradient and actor-weight-change checks apply to every update.

Final-only mean requests, same projection, checkpointu20, fresh568/569:1536
trajectories,384per arm. Primary: pooledall-motion Cm gain>=5ppover BOTHlearned
controls, no worse than unchanged pooled, eachseed no worse than BOTHlearned
controls. Every gate required. Motions0/1/2 separately mandatory descriptive
tables, no subgroup rescue. PROMISING/UNPROMISING only; one optimization seed,
synthetic references, privileged native state, no hardware/generalization.

One freshly admitted GPU1UUID verified before every GPU phase. Whole<=3600s,
ownoutput<=6GiB, own process groups only, uniquely named run, no external writes,
forces, physics/material changes, checkpoint overwrite or push. CPU for tiny
initialization, independent NumPy replay/statistics only; actual training/inference
and supported simulation are GPU. Freeze source/input SHA and Git before launch.
