# Actual task critic learning from physical pretrained encoders

Decision Probe; frozen before outputs. Source option-model-policy-r1 FIT603/604
ONLY1536episodes, existing current152 and sameFITSDKmean/std. No618, compatibility
predictions or anyEVALdata enters training, feature selection or normalization.

Reuse physical models cm/dynamics_off trained1500updates EACH at3551common
initialization; no pretraining repeat. Copy EXACT0.weight/bias and2.weight/bias
(firsttwo164->64ReLU->64ReLU layers) into task critics; discard131-output physical
decoders entirely. Same fresh4.weight/bias from initialized_network('value')
seed3553, sigmoid output. Record all initial critic parameters and verify source
copies/commonhead bitwise. Task input BOTHcritics=current152+tanh(actualraw12),
even for pretrained off encoder. All critic parameters are trainable. Original
physical off pretraining had BOTHaction inputs zero; that treatment is preserved
without depriving its actual task critic of executed actions.

Critics1500Adam3e-4/weight_decay1e-4/gradclip10steps EACH, shared privateCPU3554
batch256, measured terminal physical105binary crossentropy only. Same underlying
1536labels as original direct-Q, no fictitious reward, physical auxiliary loss,
future inputs or continuation V. Source physical/model/actor cost6000/3000steps
reported separately; reused directQ1000actor steps not repeated. DirectQ is a
strong additional comparison, not an assertion that its initial encoderseed3553
is identical to the common physical-pretraining origin3551. Main matched Cm/off
contrast has the same original pretraining initialization, schedules, data,
parameter counts and common new task head.

Freeze both finaltaskcritics, then152->64ReLU->64ReLU->12tanhactors EXACToriginal
common3555initialparameters.1000steps EACH, same privateCPU3556batch256,
Adam3e-4/gradclip10, objective -Q(current,tanh(actorraw))+.05mean(actorraw^2).
No predicted future used during actor learning. Every taskcritic and actor step
finite; first gradients and actual parameter changes recorded. Distinct
critic_step1500/actor_step0500/1000 save weights and optimizer/privategenerator;
no overwrite. Reused direct-Q model/actor remain bitwise source.

Fresh625/626EVAL768env/202ticks each; unchanged native current/causal features,
private balanced random placements/groups P0/Cm/off/directQ,384pooled perarm,
128motion/192seed. Actors only deployed at lift-8 PREdecision, existing raw12
independentcoordinates/couplings/projections/scales and held-untilstop+30contract.
Physical105rootrise30mm/fullmeshclearance20mm on ALL75plateau+30dropcheckingticks;
synthetic90-frameplateau explicit. No task or physicalmodel at deployment.

PROMISING iff Cm>=5pp EACHof P0/off/directQ, noninferior ALLthree on EACHseed,
and motion1 loss<=5ppvsP0. Allmotions included, no subgroup/checkpoint rescue.
Otherwise UNPROMISING; representation/objective/forecast losses cannot override.
One training fit/twoevaluation seeds, not formal Validation or novelty proof.

Independent fullFITrawcurrent/futureSDKlabels source audit retained; independently
check all current/sdk/normalization/labels again, initial transferred weights and
commontaskhead bitwise, final ALLFITtaskcritic/actor NumPy forwards<=2e-5,
reusedcontrol bitwise, finite actual updates. Original allcurrent/P0/PD/fullmesh/
privateRNG/nativeactor audits bothEVAL. No independent optimizer replay claim.
Tiny CPU parameter-transfer/physical-decoder-removal engineering smoke permitted;
allscientificmodel/actortraining GPU. Whole1200s/1GiB including freshidle GPU
admission, original675/sources/newcode protected, owned children only.
No layer freeze/head/width/LR/steps/penalty/seed/label-budget scans afterfailure.
