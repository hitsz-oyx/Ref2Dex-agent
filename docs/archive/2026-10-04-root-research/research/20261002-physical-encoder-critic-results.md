# Physical encoder into measured-return critic: actual policy result

Decision Probe P-20261002-physical-encoder-critic, frozen designa085032,
implementation56ed3e0, run-r1 COMPLETED / UNPROMISING. Reuse physical Cm/off
encoders trained1500steps EACH on FIT603/604ONLY. Discard physical decoders;
common fresh task head, all critic parameters fine-tuned1500steps EACH on actual
terminal physical105labels. Both task critics get actual options. Two identically
initialized actors1000steps EACH use those frozen task-Qcritics. Existing strong
direct-Q actor is reused bitwise, never repeats its1000valid optimizer steps.

Reused1536FITepisodes; new3000taskcritic+2000actor steps, new1536EVAL625/626
native episodes/202ticks each. Source6000model/3000actor steps and pretraining
interaction cost separately reported. No imagined successor/value composition,
physical auxiliary loss or synthetic reward during new task fitting/actor updates.
The matched Cm/off treatment originates in common3551physical initialization
and action-conditioning versus removal. Their task head and actor init are
common. Strong additional directQ initially used3553encoder, not identical to
physical3551origin; no full-init match across that extra control is claimed.

| Method | Physical105 /384 | Rate | EVAL625 /192 | EVAL626 /192 |
| --- | ---: | ---: | ---: | ---: |
| P0 |143|37.24%|69|74|
| Cm physical encoder -> measured-return Q -> actor |219|57.03%|106|113|
| Physical-action-removed encoder -> task-Q -> actor |218|56.77%|118|100|
| Reused strong direct-Q actor |225|58.59%|110|115|

| Motion | P0 /128 | Cm /128 | Physical-off /128 | Direct-Q /128 |
| --- | ---: | ---: | ---: | ---: |
|0|0|0|0|0|
|1|106|113|108|113|
|2|37|106|110|112|

Frozen >=5ppoverEACHcontrol: FAIL. Each-seed noninferiority toALLcontrols: FAIL.
Motion1 loss<=5ppvsP0: PASS. Primary **UNPROMISING**. Cm exceeds P0descriptively
by19.79pp, exceeds physical-off byonly.26pp and trails strongQ by1.56pp.
One FIT/twoEVALseeds, no formal Validation. This does not show Cm-specific
policy utility. Cross-experiment changes are not mechanism attribution.

Independent all1536FIT raw current/future SDK error0; initial encoder copies and
common new taskhead verified bitwise; physical decoder removed. Both actual
critic inputs independently checked with tanh numerical error<=8.67e-8.
AllFITtaskcritic forwards<=6.48e-7, actors<=4.51e-7. Nonzero first critic/actor
gradients and actual parameter changes retained; all parameters trainable during
task fitting, frozen only before actor optimization. Critic_step1500 and actual
actor_step0500/1000 checkpoints include optimizer/privategenerator states.
No independent optimizer replay claimed.

All native/P0/PD/privateassignment/fullmesh105/actual actor audits PASS:
deployed observations reconstructed exactly, allactor forwards<=3.52e-7,
fullmesh clearance<=3.56e-7m. Policies alone deployed at PRElift-8 and options
held untilstop+30; synthetic90plateau, privileged current context explicit.
Physical105unchanged:30mmrootrise and20mmfullmeshclearance on ALL75plateau
plus30dropcheckingticks. No same-state replay, sourceactor inference, model
lookup or future observation at deployment.387.568s/504286350bytes<=1200s/1GiB;
all protected hashes unchanged, all owned parent/child PIDs absent.
Result SHA25650cefe3b7cfdd1fa4bbc39553d861d6ea590e6291115c64477e3f15248ee7430.
Actor SHA25655842fb3731bbc94297b58e8967d561310c2298cddeffd3f334db733d41fb8f2.

Stop this exact transfer recipe without layer-freeze/LR/head/steps/penalty/seed/
label-subset scans. Reconsider the broader fixed late held-option experimental
setting before another related learner. Cheapest next Decision is a FIT-only
support audit for the unsolved motion: are there successful held-option outcomes,
only transient lifts, or neither? It selects reward-learning, retention or earlier
contact acquisition data, not a formal claim that any controller cannot succeed.
Mission unchanged, goal ACTIVE; Cm utility/new methodology/journal readiness unmet.
