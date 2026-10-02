# Transfer physical representation into a critic fitted to actual outcomes

Question: can action-conditioned physical pretraining help policy learning
when measured returns correct the critic, without composing imagined futures?

Evidence: deterministic-successor and observed-support successor actors fail
against action-removed models and strong direct-Q. The atomic version gives
Cm188/off188/directQ222 per384 with all audits passing. Actual successor oracle
has useful predictive headroom, but available model value composition has not
accessed it. State-value auxiliary PPO is also negative; it did not pretrain an
action-conditioned Q encoder and then fit its whole representation to returns.

Choose one bounded actual-learning role: reuse already valid1500-step mean
physical encoders from FIT603/604, remove their physical decoders, install a
common fresh task-value head and fine-tune the ENTIREcritic on measured terminal
physical105 labels. Compare Cm-pretrained versus physical-action-removed
pretrained encoders under identical task updates/actor initialization. The
original strong direct-Q actor is reused unchanged as additional control.
No physical optimizer repeats, new FITdata, generated successors, support
lookup or synthetic returns. Physical pretraining/common model origin is the
causal treatment; the task critic gets the actual option in BOTHarms.

This is generic representation pretraining, not novel by renaming. Primary
source Fan et al. RA-L2026 already warm-starts actor/critic with dynamics
representations and keeps them trainable. Our forward-Q-only use is different
engineering, not a distinct-method claim. No PPO or online efficiency claim.

Cost<=1200s/1GiB, one idle GPU,3000newtask-critic+2000newactor optimizer steps,
1536reusedFIT and1536freshEVAL625/626. Complete existing5ppALLcontrols,
each-seed noninferiority and motion1 safeguard choose the outcome. Positive
requires matched multi-training-seed Validation and novelty work. Negative
closes this weight-transfer recipe without layer freeze, learning rate, penalty,
steps, label subset or seed scans, and triggers review of the wider fixed late
held-option experimental setting before another related learner.
Mission/claim/resources unchanged; no external authorization needed.
