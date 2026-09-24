# P-20260924-duck-contact-action-opportunity

- Classification: Decision Probe.
- Source actor: self-trained duck specialist e340, checkpoint SHA256
  `9bcac13e814cc0de03deb9dcf9fdc8ee8bd9af4e6bb1c71e795c37f8a97e4a7a`.
- Cm: off during this randomized physical data collection.

## Question and decision

On the newly successful duck specialist, does a local wrist-z action
choice still change ten-step contact-supported object lift without losing
contact? This tests whether there is a learnable physical decision for Cm
in the successful grasp distribution, unlike the previously failed s3
local-option family.

On corrected duck, 64 environments, new seed210, at global steps50–190
every ten steps, randomize eligible currently contacting environments
between ±0.1 normalized wrist-z action for one control step, then resume
the frozen actor for ten steps. Require actual unclipped dose and complete
followup. Primary effect is plus-minus ten-step object-z displacement
weighted by followup contact fraction. Continue to a duck-specific
action-aware Cm only if this effect is >=10 mm and the contact-fraction
effect is nonnegative; otherwise stop this local z action family on duck.
One seed is a Probe, not proof of policy utility. The later Cm-on policy
test, if reached, must compare against both base and fixed-action controls.

One idle GPU, <=20 minutes, <100 MB. Stop on input/checkpoint drift, GPU
conflict, incomplete followup or nonfinite values. The intervention split
manifest is pinned at
`outputs/Dexplore/agent_duck_specialist_s70_e340/intervention_split_manifest.json`
with SHA256
`8c32852816a0f13c7be65d473366af6f1618e03fcc7ea5549f35cdd07a6fb27c`.

## Results

Pending.
