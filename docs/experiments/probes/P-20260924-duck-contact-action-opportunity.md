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

The randomized seed210 collection completed on GPU6 with 789 eligible
contact interventions across 15 scheduled steps (398 plus, 391 minus).
Executed dose was verified as unclipped. Environment-cluster, step-stratified
500-resample analysis gave plus-minus ten-step contact-supported object-z
effect **+10.10 mm** (95% interval [7.60, 12.47]) and followup contact
fraction effect **−1.38 percentage points** (95% interval [−2.97, −0.16]).
One-step object-z effect was +31.34 mm, but ten-step unweighted effect fell
to +9.55 mm. The predeclared *joint* gate failed because contact decreased.
Status: `UNPROMISING` for the single-step wrist-z option as a safe duck Cm
candidate. Do not train a duck z-selector on this seed or retune the
contact threshold. This does show real physical action information, while
the lift/contact tradeoff still blocks this particular decision rule.

Artifacts: `outputs/CmResidual/agent_duck_contact_action_s210_h10_n64/`
(`run_manifest.json`, `transitions.pt`,
`followup_report_supported.json`). GPU5 was occupied at the first start
attempt, so no collection ran there; GPU6 was used without disturbing that
process.
