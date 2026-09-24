# P-20260924-train5-balanced-coverage

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe — sampling imbalance versus policy limitation.

## Question and fixed decision

The train5 e320 actor supplies adequate H10 load-bearing support on only 3/5
identities. DExplore's default hard-object substring rule duplicates
`cubesmall`, giving it 21/64 persistent environments while the other objects
receive only 10–11. Is this accidental imbalance a material cause of the
missing toothpaste/waterbottle coverage?

Add an opt-out that preserves upstream hard-object oversampling by default,
then use an otherwise identical config with oversampling disabled. Resume the
same self-trained train5 e320 checkpoint for 40 epochs to e360, seed179,
without Cm or held-out data. Evaluate train5 on the same seed178 and export
first-episode transitions. Apply the unchanged H10 coverage definitions and
4/5 object gate from `P-20260924-train5-state-coverage`.

If >=4/5 objects pass and no previously passing identity loses its support,
retain e360 as the substrate for a dense Cm representation Probe. Otherwise
mark simple object balancing/short continuation `UNPROMISING`; do not add
epochs or tune object weights on these results, and move to a policy
representation/curriculum redesign. Held-out alarmclock is not evaluated
unless the train coverage gate passes.

One idle GPU, <=20 minutes training plus <=10 minutes evaluation, <5 GB new
artifacts. Stop on source checkpoint/config/split drift, GPU conflict,
non-finite training, missing equal object assignment, or budget breach. This
is a Cm-off coverage Probe, not evidence of generalization or Cm utility.

## Result

Status: `UNPROMISING` for simple object balancing and a 40-epoch
continuation. The e360 run and balanced evaluation completed. Environment
assignment became 12–13 episodes per identity as intended, overall held-lift
rose from the coverage replay's 17/64 to 22/64, and contact fraction rose;
however support merely moved between objects rather than becoming broad.

Airplane, mug and toothpaste passed the H10 support gate. Toothpaste improved
to 11/13 held-lifts, but cubesmall fell from a passing transient-support state
to zero positive H10 windows and 0/13 held-lifts; waterbottle also had zero
positive windows and 0/13 held-lifts. The result remains 3/5 and loses a
previously passing identity, failing both fixed conditions. Do not tune
sampling weights or add epochs to this run.

This redistribution indicates shared-policy interference or object-dependent
learnability, not just the accidental cubesmall duplicate. The cheapest next
diagnosis is a bounded waterbottle-only continuation from the pre-balance
e320 checkpoint. Success would identify multi-object interference/state
representation as the blocker; failure would point to reward/curriculum or
initialization for this object before any larger Cm is justified.

Artifacts: `outputs/Dexplore/agent_crossobject_train5_balanced_s179_e360/`
and `outputs/CmResidual/agent_train5_balanced_coverage_s178/`.
