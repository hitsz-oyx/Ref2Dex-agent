# P-20260924-expert-multiaxis-choice

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe.

## Question

The expert-state wrist-z `+/-` intervention has a nearly universal positive
object-z effect. A constant action rule transfers better than learned Cm.
Is there a genuinely state/object-dependent *choice among actions* that
could give Cm a reason to matter?

## Minimal protocol and decision

Use the same pinned read-only official actor strictly as a simulator data
collector, never as the final policy. On the train object split only
(airplane, mug, toothpaste), randomize signed wrist x/y/z action doses
`+/-0.1` at pre-contact states, follow each for 10 actual simulation steps.
128 environments, steps 50..200 stride 10, seed 188. One idle GPU,
<=30 minutes and <=100 MB. Exact source/dose, object identity, reset and
followup must be audited. Apple remains outside this route choice.

Primary exploratory task outcome: 10-step object-z displacement multiplied
by 10-step hand-object contact fraction. Estimate each signed arm's
object/step-adjusted mean and rank signed arms separately by object. A
potential choice problem requires at least two training objects whose best
arms differ, with each object's best-vs-global-fixed-arm point margin >=5 mm
and nontrivial contact support. This is a Probe gate, not proof: if passed,
replicate with independent seed before training any Cm. If failed, stop
fixed small wrist-action ranking and reconsider a different task target,
action horizon or data distribution. No PPO run on this result alone.

## Result

Status: `UNPROMISING` for fixed small wrist-action ranking at a ten-step
closed-loop horizon. The 128-env source run
`agent_expert_multiaxis_train_s188_h10_n128` completed, pinning official
collector and split hashes and verifying all 2,021 selected executed doses.
Every object/arm had >=92 treated rows and all 16 intervention steps.
Ten-step contact stayed ~0.99-1.00, so loss of contact was not the
discriminator on this expert-state distribution.

The globally best signed arm in this exploratory sample was x+.
Airplane's best was also x+; mug's x- exceeded x+ by only +2.03 mm;
toothpaste's z- exceeded x+ by +9.95 mm. Only one, not two, objects met
the predeclared >=5 mm distinct-best margin. Do not replicate or fit Cm to
this six-arm ranking as if it were a robust choice problem. Because best
arms were selected and evaluated in the same sample, even these margins
are optimistic.

An offline horizon diagnostic on the same complete run showed the
contact-supported randomized z+ minus z- effect at one step to be
+19.58 mm for airplane (environment-cluster 95% CI [+16.30,+22.85]),
+30.11 mm for mug ([+27.27,+33.41]), and +30.03 mm for toothpaste
([+25.57,+34.18]). At ten steps it was -12.09 mm
([-27.36,+1.20]), +9.84 mm ([-12.62,+32.65]), and -8.80 mm
([-42.93,+23.63]), respectively. Thus an immediate positive physical
effect is reliable, but it is not a reliable proxy for the subsequent
actor-closed-loop lift outcome. The ten-step contrasts are noisy and do
not establish a sign reversal as a formal conclusion. The next route
should target policy-closed-loop grasp retention/progress and capture
pre-contact as well as contact states; further one-step wrist-z network
tuning would not address this horizon problem.
