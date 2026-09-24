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

Pending.
