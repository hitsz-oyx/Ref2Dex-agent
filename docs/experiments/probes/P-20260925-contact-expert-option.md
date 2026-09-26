# P-20260925-contact-expert-option

- Classification: Decision Probe.
- Cm: off; this identifies whether a contact-stage policy option has
  an action effect worth modeling with Cm.
- Source and candidate: self-trained source e260 and balanced e360.

## Question and decision

The old initial-action Cm selector did not improve the fixed route.
Does replacing the source actor with balanced e360 for ten steps at
the **first actual hand-object contact** on airplane produce a
meaningful change in subsequent contact-supported lift? If so, the
action has a short enough causal horizon to justify learning an
option-level Cm at contact and testing Cm-guided selection. If not,
do not train a Cm for this expert pair and option duration.

Use the three existing corrected airplane lift motions, repeated across
64 environments with seed234. Independently randomize half of each
motion's environments to the ten-step balanced option and the others
to ten steps of source; all continue with source thereafter. Trigger
once per environment at first actual contact. Record pre-option actor
observation, hand/object state, both candidate actions, executed
assignment, 20-step contact fraction and object displacement, plus
the full first-episode held-lift result. No tuning on seed234.

The signal gate requires at least 40 valid 20-step follow-ups, a mean
initial action gap >=0.1 in 18-D norm, candidate minus source mean
contact-supported vertical displacement >=5 mm, and candidate
contact fraction no more than 2 percentage points worse. Full held-lift
is reported but is too noisy to gate this small Probe. A passed gate
leads to a matched action-aware/blind/shuffled Cm fit and a fresh-seed
online Probe. A failed gate leads to another intervention family or
object, not to a claim against Cm generally.

One idle GPU, <=15 minutes, <100 MB output; stop on input/checkpoint
drift, incomplete first episodes, nonfinite actions or GPU conflict.

## Results

The seed234 run at commit `1d473f5` completed after an engineering-only
failed run at `65bfba0` (recursive action call, no intervention result).
With 64 first episodes, 62 environments triggered and completed a
valid first-episode 20-step follow-up. Candidate and source groups had
31 each; mean initial 18-D action gap was 0.17. Candidate
minus source mean contact-supported object-z displacement was
**+32.6 mm**, and 20-step contact fraction was **+5.0 pp**. Full
held-lifts were 15/31 assigned candidate versus 9/33 assigned source.
The predeclared signal gate passed (`PROMISING`).

The three airplane motions had respective supported-z candidate minus
source differences of approximately +68.7, +24.7 and +0.8 mm. The
largest motion also had unequal mean start frames (33.4 candidate,
19.5 source), so this single randomization may exaggerate the effect.
Before fitting Cm, repeat the identical intervention on fresh simulator
seed235 with a new assignment seed20260925235. Require >=40 valid
follow-ups, >=0.1 action gap, >=5 mm candidate-minus-source supported-z
and contact decline no worse than 2 pp. If this fresh repeat passes,
train a contact option Cm with action-blind and shuffled controls.

Artifacts: `outputs/CmResidual/agent_contact_expert_option_s234_r2/`.

The independent seed235 repeat at commit `e194036` completed with
61 valid first-episode follow-ups (30 candidate, 31 source).
Candidate minus source 20-step contact fraction remained **+6.11 pp**,
but contact-supported z was **−0.90 mm** and full held-lifts were
**11/31** candidate versus
**11/33** source. Its predeclared joint gate failed. Across the
three motions, the candidate-minus-source supported-z differences
were approximately −15.6, +11.1 and +1.0 mm. The first run's large
gain is not stable; the short ten-step option is `UNPROMISING` as a
lift-improving intervention. Do not fit a Cm to select this exact
option. The persistent contact increase suggests a longer expert
commitment may be needed to turn contact into lift; test that as a
separate decision Probe. Repeat artifacts are under
`outputs/CmResidual/agent_contact_expert_option_s235/`.

First-episode filtering correction: the initial runner could record
a trigger after an environment's first episode ended. Comparing each
trigger step with the first-episode length excluded one source row in
seed234 and one candidate plus one source row in seed235. The values
above use the corrected first-episode follow-ups. The predeclared
decision is unchanged. Future runs now prevent post-first-episode
triggers in the evaluator itself.
