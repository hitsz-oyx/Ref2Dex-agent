# P-20260925-contact-expert-suffix

- Classification: Decision Probe.
- Cm: off; this tests whether a sustained expert option has physical
  grasp utility before fitting a Cm selector.
- Source/candidate: self-trained source e260 and balanced e360.

## Question and decision

Two randomized ten-step airplane switches increased 20-step contact
by about 6.5–6.9 pp, but only one increased supported lift and only
one improved full held-lift. Does keeping the candidate expert from
first actual contact through the rest of the first episode convert
the contact increase into grasp? Use the same three corrected airplane
motions across 64 environments, new simulator seed236 and a new
motion-stratified assignment seed20260925236. The control stays with
source; both arms have the same initial policy and trigger condition.
Record 20-step contact-supported z and full first-episode held-lift.

Pass if candidate minus source has >=5/64 more full held-lifts,
20-step supported-z >=0 mm, and contact fraction no more than 2 pp
worse, with >=40 valid follow-ups. If passed, consider an option-level
Cm that predicts long-horizon expert choice from contact state and
candidate actions. If failed, stop this source/balanced contact-switch
family and seek a different decision representation or training use
for Cm. One idle GPU, <=15 minutes, <100 MB output; stop on drift,
incomplete episodes, nonfinite actions or GPU conflict.

## Results

The seed236 run at commit `78a8ebc` completed with 64 first episodes,
60 first-contact triggers and 60 valid 20-step follow-ups. Candidate
and source had 29 and 31 valid follow-ups, respectively; mean initial
action gap was 0.17. Candidate minus source contact-supported object-z
was **+3.05 mm**, 20-step contact fraction was **+1.80 pp**, and full
held-lifts were **13/31** candidate versus **14/33** source. The
predeclared +5/64 full held-lift gate failed. The result is
`UNPROMISING` for source→balanced contact-stage suffix switching on
these airplane motions. Do not fit a Cm selector to this expert pair;
return to a representation or training objective that targets
contact-supported lift directly.

Run manifest, randomized assignment and first-episode results:
`outputs/CmResidual/agent_contact_expert_suffix_s236/`.
