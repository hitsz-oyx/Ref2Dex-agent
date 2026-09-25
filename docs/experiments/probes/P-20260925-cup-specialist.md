# P-20260925-cup-specialist

- Classification: Decision Probe.
- Branch: `agent/grab-cup-specialist`.
- Cm: off; actor initialized from self-trained airplane e260.

## Question and decision

The fixed 12-motion specialist route had **0/18** cup held-lifts on
seeds211–213, with only 0.066 mean hand-object contact. Yet the
same-start seed219 best-of-five diagnostic saw cup successes under
balanced e360 and duck e340. Can the existing single-object
continuation method learn a robust cup grasp as it did for duck?

Train on the already corrected `s1_cup_lift`, seed70, 64 environments,
source e260→e340, the same reward, optimizer and sampling settings as
the duck specialist. Evaluate 64 first full episodes on new seed226,
early termination disabled. If held-lift >=16/64 and contact fraction
>=0.25, repeat on new seeds227/228 and consider adding the frozen
expert to the object route. If not, stop this exact continuation;
inspect cup approach/grasp geometry instead of blindly increasing
epochs. A single seed is a Probe, not stable policy evidence.

One idle GPU, <=60 minutes, <5 GB output. Stop on checkpoint or
corrected-motion drift, occupied GPU, nonfinite training, missing e340
checkpoint or incomplete evaluation.

## Results

Pending.
