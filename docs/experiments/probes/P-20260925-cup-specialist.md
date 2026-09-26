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

The seed70 e260→e340 continuation completed with the corrected cup
motion. The source e260 actor scored **2/64** on the same seed226 and
motion. The cup e340 actor scored **59/64**, with mean hand-object
contact fraction 0.895. The predeclared first-seed gate passed, so
new seeds227/228 were run without changing the checkpoint or protocol.
They scored **60/64** and **61/64**, totaling **180/192 (93.75%)** over
three evaluation seeds. Result: `PROMISING` for a self-trained cup
specialist on this one corrected trajectory. It is not multi-trajectory
or multi-training-seed Validation.

The checkpoint SHA256 is
`c7367b92248a01795abafe1761e2e96c86615f70fd406dbc9759c1dd3fc368fc`.
Training and evaluation manifests are in
`outputs/Dexplore/agent_cup_specialist_s70_e340/`. Next compare a
frozen route containing this specialist against the previous fixed
route on fresh 12-motion seeds; isolated cup success need not transfer
unchanged into the mixed simulator.
