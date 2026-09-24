# P-20260924-waterbottle-specialist

- Classification: Decision Probe.
- Cm: off.
- Source: self-trained s3 airplane e260, same pinned source as duck.

## Question and decision

Duck-only continuation reached 48/64 on unseen seed206, while the
12-motion shared actor and train5 actor had zero waterbottle held-lifts.
An earlier waterbottle-only continuation from train5 e320 also collapsed.
Can the same e260→e340 single-object continuation transfer to waterbottle?
If fresh seed209 achieves >=16/64 held-lift and >=0.30 contact, retain
single-object specialization across distinct shapes as a promising baseline
route. If it fails, investigate source-dependent training and physical
contact patterns before scaling the portfolio.

Use the fixed corrected waterbottle tensor, 64 environments, seed70,
same reward and reset curriculum as duck, one idle GPU, <=60 minutes,
<5 GB outputs. Stop on checkpoint or input drift, GPU conflict, nonfinite
training or absent e340 checkpoint. Evaluate the first complete episode in
64 unseen-seed209 environments. A pass is exploratory, not a multi-seed
scientific conclusion.

## Results

Training `agent_waterbottle_specialist_s70_e340` completed. On unseen
seed209 it achieved **0/64 held-lifts**, mean contact fraction **0.016**,
and essentially zero maximum contact-supported lift. The matched frozen
e260 source also scored 0/64 but had contact fraction 0.167. The prespecified
gate failed: `UNPROMISING` for this fixed 80-epoch single-object protocol.
No more epochs of the same reset distribution are justified.

The final training batch had hand/object contact fraction about 0.42 and
positive held-lift reward on ~32% of sampled transitions, while full
start-frame evaluation almost never contacted the object. This points to
a reset-distribution gap, not proof of waterbottle's physical impossibility
(the read-only official diagnostic had 64/64). The next distinct Probe will
anneal near-contact/lift resets to ordinary starts.
