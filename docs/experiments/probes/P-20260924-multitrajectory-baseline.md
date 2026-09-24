# P-20260924-multitrajectory-baseline

- Classification: Decision Probe.
- Branch: `agent/grab-multitrajectory-baseline`.
- Work version: `multitrajectory-baseline-probe`.
- Cm: off.

## Question and decision

Can one self-trained actor continued from the s3 airplane e260 checkpoint
make useful held-lift progress on a fixed 12-motion, nine-object corrected
mixture while retaining airplane performance? If several object identities
show held-lift and airplane does not collapse, expand the converted GRAB pool
and design a matched Cm-on/off intervention against this baseline. If the actor
collapses or most objects remain at zero, first inspect motion sampling,
reference contact semantics and curriculum rather than attributing failure to
Cm. The cheapest informative test is a single-seed e260→e300 continuation,
followed by per-motion first-episode assessment.

This is a local converted subset, not the full GRAB dataset. The study does
not establish generalization, stable multi-seed performance or Cm utility.

## Fixed run contract

Use `src/task/CmResidual/configs/multitrajectory_12_motion_probe.json` and
`run_multitrajectory_baseline_probe.py`. Each listed motion appears once;
hard-object oversampling is off. Source checkpoint SHA256 is
`16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`.
Train seed70, 64 environments, horizon32, endpoint e300, one idle GPU,
wall time <=60 minutes, output <=5 GB. Stop on source/input drift, occupied
GPU, nonfinite training or missing endpoint checkpoint. Run status and
scientific outcome will be recorded separately below.

## Results

Pending.
