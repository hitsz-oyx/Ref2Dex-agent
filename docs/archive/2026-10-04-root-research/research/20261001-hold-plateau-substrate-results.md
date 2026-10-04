# Frozen synthetic holding substrate: no direct transfer

Physical collection code `40e73f2`: four192-env panels, actors286/287, fresh
init498/499. All768 first native episodes completed. Actual loaded reference
lengths633/743/578, stationary plateau poses, seven exactly zero GPU velocity
tensors, per-step progress increments and90 plateau samples were verified.
Actor and both evaluation-normalizer fingerprints did not change.

The original parent run is FAILED only because the pure reporting expression
passed duplicate keyword fields after independently checking the first trajectory.
Its manifest/log/source remain intact. Separate correction code `3f4ae77` fixes
that expression, checks every protected input, and uses IDENTICAL phase scoring
and gates. It recomputes all768 episodes from saved traces, with no new physics,
at `P-20261001-hold-plateau-substrate-r1/analysis_correction_r1`.

| Motion | 75-step retained | 45-step stable | Mean plateau rise |
| --- | ---: | ---: | ---: |
| s3 | 4/256 (1.5625%) | 6/256 (2.3438%) | -10.025mm |
| s7 | 0/256 (0%) | 0/256 (0%) | 5.149mm |
| s9 | 1/256 (0.3906%) | 13/256 (5.0781%) | 14.190mm |
| All | 5/768 (0.6510%) | 19/768 (2.4740%) | 3.105mm |

All four fixed gates fail: pooled>=10% and each motion>=5%. Scientific label
UNPROMISING for this frozen task/actor pairing. Initial frame0 surface gaps are
all above20mm, as expected for an approach-from-start task. Force proxies are
not attributed hand-object contact and can include table forces; this unchanged
limitation does not rescue a failed height/hold gate.

The task labels request sustained holding, but the existing self-trained actors
do not transfer sufficiently. End this direct-transfer pairing. A new single-seed
task-specific baseline training Probe will test a predetermined holding reset
curriculum and dense phase reward. It is not a Cm learner, novel method, matched
causal superiority comparison, or journal-level validation.
