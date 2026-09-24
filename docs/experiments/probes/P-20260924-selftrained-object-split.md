# P-20260924-selftrained-object-split

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe, not Validation.

## Question and competing explanations

The frozen self-trained airplane actor achieved only 3/64 held-lifts on
physically feasible held-out apple (official diagnostic 61/64). Is a short,
strictly object-disjoint multi-object PPO adaptation enough to make the
self-trained policy a useful foundation for cross-object Cm experiments?

H1: exposure to three non-apple object identities yields a material held-out
apple improvement without using official actor weights or apple training data.
Alternative: this actor/training recipe does not generalize in a short Probe;
the next step should inspect object-centric representation, curriculum or
policy training before interpreting a Cm-on/off result.

## Fixed minimal protocol

Use `cross_object_probe_split_v1.json`, materialized by the object-disjoint
split builder. Train objects: `airplane`, `mug`, `toothpaste`; held-out object:
`apple`. All sequences are corrected DExplore tensors with source manifests.
No apple sequence, Cm transition, supervised label or evaluator result is
admitted to training. Keep official actor checkpoint *only* as a diagnostic.

First evaluate the read-only official actor on the three training motions,
64 first complete episodes. Require each object to show at least 20% held-lift
before treating it as a feasible training task. Then evaluate the frozen
self-trained source actor on the train split to establish a per-object start
point. If the input gate passes, warm-start only from the self-trained s3
airplane e260 checkpoint and continue the same Cm-off PPO recipe to e320
(60 additional epochs, one training seed). Evaluate e320 separately on the
three training objects and held-out apple, with early termination disabled.

Continue to a Cm cross-object policy Probe only if apple reaches at least
15/64 and is at least +10/64 over the frozen source actor, while at least
two training objects improve or remain viable. If apple stays below that,
the immediate next decision is policy/task representation, *not* claiming Cm
failed. The thresholds are triage gates, not formal efficacy tests.

One idle GPU per run, <=60 min PPO wall time, <=20 GB local output. Stop on
input/split hash drift, official feasibility failure, non-finite training,
GPU conflict, lost provenance or budget overrun.

## Result

The official physical feasibility gate passed on 64 episodes: mug 22/22,
toothpaste 21/21, airplane 19/21. The official checkpoint remains diagnostic
only. Frozen self-trained source e260 on the same three-motion root gave
mug 12/22, toothpaste 2/21, airplane 15/21, overall 29/64. On the held-out
apple it previously gave 3/64. Thus the training split has learnable but
heterogeneous starting performance; multi-object PPO is the decision run.
A single seed and one held-out object can only be `PROMISING`,
`UNPROMISING` or `UNCLEAR`, never proof of cross-object policy or Cm utility.
