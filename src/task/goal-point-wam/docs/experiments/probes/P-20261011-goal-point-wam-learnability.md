---
schema: ref2dex.probe.v2
probe_id: P-20261011-goal-point-wam-learnability
experiment_id: P-20261011-goal-point-wam-learnability
date: 2026-10-11
task: goal-point-wam
branch: main
git_commit: fa6df241e598930de39ed0cfdbe46be41fec78c0
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 4
seed_pool: probe
seeds: [247, 251]
decision_changed_if_positive: diagnose stochastic mixed-source optimization before scaling or preparing a matched world-loss-off comparison
decision_changed_if_negative: inspect representation and branch optimization rather than adding epochs or robot integration
status: PROMISING
run_id: fixed-train-learnability-20261011-r1
---

# Fixed-train-window WAM learnability diagnosis

Class: Decision. The previous 600-update aligned WAM Probe was UNCLEAR, with
less than 1.2% moving-world improvement over static and no trajectory benefit.
This serves the WAM representation subgoal; final matched self-trained RL
utility remains required and is not tested here.

Hypotheses: (1) the present mapping can fit fixed examples, but noisy/mixed
training or the short budget prevents learning; (2) the old optimizer scale
limits fitting; (3) the current inputs/targets/representation remain problematic.
The cheapest discriminator is a train-only fixed six-window fit, without a new
corpus, simulation, or held-out metric. These are fitting diagnoses, not
independent repeats or evidence of generalization.

## Protocol fixed before execution

- Parent: ref1-alignment best checkpoint, update400; retain its model, target,
  normalization and full528-token contracts. Fresh AdamW in each arm.
- Select two train windows per source, from the first qualifying entries of
  seed251's64-candidate moving-category order. Require moving h24 static EPE
  and average hand motion each >=5mm. No model error or held-out label selects
  windows. Save exact indices, raw sequence identity and the fixed collated batch.
- Independently reconstruct current/future hand, current/step delta and trajectory
  target from raw sequence poses/hand coordinates, with2e-5m tolerance. Check
  finite data and GT predictions' zero trajectory/point loss before fitting.
- Full six-window fixed batch, eval-mode layers with autograd enabled, BF16,
  fixed train-only normalization and serialization order. This deliberately
  removes dropout/order/sample noise and is a capacity diagnosis, not the
  historical training protocol.
- Original learning rate1e-4,300updates,20update linear warmup then constant
  rate, AdamW decay.01, gradient cap1, seed247. Read no val/test dataset.
- Only if original-rate fit fails: same initial weights/data/RNG/schedule,
 300updates at1e-3. Learning rate is the only difference between these arms.
- Fit gate: final/initial moving-world h24 EPE and average trajectory EPE
  both <=.5 on at least two sources. Report every source, including failures.
- If either arm passes, mark fixed-input learnability PROMISING and investigate
  stochastic/mixed-source optimization next. Failure remains UNCLEAR until a
  specific implementation or representation defect is established.
- One idle GPU2, <=15minutes including loading/audits, <=1GiB new artifacts;
  stop on nonfinite, OOM, input drift, resource conflict or deadline. Model-only
  final checkpoints, no redundant optimizer snapshots. Preserve prior outputs.

## Decision Note

The family now has two valid short mixed-source Probes without clear utility;
this fitting diagnosis could become the third. Root chooses the cheapest
train-only discrimination before investing in more epochs or new interfaces.
A fit pass changes the next investigation toward optimizer/noise/source
interference; a failure prompts targeted input/target/decoder checks. No core
hypothesis, claim, resource limit, branch or external authorization changes.

## Result

Both arms completed300updates on GPU2 in51.0/48.3seconds; initial parameter
hashes were identical. Initial EPE differences were <=4.77e-7m (CUDA numerical
variation, not bitwise-identical measurements). All six raw-coordinate
reconstructions were exactly equal (max error0), and all oracle loss terms
were0. Shared projection branch gradient norms were .001770/.004137,
cosine+.2217 at the parent state; this batch does not show initial opposition.

Final/initial EPE ratios (smaller is better):

| Source | lr1e-4 trajectory / moving world | lr1e-3 trajectory / moving world |
| --- | ---: | ---: |
| OakInk2 | .2075 / .5954 | .0865 / .0602 |
| GRAB | .0641 / .5943 | .0320 / .0265 |
| ARCTIC | .1583 / .1541 | .0667 / .0348 |

The old-rate arm passes the joint50% gate only on ARCTIC; the higher-rate arm
passes on all three sources. Final higher-rate trajectory EPE is2.58/4.00/3.94mm,
and moving-world h24 EPE is2.30/6.68/3.14mm. Fixed-input learnability is
**PROMISING**: the model can fit these examples, and optimizer scale changes
this diagnosis. This does not establish generalization, action dependence or
final policy utility. Sampled memory was below1.4GiB; artifacts total403MiB.
No val/test loader was constructed, and these weights are diagnosis-only.

Root chooses a separately recorded600-update stochastic run, changing only
learning rate1e-4 ->1e-3 and reusing the original PW initialization/RNG/draws
and96-window validation. No fixed-window fitted weights or test-selected
settings are imported.

## Limitations / future evidence

Fixed-batch memorization cannot demonstrate action conditioning, generalization,
world-supervision benefit to trajectory, or robot policy utility. The higher-rate
conditional arm is not selected using dev/test. Eval-mode fitting differs from
ordinary train-mode drop-path/randomized serialization; any subsequent recipe change must
be evaluated separately. Matched world-loss-off and RL Validation remain deferred.
CPU handles raw label reconstruction/hashes only; all model computations use GPU.

## Artifacts

- [Fixed window and raw label identities](../../../../../../outputs/goal-point-wam/fixed-train-learnability-20261011-r1/input_manifest.json)
- [Paired arm results](../../../../../../outputs/goal-point-wam/fixed-train-learnability-20261011-r1/result.json)
- [Saved fixed train batch](../../../../../../outputs/goal-point-wam/fixed-train-learnability-20261011-r1/fixed_train_batch.pt)
- [Original-rate parameters](../../../../../../outputs/goal-point-wam/fixed-train-learnability-20261011-r1/lr1e-4/final.pt)
- [Higher-rate parameters](../../../../../../outputs/goal-point-wam/fixed-train-learnability-20261011-r1/lr1e-3/final.pt)

## Follow-up train-mode control (fixed before launch)

The separately registered stochastic600-update lr1e-3 run did not pass its
validation gate. To isolate forward-mode noise from changing examples, run
`fixed-train-learnability-20261011-r2` at commit635f4567: same six-window
selection/batch, parent checkpoint, seed, AdamW, lr1e-3 and300updates as r1's
higher-rate arm; only forward mode changes to train (drop-path.3 and randomized
serialization). Measurements temporarily switch to eval and restore the fit
mode, with a runtime invariant. At most5minutes/300MiB on idle GPU2. Compare
batch and initial parameter hashes. A fit pass excludes forward-mode noise
alone as a total fitting blocker; it does not exclude its effect on larger data.
Failure motivates regularization/serialization diagnosis. No val/test data.

A separate frozen train-batch mechanism audit will translate valid endpoint
goals by+0.1m in X, keep scene/executed action unchanged, and report signed
h24 X response divided by0.1m, along with endpoint-target error and world
isolation. Linear interpolation responds with gain1 by construction. Absolute
learned gain below.1 is called weak response for this intervention, not a
physical or general goal-dependence conclusion. No additional fitting.


## Completed follow-up diagnoses

The train-mode control ran at actual manifest commit `da8f4296` (including the
`635f4567` mode-control implementation), completed 300 updates in 52.7 seconds,
and passed the joint fit gate on all three sources. Saved batch tensor and
initial parameter hashes match r1's higher-rate arm exactly.

| Source | Train-mode final/initial trajectory / moving world EPE |
| --- | ---: |
| OakInk2 | .0836 / .0807 |
| GRAB | .0267 / .0281 |
| ARCTIC | .0767 / .0199 |

This is PROMISING for fixed-input fitting. Forward-mode noise alone does not
prevent fitting these six examples; larger-data optimization remains unresolved.

The frozen +0.1m X goal intervention at evaluator commit `e6fbb091` gave:

| Frozen parameters | Signed endpoint X gain | Endpoint X response RMS (mm) | Endpoint EPE (mm) |
| --- | ---: | ---: | ---: |
| Original-rate best400 | 1.25e-7 | .000425 | 110.88 |
| Higher-rate best500 | 2.89e-6 | .000290 | 110.19 |
| Fixed-fit higher-rate300 | 6.16e-5 | .03559 | 4.33 |

All have weak response under the predeclared gain threshold. Low fixed-window
error therefore does not demonstrate goal dependence. These are six train
windows and one translation direction, not a general goal-invariance claim.

The first whole-forward world isolation check failed on the fixed-fit model.
Repeated unchanged inputs also differed: in the final diagnostic, repeat max
was 1.51e-5 versus goal-change max 1.60e-5, above the 1e-5 gate. The scene
encoder's repeat numerical variation can confound this comparison. The added
cached-identical-scene audit records world change and repeat exactly zero for
all three checkpoints. The original whole-forward failed flag and failed log
remain visible; this supports decoder isolation under identical scene memory,
without declaring the original whole-forward tolerance passed.

A frozen memory audit of both mixed-source checkpoints found scene/goal/action
mean token norms about 19.7--19.8 / 20.4--20.8 / 19.4--20.3. Goal attention
mass is .162--.171 across trajectory layers; world action mass is .654--.712.
These observations do not support the proposed much-larger-scene-norm starvation
mechanism on this batch. Attention mass does not prove geometric information use.
No normalization or architecture change was made from this audit.

Root's next bounded decision is goal-geometry/boundary-contract diagnosis before
scaling, robot integration, or matched world-loss-off. The user subsequently
requested a code checkpoint; no further experiment is launched here.

### Follow-up artifacts

- [Train-mode manifest](../../../../../../outputs/goal-point-wam/fixed-train-learnability-20261011-r2/input_manifest.json)
- [Train-mode fit result](../../../../../../outputs/goal-point-wam/fixed-train-learnability-20261011-r2/result.json)
- [Cached-scene and whole-forward goal responses](../../../../../../outputs/goal-point-wam/fixed-train-learnability-20261011-r2/goal_response_cached_scene.json)
- [Earlier unchanged-repeat diagnostic](../../../../../../outputs/goal-point-wam/fixed-train-learnability-20261011-r2/goal_response_repeat_diagnostic.json)
- [First failed whole-forward check log](../../../../../../outputs/goal-point-wam/fixed-train-learnability-20261011-r2/goal_response_first_attempt.log)
- [Memory norms and attention mass](../../../../../../outputs/goal-point-wam/fixed-train-learnability-20261011-r2/memory_attention.json)
- [Task-local frozen attention audit](../../../tools/audit/audit_memory_attention.py)
