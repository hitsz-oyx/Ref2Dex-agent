# PointWAM architecture / training primary-source audit

Checked 2026-10-11 against the author paper and project, after reading `schema.json` and the existing Task research notes. This audit concerns the published method; it does not assess our six-window Probe as a formal failure.

| Contract | Paper evidence |
|---|---|
| Inputs | Current colored scene, current hands, language; **no object-goal point-set input**. [§3.1, Eq.2](https://arxiv.org/html/2610.02840v1#S3.SS1) |
| Features | Frozen Mosaic3D geometry/semantic scene features; hand-index/hand-ID MLP; frozen Recap-CLIP text. [§3.2](https://arxiv.org/html/2610.02840v1#S3.SS2) |
| Spatial fusion | Scene/hands share 1cm voxels; sparse-convolution stem; ordered 5³-voxel patches; global bidirectional attention with axial 3D RoPE. [B.3](https://arxiv.org/html/2610.02840v1#A2.SS3) |
| Forecaster | Volt-H: 32 layers, width1280, 20 heads; detokenizer plus local-stem residual; separate deterministic MLP trajectory heads, squared-error supervision; no diffusion/flow-matching sampler. [§3.2, Eq.3–5](https://arxiv.org/html/2610.02840v1#S3.SS2) |
| Targets | Observed-scene trajectories and semantic hand displacements, relative to current coordinates; ten points/hand: six palm, four fingertips. [§3.1/B.1](https://arxiv.org/html/2610.02840v1#A2.SS1) |
| Coupling | Shared language-conditioned scene/hand backbone; human joint pre-training from scratch; robot joint fine-tuning plus learned action retargeter. [§3.3–3.4, Eq.7](https://arxiv.org/html/2610.02840v1#S3.SS4) |
| Budget | 1,148,324 human episodes; 60k updates/stage, global batch192, AdamW LR1e−4; 4×B200, 34h pre-training/19h fine-tuning; 651M trainable/592M frozen pre-training parameters. [C.1, Table4](https://arxiv.org/html/2610.02840v1#A3.SS1) |
| Timing | Horizon30: human31frames/1s; DexJoCo31frames/0.6s. [C.1](https://arxiv.org/html/2610.02840v1#A3.SS1) |

## Release boundary and unknowns

The author project still displays **Code soon**; its code link targets its own citation section. No author-linked training repository/checkpoint was available on that page, so official source paths cannot be supplied. [Author project](https://chrockey.github.io/PointWAM/)

Same-scene/multiple-object-goal counterfactual pairings are **not specified**; the published input interface has language, not object-goal coordinates. This does not establish that such pairings never occur in the underlying data. The exact Mosaic3D checkpoint and any unpublished implementation differences remain unverified.

## Interpretation for this question

An object-goal conditional trajectory generator with a separate goal-blind action-conditioned world branch is a new contract. Matching “scene + hand point trajectories” alone would not establish equivalence to PointWAM. Neither a single-axis perturbation nor fitting six unique windows establishes equivalence to the paper’s robot-control experiments.

Earlier local reading, useful for provenance but rechecked here: [human-video usage](../../src/task/cm-pointflow-effect-pretrain/docs/research/2026-10-10-pointwam-human-video-usage.md), [reproduction feasibility](../../src/task/cm-pointflow-effect-pretrain/docs/research/2026-10-11-pointwam-reproduction-feasibility.md).


## Local implementation comparison and ranked explanations (root audit)

Checked against `ed7fc5f0` and the saved object-goal Probe. This section records
our code/artifacts and hypotheses, rather than facts established by the paper.

| Item | Local contract |
| --- | --- |
| Conditioning | Object start/future point sets; no language. Goal enters the small trajectory decoder after the goal-blind scene PTv3. |
| Decoding | 528 learned time/hand/keypoint queries; current hand XYZ does not explicitly enter queries. Scene point features are sampled down to256 tokens; goal XYZ passes through a raw-meter MLP plus endpoint-type embeddings. |
| World training | Executed future hand is input; rigid per-object effects are output. Joint losses share the scene encoder, but this is not the paper's language-conditioned co-forecasting contract. |
| Initialization | Own action-conditioned PW checkpoint supplies scene weights only. Goal/trajectory/world heads are new; this is not PointWAM scene+hand co-pretraining. |
| Latest sample/compute | Six distinct train windows, one observed goal/trajectory each,300updates; model about52M parameters. Earlier mixed-source probes600updates; their held-out result cannot be treated as equivalent to PointWAM evaluation. |

See [local model](../../src/task/goal-point-wam/src/goal_point_wam/model.py),
[object-goal Probe](../../src/task/goal-point-wam/docs/experiments/probes/P-20261011-goal-point-wam-object-goal-response.md),
and [reload/input/gradient audit](../../outputs/goal-point-wam/object-goal-response-20261011-r1/replay_goal_response.json).

1. **Scene memorization is the leading explanation of this particular fitting
   result.** Every fixed scene has one target, so a model depending only on the
   scene can minimize the same finite training set. Identical scene/multiple
   labeled goal-dependent trajectories are absent. Goal sensitivity after such
   fitting is a useful observation, not an identifiable causal training test.
   Prediction: a correctly labeled fixed-scene/multiple-goal controlled test
   would distinguish geometry-learning failure from scene memorization. An
   arbitrary shifted goal has no known physically correct hand label here.
2. **Weak geometric anchoring is a concrete architectural concern.** Original
   PointWAM decodes features at each current hand voxel; our query has identity
   and time embeddings, with position only indirectly accessible in sampled
   scene features. Direct hand rows retained by256-token sampling on these
   windows are OakInk2 3/22 and2/22, GRAB11/22 twice, ARCTIC6/22 twice. This
   does NOT prove all other hand information is lost, since PTv3 propagates
   neighborhood features. Prediction: preserving all semantic hand features
   and anchoring queries to their current coordinates should improve a
   controlled conditional geometric test if this bottleneck matters.
3. **Coordinate conditioning can be weak without a disconnected path.** Strict
   reload verifies real object inputs, finite nonzero endpoint coordinate
   gradients, and changing goal tokens. However, +10cm changes goal token
   coordinates by RMS .01161 while mean full-token L2 norm is24.24 (different
   summaries; do not directly divide them as a norm ratio). The decoder's
   endpoint response is only about.034mm; coordinate scale/type embeddings and
   fusion deserve comparison, but these norms alone do not establish a cause.
4. **Pretraining and budget differ substantially.** Fitting six examples rules
   out gross inability to memorize them; it does not establish sufficient
   transferable capacity, human/robot alignment, or conditional generalization.
   Our PW transfer does not supply a jointly pretrained action forecaster.

[Saved scene-token retention audit](../../outputs/goal-point-wam/object-goal-response-20261011-r1/scene_token_retention.json)
contains the deterministic CPU input-contract counts; no new fitting was run.

Priority: resolve the spatial hand/condition contract and establish correctly
labeled goal variation before choosing larger-scale training. The existing
weak-response screen alone is insufficient to abandon the object-goal idea,
prove a unique optimization cause, or claim PointWAM cannot be reproduced.
