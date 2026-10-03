# Data scale and MANO–Inspire prior transfer: bounded result

P-20261003-cm-scale-cross-hand-r1 COMPLETED/UNPROMISING atbc6b576. Same
surface-motion definition,19651parameters, common initialization; one idle
GPU6 for all1500step pretraining arms/600step adaptation arms and batch
predictions. Eight pretraining arms and six adaptations,15600actual optimizer
updates.458.194s/273459235bytes at closeout, inputs unchanged, all own PIDs absent.
Post-closeout figure is descriptive, not another model fit or gate change.

## Scale

| Training bank windows | MANO held-parent EPE/mm | Inspire held-parent EPE/mm |
| --- | ---: | ---: |
|512|3.869332|4.743223|
|2048|3.601316|4.416893|
|7168|3.559204|4.432058|

512to7168 gives8.015%/6.560% improvements, below both frozen10%gates. The
2048safeguards pass; most improvement appears before2048in this fixed update
budget. This does not show that data is irrelevant or that all larger models
and datasets would plateau. Equal1500updates means48000window draws perarm,
bank reuse93.75/23.4375/6.696times. Data size and optimization saturation must
not be conflated; this is a fixed-compute curve, not a scaling law.

Each train corpus spans50object_names:7946MANO windows/254parents,
7329Inspire windows/255parents. The7168prefix covers all50objects perhand.
Evaluation MANO2002windows/63parents/32objects; Inspire884/30/25. Frames
spaced to avoid overlapping previous/current/next triplets; observations
within parents remain correlated. Metrics give each parent equal weight.
This new model never selected an evaluation checkpoint, but existing Inspire
val data is not a new independent test corpus or unseen-object benchmark.

## Cross-hand and low-data adaptation

All adaptations use identical256Inspire training windows/154parents/47objects,
600updates/19200draws (mean75reuse), same minibatches and fresh optimizers.

| Initialization/input condition | Inspire held-parent EPE/mm |
| --- | ---: |
|MANO7168, frozen zero-shot|4.411876|
|MANO7168, adapted|4.726920|
|Common random initialization, adapted|4.830451|
|MANO shuffled-label pretraining, adapted|4.839696|
|MANO motion-off pretraining and adaptation|6.081358|
|Previous-object-flow persistence|5.429061|
|Zero object motion|14.999536|

Adapted informative prior improves2.143%over scratch and2.330%over shuffled
pretraining, failing required10%/5%. It improves22.272%over motion-off and
12.933%over persistence; these two subconditions pass but do not rescue the
complete gate. Motion-off removes information during adaptation too, so that
contrast alone cannot attribute gain to pretraining knowledge.

Zero-shot prior beats persistence by18.736%descriptively and is close to
Inspire7168training(4.432058). Thus broad motion prediction contains usable
offline information; no dramatic zero-shot collapse appears in this metric.
The fixed adaptation actually worsens the frozen MANO model by7.141%.
This is not evidence that all adaptation is harmful, that morphology has no
effect, or that previously observed policy failures have been explained.

All three complete prospective gates fail: UNPROMISING for this fixed scale/
adaptation design. Single initialization, no formal Validation or policy gain.

## Audit and interpretation limits

Independent audit rebuilds all18161selected source rows byte-exactly, selector
and nested packet order, labels/features, all held predictions and parent
aggregates, initial weights, all saved batch schedules/generator states,
optimizer step counters and gates. Closest-neighbor error1.926e-8m; rigid
point correspondence3.582e-7m; feature max3.815e-6; NumPy model max4.768e-6;
metric max3.210e-7mm. Future object perturbation leaves inputs unchanged.
Full optimizer-trajectory replay is not claimed. Source metadata/stat checks
and selected bytes are verified; whole multi-GB source file hashes are not
claimed. One read-only supervisor independently confirms metric/gate arithmetic.

Input is REALIZED next hand-surface movement. It is an offline motion-conditioned
diagnostic, not a deployable native action-conditioned predictor. Different
hand geometries,2048/10135sampling densities, trajectory distributions and
human/native-RL sources confound pure morphology attribution. Historical
Inspire physics predates our ownership fix; it is not corrected-physics task
evidence. Unsigned contact masks do not certify penetration-free physical
interactions or force closure. No existing actor/model weights initialized
this new predictor; source trajectory data is existing read-only experience.

[Figure PNG](../../src/task/CmResidual/research/contact_response/output/P-20261003-cm-scale-cross-hand-r1/figures/scale-cross-hand.png)
and [PDF](../../src/task/CmResidual/research/contact_response/output/P-20261003-cm-scale-cross-hand-r1/figures/scale-cross-hand.pdf)
show the complete scale curves and fixed adaptation controls. Perparent metrics,
all predictions, optimizer/generator checkpoints and selected raw packets remain
in that run; no confidence intervals or scaling-law fit are inferred.

## Decision

Keep the useful surface-motion prior artifact, close this fixed matrix without
additional step/seed/LR/width scans. Do not declare scale insufficient as the
main cause or abandon the knowledge-prior hypothesis. Next decision should
test whether a current-state/command hand-execution bridge can supply the
needed motion input on corrected native physics, distinguishing realization
mismatch from cross-hand prediction. This is a separate input-contract design;
more bulk collection before that check would not address the current boundary.
