# GRAB progress evaluator: primary-source contracts for ref4

Read on 2026-10-08 for `../user/ref/ref4.md`. This note separates original methods, our adaptations, and one local data check. No training, checkpoint, or existing ref file was changed.

## SARM: the stage-duration prior must be defined precisely

The ref4 citation **2509.25358v1 is correct**: *SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation*. Section 3.1, equations 1–2 use the **mean of each demonstration's normalized stage durations**, not the ratio of aggregate durations:

```text
alpha[s] = mean_i(duration[i,s] / total_duration[i])
p[t] = cumulative_alpha_before_stage + alpha[stage] * within_stage_fraction
```

The stage estimator supplies predicted stage context to the progress estimator. Appendix A.4 uses stage CE and scalar progress MSE, with a substantially larger visual Transformer. Ref4's geometry encoder and 10-bin soft CE are adaptations. [Original SARM, §3.1 and Appendix A.4](https://arxiv.org/html/2509.25358v1)

LeRobot's `compute_temporal_proportions` implements the per-episode normalized mean and renormalizes its output. Its dictionary assignment overwrites repeated stage names; averaging only episodes containing a stage also changes the interpretation when stages are missing. Therefore complete, single-occurrence stages are the simplest contract for this first probe. [Pinned LeRobot annotation implementation, lines 92–147](https://github.com/huggingface/lerobot/blob/ca69a2068462a37f7cdcb74180927a2f863d2bf7/src/lerobot/data_processing/sarm_annotations/subtask_annotation.py#L92)

**Implementation inference:** ref4 fixes hold at `p=1`, so compute three active-stage proportions using `total_duration=b3-b0`, excluding subsequent holding time. Estimate them only on accepted training sequences, freeze them for validation/test, and reject zero-length/ambiguous stages rather than dividing by zero. Including hold in the denominator would prevent pre-hold progress from reaching 1, followed by an artificial jump at b3. The hold plateau and train-only prior are our experiment definitions, not claims that SARM prescribes these grasping labels.

## ReWiND: causal means every path to each output is causal

Appendix A.1.2 confirms a causally masked `nn.TransformerEncoder` and an MLP producing a reward for each frame. It embeds **only the first frame's position**. Full positional embeddings improve demonstration alignment but worsen policy-rollout ranking in its ablation, consistent with learning a time shortcut. Its rewind procedure appends an earlier sequence in reverse after a sampled split, and supervision follows the original frame's decreasing progress. The paper uses an 80% rewind probability, not ref4's 25%. [Original ReWiND, §3.1.1, §4.3 and Appendix A.1.2](https://arxiv.org/html/2505.10911v1)

**Implementation inference:** changing future frames or candidate A/E/I must leave the current output unchanged and leave earlier future outputs unchanged. This must cover preprocessing and normalization, not just an attention mask: a pooled 24-frame action summary must not enter the history/context tokens. A local reversed geometry sequence must reconstruct displacement features and relative transforms from its new frame order. Ref4's relative-window positions, 4-frame history, stage labels, geometry tokens, and 25% rewind are explicit adaptations; they should not be described as the original ReWiND recipe. Rewind supplies synthetic reversal evidence, not measured contact-force or drop-risk truth.

## PROGRESSOR: progress distribution is not success calibration

PROGRESSOR takes initial/current/goal image observations, predicts a Gaussian over relative temporal progress, and trains with KL against a frame-ratio target distribution. Its reward subtracts an entropy penalty from predicted mean progress. Online push-back reduces predictions on non-expert rollout observations while retaining expert supervision; uncertainty prediction alone does not solve distribution shift. [Original PROGRESSOR, §§4.1–4.2](https://arxiv.org/html/2411.17764)

**Implementation inference:** keep task context available at decision time, but do not feed the real future goal frame or completion time into C. Our 10-bin head is not PROGRESSOR's Gaussian head, and its softmax distribution is not automatically calibrated uncertainty. Demonstration progress accuracy alone cannot establish failure detection, stable holding, or control improvement.

## DexWM: the cited q-plan → FK interface is real

The relevant work is *World Models Can Leverage Human Videos for Dexterous Manipulation*, arXiv **2512.13644v1**, with the [official Meta/NYU repository](https://github.com/facebookresearch/dexwm). It is distinct from the similarly named CVPR video-diffusion work *Dexterous World Models*.

Section 3.4 / Appendix B optimize robot joint trajectories with CEM, convert joints to hand-point actions using FK, predict states, and score against an image goal. Low-level controllers take joint targets and interpolate small control steps. Simulation replans; its real-robot two-step experiment executes open loop. These facts support ref4's joint-target interface, without establishing our 24-frame horizon or evaluator-based cost. [Original DexWM, §3.4 and Appendix B](https://arxiv.org/html/2512.13644v1)

**Implementation inference:** commanded FK geometry and realized geometry differ under tracking error/contact. Preserve command/measurement identities and execution masks when later calibrating the robot interface; a successful offline GRAB evaluator does not establish executable robot plans.

## GRAB: contact codes and rigid-transform convention

The official repository specifies 120 Hz sequences, a separately posed table with sequence-dependent height, and object-vertex contact codes for body parts. Its lookup defines right palm **22**, right fingers **41–55**, left palm **21**, and left fingers **26–40**. Code **2 denotes left thigh**, not the table. [Official GRAB dataset description](https://github.com/otaheri/GRAB/tree/284cba757bd10364fd38eb883c33d4490c4d98f5), [pinned body-part lookup](https://github.com/otaheri/GRAB/blob/284cba757bd10364fd38eb883c33d4490c4d98f5/tools/utils.py#L152)

The renderer constructs **both object and table** with `ObjectModel`. Its vertex rule is row-vector `vertices = template @ Rodrigues(axis_angle) + translation`; the equivalent column-vector SE(3) rotation is therefore `Rodrigues(axis_angle).T`. Do not reuse the untransposed axis-angle matrix to transform table normals or compute object support clearance. [Pinned ObjectModel, lines 86–87](https://github.com/otaheri/GRAB/blob/284cba757bd10364fd38eb883c33d4490c4d98f5/tools/objectmodel.py#L86), [official renderer, lines 71–84](https://github.com/otaheri/GRAB/blob/284cba757bd10364fd38eb883c33d4490c4d98f5/examples/render_grab.py#L71)

**Local read-only check:** `data/raw_data/GRAB/grab/s5/cup_lift.npz` has `contact.body[2662,10475]` and `contact.object[2662,20002]`, both int8 with nonzero values drawn from hand-part codes 21/22/26–55; **neither is `{1=object,2=table}`**. `contact` keys are `body`, `object`, `threshold`, with threshold `2e-05`; there is no table-contact channel. Contact presence using `body[:,rhand_smplx_ids] > 0` agrees exactly with right-part codes on `contact.object` for all 2662 frames (1211 right-contact frames); the left check also agrees (1036 frames). This single-sequence check supports the encoding locally, not a dataset-wide semantic audit or a universal distance threshold.

Correspondence files each contain 778 unique hand-vertex IDs into the **SMPL-X body mesh**, not the 11-point network hand representation or arbitrary sampled object vertices. For this local check the right IDs span 7331–8128 and left IDs 4595–5394. Source SHA256s:

```text
grab/s5/cup_lift.npz:
  f7f8be4ee6453e28f879abaef53a8fa6540f8c096ce65309c8c7e1fa1d8df80e
tools/smplx_correspondence/rhand_smplx_ids.npy:
  1ccc70aa6b5e74989cb98a0b23a9f4ddc25882077efc56f546ea13e4b83572a9
tools/smplx_correspondence/lhand_smplx_ids.npy:
  54738b4b9ecafcaa69ef495a56d7c6a78121a258d23b9c832e3afc52ba72cada
```

**Implementation inference:** use right/left part-code membership on original object-vertex contact arrays, cross-check body-hand membership, and audit target versus other-body support explicitly. Do not treat `body == 1` as target contact or assume `body > 0` is exclusively right-hand contact. Table contact/support must come from posed table geometry. Recover original source frame IDs before accessing contact; loader-created compatibility frame IDs and contact-only filtered sample indices are not raw GRAB frame IDs.

## Decisive next checks

Before training: verify original-frame mapping, support-plane transforms, contact encoding across accepted sequences, label masks, and shared current/history input boundaries. A small geometry model is a valid ref4 adaptation without importing video encoders. The first decision is whether GT physical future adds held-out progress information under these labels; only afterwards replace all E/I jointly with frozen PointWorld predictions. These checks do not require forked preference pairs, expert policy retraining, or PointWorld retraining.
