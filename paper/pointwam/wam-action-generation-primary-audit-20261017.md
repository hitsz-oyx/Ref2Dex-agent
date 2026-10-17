# How current World-Action Models turn state into robot actions

Checked 2026-10-17 against primary arXiv papers and author project pages. “State decoding” is used here in the broad sense of reconstructing/predicting a future observation or latent state; it is not the same as decoding the robot action. The papers use several distinct inference contracts.

## Short answer

A WAM does **not** have to decode a full future state or video at deployment in order to output actions. Current work uses at least four patterns:

1. **Joint video/state + action generation:** future video/state latents and an action chunk are denoised in one coupled model. The action is an explicit output head/stream; pixel decoding can be skipped. DreamZero and $\tau_0$-WM are examples.
2. **Future-state then inverse/action dynamics:** first predict a future visual/latent trajectory, then use an inverse-dynamics or action head conditioned on that predicted future to produce actions. This is the classical “imagine then execute” interface.
3. **World supervision during training, direct action at test time:** a world-model head shapes the shared representation during training, but is removed at deployment; an action head maps the current observation representation directly to an action chunk. Fast-WAM and EgoWAM explicitly test this design.
4. **Intermediate motion representation + retargeting:** the model predicts a task-relevant trajectory (for example, hand keypoints), and a retargeter converts that trajectory plus robot state into executable actions. PointWAM uses this pattern; it does not feed its scene trajectory into the action retargeter.

Thus, “direct action decoding without decoding state” is a valid and now explicitly demonstrated WAM design. The important distinction is whether the action head receives (a) current latent features, (b) generated future latents, or (c) an intermediate predicted motion trajectory. “No pixel video output” does not mean “no world modeling”: the world may be represented by hidden features, latent future tokens, or a separately trained auxiliary head.

## PointWAM: predicts a motion state, then retargets it

PointWAM defines the policy as

\[
(\mathcal U_t,\mathcal V_t)=f(\mathcal P_t,\mathcal Q_t,\ell),\qquad
\mathcal A_t=g(\mathcal V_t, s_t),
\]

where $\mathcal U_t$ are future scene-point displacements, $\mathcal V_t$ are future hand-keypoint displacements, and $s_t$ is current robot state. The paper says scene trajectories supervise world modeling, while hand trajectories are the intermediate action representation. Its action retargeter encodes the forecast hand trajectories and robot state, then a Transformer decoder predicts an action chunk (end-effector poses and finger-joint targets). The scene forecast supervises the shared backbone but is not an input to the retargeter.

This is therefore **not** a direct $o_t\rightarrow a_{t:t+H}$ policy in the narrow sense: it has a learned, structured motion intermediate. It also does **not** decode a complete future scene to obtain actions. The action path is `current scene/hand/instruction → future hand trajectory → robot action`.

Sources: [PointWAM abstract and formulation, §3.1](https://arxiv.org/html/2610.02840v1#S3.SS1), [trajectory forecaster and retargeter, §3.2–3.3](https://arxiv.org/html/2610.02840v1#S3.SS3), [retargeter details, Appendix B.4](https://arxiv.org/html/2610.02840v1#A2.SS4). The paper explicitly contrasts PointWAM with PointWorld, which predicts no actions and requires a separate planner, and with 3PoinTr, which predicts tracks followed by a separate action policy ([§2](https://arxiv.org/html/2610.02840v1#S2)).

## DreamZero: coupled future video and action denoising

DreamZero formulates a joint distribution over future video and actions conditioned on observation history, language, and proprioception. The paper writes the conceptual factorization as video prediction followed by an inverse-dynamics model, but trains a single end-to-end model instead of two separate models. At inference it **jointly denoises video and action chunks**. Ground-truth observations replace generated frames in the closed-loop KV cache after each executed chunk, so pixel/video generation is not a separate post-processing requirement for every control decision.

This is an action-generating WAM whose action stream is explicit. It does model/denoise a future visual latent stream, but the robot receives the action chunk; the future visual stream is part of the coupled generative process and can be maintained in latent form. The authors also report that trying to generate only actions does not automatically make inference cheap, because the shared DiT and diffusion steps still dominate runtime.

Sources: [DreamZero formulation and single-model design, §3.1](https://arxiv.org/html/2602.15922v1#S3.SS1), [joint action/video inference](https://arxiv.org/html/2602.15922v1#S3.SS1), [closed-loop replacement with ground-truth observations](https://arxiv.org/html/2602.15922v1#S3.SS1), [runtime discussion](https://arxiv.org/html/2602.15922v1#S3.SS2).

## $\tau_0$-WM: action-only deployment is an explicit mode

$\tau_0$-WM has a policy-facing Video Action Model (VAM) and a separate Action-Conditioned Video Simulator (ACVS). The VAM jointly predicts future visual latents and executable continuous action chunks. At inference, it can run in two modes:

- **Action-only deployment:** generate and execute the action chunk in receding-horizon control; future latents remain latent and need not be decoded into frames.
- **Rollout-enabled deployment:** additionally decode future latents to video when explicit visual rollout is desired.

When candidates are unreliable, ACVS takes a candidate action chunk as a clean condition and predicts future latents/reward to rank candidates; ACVS itself is not an action policy. This separates “propose actions” from “evaluate consequences” while still allowing action generation without pixel reconstruction.

Sources: [VAM jointly predicts future latents and action chunks](https://arxiv.org/html/2606.01027v1#S4.SS1), [action-only vs rollout-enabled inference](https://arxiv.org/html/2606.01027v1#S4.SS4), [ACVS is a simulator, not an action policy](https://arxiv.org/html/2606.01027v1#S5.SS1).

## Fast-WAM: direct action from a world-grounded latent

Fast-WAM makes the distinction particularly clean. During training it jointly optimizes an action flow-matching loss and a future-video latent loss. At inference it removes the future-video branch, encodes the current observation once with the video DiT, and uses the resulting latent world representation to denoise an action chunk directly. The paper writes this as

\[
p_\theta(a_{1:H}\mid o,l)=p_\theta(a_{1:H}\mid z(o,l)).
\]

Its action tokens cannot attend to future video tokens during training, avoiding leakage. Fast-WAM reports performance close to its joint and video-then-action variants, while removing video co-training causes a much larger drop. This is direct action decoding at test time, with world modeling retained as a training-time representation objective.

Sources: [three WAM inference paradigms](https://arxiv.org/html/2603.16666v2#S1), [direct-policy formulation](https://arxiv.org/html/2603.16666v2#S3.SS1), [single-pass inference architecture](https://arxiv.org/html/2603.16666v2#S3.SS2), [controlled results](https://arxiv.org/html/2603.16666v2#S4.SS3).

## EgoWAM: auxiliary future-state head, action-only deployment

EgoWAM uses a shared trunk with two parallel heads: an action head that produces a continuous action chunk and a world-model head that predicts a future state target (pixel latent, DINO feature, or 3D flow). Both heads are trained jointly on robot and human data. At deployment, the world-model head is frozen/detached and dropped; only the shared trunk and flow-matching action head are unrolled. The authors explicitly state that this gives action-only inference at behavior-cloning latency while retaining the representation learned from future-state supervision.

This is the clearest example of “decode actions directly, do not decode the future state” being intentional rather than an implementation shortcut. The predicted future target is needed for the training signal and cross-embodiment transfer, not for the runtime control interface.

Sources: [shared trunk and parallel action/world heads](https://arxiv.org/html/2607.08436v1#S3.SS2), [world target variants](https://arxiv.org/html/2607.08436v1#S4.SS2), [action-only inference and dropped world head](https://arxiv.org/html/2607.08436v1#S4.SS3), [deployment details](https://arxiv.org/html/2607.08436v1#A2.SS4).

## PointWorld/UWM as useful contrasts

PointWorld is a **world model rather than a complete action policy**: it predicts full-scene 3D point flows conditioned on a supplied robot action sequence. A separate planner/MPC selects candidate action trajectories. This is the opposite interface from direct WAM policy decoding: state prediction is primary, and action selection is external.

Unified World Models (UWM) integrate action and video diffusion in one Transformer and can be switched between policy, forward dynamics, inverse dynamics, and video-generation modes by changing modality-specific diffusion timesteps. In policy mode, the action diffusion stream is decoded as the action; in forward-dynamics mode, future observations are generated conditioned on actions. Therefore, one model can expose both direct action and state-prediction interfaces, but a given control call need not decode both.

Sources: [PointWorld formulation and MPC use](https://arxiv.org/html/2601.03782v1), [UWM abstract and mode switching](https://arxiv.org/html/2504.02792v3#S1).

## Implications for interpreting a WAM-to-PPO experiment

A frozen WAM feature vector concatenated into a PPO observation is only one possible use of a WAM, and it is not equivalent to the action interfaces above. In the direct-action designs, the action head is trained jointly with the world objective and is usually conditioned on temporally aligned visual/proprioceptive context. In PointWAM, the action head consumes a forecast hand trajectory plus current robot state. In imagine-then-execute systems, the action head consumes a generated future representation.

Consequently, a negative result from “frozen WAM embedding + PPO” does not test the general claim that WAMs help action generation. It tests whether a particular representation, with its training-time semantics and temporal context removed or altered, is useful as an extra PPO feature. A faithful comparison should identify which contract is being reproduced: direct action head, future-state-conditioned inverse dynamics, intermediate motion retargeting, or auxiliary world supervision with a separately trained policy.
