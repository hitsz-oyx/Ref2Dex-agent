# CHORD wrench review (2026-10-11)

## Scope and identity

“CHORD” is unambiguous in this task context: NVIDIA’s **Contact Wrench Guidance from Human Demonstration in Robotic Dexterous Manipulation**, arXiv:2607.00033. It is a contact-wrench-space reward for RL policy training, rather than a point-flow network or a force/torque prediction benchmark.

Primary sources reviewed:

- Paper: <https://arxiv.org/abs/2607.00033>; HTML equations: <https://arxiv.org/html/2607.00033>, especially §3.1 (HTML lines 120–141) and Appendix B (352–405).
- NVIDIA project page and released repository: <https://github.com/nvidia-isaac/video_to_data/tree/main/docs/chord> and <https://github.com/nvidia-isaac/video_to_data/tree/main/robotic_grounding>.
- NVIDIA code, `main` branch as retrieved on 2026-10-11 (the raw endpoint did not expose a SHA; before using it in a reproduction, pin the repository’s current commit):
  - `robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/mdp/rewards.py`, lines 374–445: <https://raw.githubusercontent.com/nvidia-isaac/video_to_data/main/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/mdp/rewards.py>
  - `.../tasks/v2d/mdp/utils_jit.py`, lines 278–325 and 437–485: <https://raw.githubusercontent.com/nvidia-isaac/video_to_data/main/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/mdp/utils_jit.py>
  - `robotic_grounding/README.md`, lines 423–439 (force-closure recipe for noisy monocular video): <https://raw.githubusercontent.com/nvidia-isaac/video_to_data/main/robotic_grounding/README.md>
  - `robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/README.md`, lines 24–37 (wrench matching versus force closure by environment): <https://raw.githubusercontent.com/nvidia-isaac/video_to_data/main/robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/README.md>

## What the wrench represents

At each timestep and object part, CHORD receives contact positions and contact normals in the **object frame**. A feasible edge force `f ∈ R^3` at contact `p` produces a six-vector

`w = [ f ; p × f ] ∈ R^6`.

The friction cone is approximated with `d` unit-magnitude edge forces per contact, giving a wrench matrix `W ∈ R^(6 × c·d)`. The method does not use a measured net force/torque label and does not recover the exact multi-contact feasible polytope. It compares directional capability of the contact configuration.

A fixed set of `b` unit directions `B ∈ R^(6 × b)` is sampled. The support vector is `σ = max_col(Bᵀ W) ∈ R^b`, with the max independently taken over primitive wrenches. The robot and reference support vectors are compared with a relative tolerance `β`:

`r_cws = exp(-||max(0,(1-β)σ_h - σ_r)||²/v_cws - ||max(0,σ_r-(1+β)σ_h)||²/v_cws)`.

The paper adds penalties for unintended contacts (`σ_h=0, σ_r>0`) and missed contacts (`σ_h>0, σ_r=0`). The released implementation also masks supports at `>1e-3`, averages over active hand/body/basis groups, and computes the two directional squared errors before the exponential (`utils_jit.py:437–485`).

The torque part is therefore the lever-arm cross product `p × f`, expressed in the object COM frame and normalized in the implementation by a body-radius scale `rc` before projection onto the basis (`utils_jit.py:278–325`). This is a **geometric contact capability representation**, not a time series of wrist sensor values.

## Temporal alignment and training target

The reference is indexed by the environment’s current trajectory frame. The paper defines per-frame hand keypoints and object-part poses, then retargets human hand motion into robot configurations. Contact positions/normals are sampled from the same reference frame. The released command caches the reference support vector at `timestep_counter` (`utils_jit.py:101–104`). There is no separate future horizon, temporal convolution, or flow target in CHORD’s wrench term.

CHORD’s RL reward is combined with object tracking and hand imitation (`r = r_task + r_imit + r_contact`). The policy itself predicts residual wrist/finger actions. The released README states that noisy monocular reconstructions should use the reduced **force-closure** reward, because matching noisy reference contact geometry can be misleading. In that variant, a binary reference contact-present gate selects the active hand and the reward is the fraction of live wrench-basis directions whose support exceeds `min_support`; it intentionally does not read noisy reference positions/normals.

The paper’s Appendix B reports residual wrist position/orientation/finger actions and a wrist force/torque tracking controller. These are action execution details, not supervised wrench outputs of the learned policy.

## Reusable implementation status

There is a public NVIDIA implementation of the contact-wrench computation and reward in `video_to_data`. The reusable portions are:

1. transform contact points and force directions into object COM coordinates;
2. construct friction-cone rays from normals;
3. construct six-dimensional `[force, torque/rc]` rays;
4. project against fixed six-dimensional basis directions and max-reduce;
5. apply CHORD’s tolerance/exponential reward plus unintended/missed contact terms.

The implementation depends on Isaac Lab contact sensors, object COM poses, per-body geometry, and a trajectory command cache. It is not a standalone PyTorch loss that can consume `(xyz, point features, target flow)`.

The current Task now contains a small dependency-free geometric adapter at
`src/oakink_wm/chord_wrench.py`. It constructs friction-cone rays from
object-frame contacts/normals, applies body-radius lever-arm scaling, builds a
fixed-basis support vector and reproduces the tolerance/missed/unintended
support loss. Its tests use finite synthetic contacts only. This verifies the
wiring and masking contract; it is not evidence that the official H5 or
EgoTouch packs contain the required contact geometry.

## Consequence for the current PointWorld/CM route

The current `VideoPointWorldWM` predicts pointwise 3-D flow with 24 temporal slots and uses masked Huber flow loss (`src/task/cm-pointflow-effect-pretrain/src/oakink_wm/video_pointworld.py` and `video_points.py`). CHORD cannot be copied as a direct replacement for that loss: current video/H5 clips contain scene trajectories and camera/robot metadata, but no per-contact object-frame normals plus friction-cone geometry suitable for `σ_h`. DROID gripper pose/action fields alone do not identify contact force directions or object-part meshes.

A feasible staged use is:

- **Downstream RL gate (closest to CHORD):** after the CM/PointWorld predictor is connected to the simulator, compute live contact supports from Isaac contact geometry and add the released CWS/force-closure term as an auxiliary reward. This preserves the current point-flow objective and tests whether wrench guidance improves contact-rich behavior.
- **Native supervised auxiliary head (only where labels exist):** for OakInk2/GRAB or simulated rollouts with object meshes and contact normals, add a small six-dimensional or basis-support head from the pooled object representation. Supervise basis supports or use the CHORD tolerance loss at the same current frame. Do not call it CHORD supervision when only point flow is available.
- **Synthetic teacher pretraining:** generate contact normals and friction-cone supports from the existing rigid simulator/mesh data, then train the head jointly with point-flow prediction. Use a matched `wrench-on/off` ablation; keep the head out of the official-DROID-only arm unless geometry/contact labels are added.

A direct “predict wrench from official DROID H5” experiment would be invalid without an explicit label-generation step. The H5 scene-flow target can still pretrain the shared PointWorld representation, but CHORD’s force/torque geometry must come from contact/mesh/simulator data or a separately audited reconstruction.

## Recommended first experiment

Implement the downstream CWS term in a small native simulator probe before changing the PointWorld head: current policy/CM, CWS off versus CWS on, same seed and environment, short horizon, report object-flow error plus contact success and unintended/missed contact rates. If contact geometry is too noisy, use the released force-closure variant and its binary contact gate. Only after this probe shows a change should a supervised wrench-support head be added to the shared PointWorld representation.

This recommendation follows the public code’s data contract and avoids treating a reward-only method as if it supplied force/torque labels.
