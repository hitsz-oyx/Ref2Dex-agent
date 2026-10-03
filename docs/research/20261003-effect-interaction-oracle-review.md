# Oracle effect + interaction: primary-source review and experimental boundary

User proposes testing perfect effect and interaction information before another
Cm model. This note adopts an oracle information-value question, not a claim
that joint prediction must succeed. Reviewed3October2026 in the same session;
no research subagent or native/training experiment was launched.

## Primary sources and what can be borrowed

1. **D-Grasp, CVPR2022.** Reviewed state/features/reward and Table4 in
   [author full text](https://arxiv.org/html/2112.03028), also
   [CVF paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Christen_D-Grasp_Physically_Plausible_Dynamic_Grasp_Synthesis_for_Hand-Object_Interactions_CVPR_2022_paper.pdf).
   The state contains hand/object pose/velocity and contact-force information;
   the derived features include object-relative hand geometry and target contact
   joints. Reward includes desired-contact achievement and capped contact force.
   Table4 reports success.89 for full method and0 without contact reward.
   This supports testing relational/contact guidance; it is a reward ablation,
   not proof that simply adding oracle contact observations causes that gain,
   and concerns a different hand/task. Target grasp contacts are goal priors,
   distinct from the actual current contact manifold.

2. **DexRepNet, arXiv2303.09806v4,17October2023.** Reviewed Sec.III-A,
   IV-B/C and TableI in [full text](https://arxiv.org/html/2303.09806).
   DexRep combines palm-centred occupancy, hand-to-object nearest-surface
   distance/normal, and local geometry descriptors. The policy uses these
   relational features with hand states. Ablations support the local features;
   adding a global PointNet feature does not automatically help. Borrow an
   object-centred local representation and controls for individual components,
   rather than an unstructured dump of all privileged values. These are
   geometric potential-contact features, not attributed contact-force ground
   truth or future physical effects. Reported gains do not transfer by assumption
   to our own three-reference airplane task.

3. **UniDexGrasp, CVPR2023.** Reviewed Sec.3.2.4/3.3 in
   [full text](https://arxiv.org/html/2303.00938).
   The oracle teacher sees hand proprioception, object state, sampled object
   geometry and a goal grasp label; a student uses scene point clouds. Contact
   maps refine grasp proposals. Borrow privileged-teacher-first qualification
   and distinguish goals from observations. This oracle is not a future-query
   world model and does not establish the value of effect+interaction outputs.

4. **ContactNets, CoRL2020/PMLR2021.** Reviewed Sec.2/4 in
   [primary paper](https://proceedings.mlr.press/v155/pfrommer21a/pfrommer21a.pdf).
   Dynamics decompose non-contact impulses and contact Jacobian-transformed
   impulses. Learned inter-body distance/contact-frame structure is linked to
   motion through complementarity/friction constraints, without measured contact
   labels. This motivates consistency between interaction and effect, rather
   than two disconnected output heads. It also shows that separate interaction
   supervision is not logically necessary for every useful effect model.

5. **DexTouch-WM, arXiv2609.20649v2,18September2026.** Reviewed formulation,
   model overview and policy-evaluator/data-generator sections in
   [full text](https://arxiv.org/html/2609.20649), verified date/status in
   [primary record](https://arxiv.org/abs/2609.20649).
   The action-conditioned model jointly predicts RGB and bilateral tactile
   dynamics using compatible human/robot sensing layouts and retargeted actions.
   It evaluates policy ranking and synthetic-data policy learning. Its TableIV
   includes both improvements and degradations, so stronger predictions do not
   imply uniform control gains. It is a recent preprint; the record lists an
   IROS2026 workshop lightning talk, not a top-journal result. Generic joint
   visual/contact prediction and human-to-robot scaling are already prior art.

6. **Dex-X, arXiv2609.07747v3,1October2026.** Reviewed Sec.3.3 and Appendices
   F/G/I in [full text](https://arxiv.org/html/2609.07747).
   Simulated interaction supplies tactile supervision to a privileged expert,
   followed by visual-tactile distillation. Default tactile inputs are five
   fingertip-force magnitudes averaged over two samples;15 reserved contact-
   position channels are zero. The critic's future trajectories are *reference
   targets*, not actual counterfactual outcomes. Borrow teacher qualification
   and contact-aware learning, without misdescribing its available interaction
   state as a full contact manifold. Recent preprint evidence is suggestive,
   not an independent confirmation of our proposed oracle.

Scope: focused primary-text passages/tables, not an exhaustive literature
review, author-code audit or independent replication. No source performance
number is treated as an expected Ref2Dex result.

## Our evidence and why the proposal remains untested

- Existing actual8step-successor diagnostic improves fixed success-prediction
  Brier from.1224600595 to.0844091163 (31.0713%). It gives the realised future
  of an executed action to a value predictor. No oracle-assisted closed-loop
  grasping or explicit hand-object contact-pair input was evaluated.
- Current152-input policies already see simulator object pose/twist, joint
  state, five SDK hand-body poses/orientations/net forces, and previous-step
  motion/clearance. Adding these again is not a new privilege comparison.
- A body net force does not identify contact partners or contact locations.
  Static geometry proximity is also not a true contact manifold. Earlier
  experiments must not be retrospectively relabelled as the proposed oracle.
- Same reset packet across parallel environments failed physical-prefix
  matching in the paired-option screen. Candidate futures from separate runs
  are not yet a verified same-state oracle. An actual branch-query mechanism
  needs an engineering verification before causal action selection claims.

## Proposed information contract

Effect E*: a **fixed short** action-conditioned object SE(3)/twist trajectory,
not only XYZ translation, under a declared executed hand-action sequence.
Interaction I*: hand-link/object-surface relations, actual contact pairs and
positions/normals, normal/tangential response where available, and local
relative motion/slip. Include hand-table/self-contact attribution separately.
Potential-contact geometry should remain available before first contact.
Mass/inertia/friction and reference goals must be separately labelled.

These are proposals, not a verified available API contract. Local IsaacGym
`RigidContact` docs list body0/body1, contact positions, normal, friction and
lambda/lambda_friction. The current GPU tensor is only per-body net force.
GPU-pipeline availability, units, substep aggregation, contact filtering and
slip reconstruction still need a native engineering smoke; no assertion that
all fields or solver memory are retrievable has been made.

Two oracle regimes must stay separate:

* **Current privileged teacher:** actor receives true current relational/contact
  state; it may use target object motion as an explicitly labelled goal. No
  actual future outcomes are supplied. Tests privileged-state usefulness.
* **Action-query oracle:** candidate action sequence is specified first; an
  oracle returns its true short future E/I from the actual decision state.
  A controller can then rank candidates or learn a policy using queries.
  Tests short-transition information value. Query cost is reported separately.

Giving a realised future to an actor before defining its action is circular.
Giving final success/expert action as a feature makes the ceiling trivial.
Future reference targets cannot substitute for actual response ground truth.
An information oracle's optimum can be at least as good when it may ignore
extra inputs; finite training and restricted control do not guarantee that
this optimum is reached. A successful teacher measures an attainable oracle
reference, not the mathematical maximum over all controllers or a guarantee
that a learned Cm can reproduce it.

## Smallest useful comparison

First qualify actual contact extraction and any effect-query replay. Then a
matched no-Cm factorial: base observation o; o+E*; o+I*; o+E*+I*. E-only and
joint arms must use the same verified oracle horizon/query protocol; hold
policy interfaces, initial own actor, architecture capacity, rewards, interaction
budget and evaluation protocol fixed. Current-contact versus future-contact
privilege must be declared consistently. Start on nominal corrected airplane
physics, retain all three motions and full105tick holding/drop criterion.

Primary decision uses **fresh deployed grasp success**, not Brier/loss or
retrospective best-of-rollouts. Joint>effect-only tests added interaction
value; joint>baseline tests usable oracle headroom. Only then replace oracle
channels with learned Cm, with learned-effect/oracle-interaction and converse
hybrids to locate the approximation gap. Positive query-controller results
still need a subsequent matched actual policy-training comparison for MissionB.
If the tested oracle fails, first distinguish interface/optimizer/action-support
failure from absence of information; a negative finite run is not a universal
Cm impossibility proof. Gates/budgets will be fixed in an experiment card
after the concrete retrievable oracle contract is qualified.
