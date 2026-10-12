---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref2-robot-aggregation
experiment_id: P-20261012-goal-point-wam-ref2-robot-aggregation
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: a4158031
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 24
seed_pool: probe
seeds: [295, 296, 292]
decision_changed_if_positive: aggregate a new learned actor rollout before admitting full execution and eventual consequence or RL learning
decision_changed_if_negative: distinguish unsupported expert recovery or conflicting actor supervision before scaling the dataset
status: UNCLEAR
run_id: wam-ref2-execution-aggcollect-20261012-r1
---

# True autonomous-state aggregation with a later expert takeover

## Decision Note

User ref2_1 requests human pretrain -> robot teacher adaptation -> student
PD/consequence aggregation -> RL, and sustained work until19:30 today.
Experiments22/23 did not fix pure-WAM validation32:48. Their live training
excluded precisely that block and mostly fitted expert-recovered states.
Unconstrained tau inversion proves numerical local R sensitivity only, not
physically valid trajectory reachability or general retargeter correctness.
The reference's stronger attribution is treated as a hypothesis, not proof.

Root chooses one new bounded recovery collection from the best startup head,
not a blind full thaw or a full-task success evaluation. Move expert takeover
16->32 and collect192 controls, verifying that the harder autonomous prefix
is locally recoverable. Use actual current observations/q for expert PD; the
pure learner always executes WAM. Store actions and measured next states,
which are valid actual consequences but not expert future hand trajectories.

If calibration teacher/GT and later takeover each show held>=45 (takeover
first-stable<=160, clipping<1%), audited unclipped PD can enter one fixed500
update aggregation fit. Otherwise preserve the failed recovery and investigate
expert support before training against its labels. No placement conclusion
from192 controls. One GPU2, each collection/fit bounded<=600s, current/new
ref2 outputs plus scratch<=100MB, shared storage<=300GB. Remain main;
unknown jobs untouched. Up to3 aggregation iterations under this card, then
review evidence/cost before further work. No new permission boundary.

## Collection and input invariants

Pin original human final10000, original Git-authenticated model implementation,
startup head2b7eff74991a8ef306b192747d716210d4348d3259f628fa73c78dfc7b617fc2,
qualified actual-PD R and official expert. Default existing542/16 bridge
protocol remains unchanged. New controls/takeover parameters must be passed
to both peers, authenticated at READY, and reflected in manifests/results.
Planned goal remains original min(t+24,542), never the collection endpoint.

192-step result explicitly partial_episode=true, calibration_pass/recovery_pass
null, no placement evaluation. Local grasp calibration is separately named.
Check command-prefix pairing through measured state32 with declared physical
bounds; never copy/fork hidden PhysX state. Verify actual action/current-q PD,
live own H4, R current state, planned goal, durable pre-step expert queries,
pure-learner expert_applied=false and recovered expert_applied=t>=32.

## First fit and honest evaluation

Use fit_wam_robot_aggregation.py with this new independently audited source,
warm the original startup head. Freeze scene/goal/action/world/R; train only
trajectory decoder and observed-hand query. This probes coverage first;
partial-thaw evidence does not justify full backbone training yet.
Seed295/model,296/draw reused for deterministic recipe continuity, not a
replicate or new seed study. Fixed500 updates/final only, lr1e-4/wd.01/clip1.
Each batch128:64 nominal (16 startup+48 later),32 pure-WAM,32 recovered,
replacement. Same nominal tau/.05 + nominal PD/Rscale + live PD/Rscale Huber.
Live rows never have tau labels; no teacher future observation enters actor.

Pure training ticks0:63, recovered takeover:160, each excluding expert
clipping. Old validation32:48/64:80 is EXPLICITLY RETIRED into training and
cannot be reported as independent validation. Late pure73:96 and recovered
160:176 unclipped queries are correlated diagnostics only, with limited
expert recovery support. Save source/row/mask identity and raw predictions.
Nominal held panel remains a retention screen, not a fresh generalization set.

Describe train and diagnostic XYZ/rotation/finger errors by role. Offline
query-halving/nominal nearest8<=21.522mm/nominal PD<=38.335mm are exploratory
screens, not DAgger success. A new bounded192-control training collection
from the frozen learned actor may proceed if all weights/data/inference are
finite, nominal nearest8<=21.522mm, nominal PD<=50mm and nominal clipping<1%.
It must query expert on THAT new actor's actual states and apply no teacher
intervention to its WAM row. New actual PD error, measured divergence and
local held behavior determine the next aggregation, never training loss alone.
Only a separately admitted full542-control evaluation can judge placement.

## Limitations / future evidence

Same deterministic reset and one motion do not establish independent robot
generalization; new actor states are distribution-shift diagnostics. Expert
labels rely on privileged planned human/body/contact reference absent from
the actor, and late unclipped outputs are not guaranteed useful recovery.
Frozen world weights do not establish consequence quality. Actual consequence
records are retained for subsequent action-conditioned robot Cm training;
this first stage optimizes expert PD only and is not RL or Cm causal evidence.
Formal matched controls/multiple independent starts remain future Validation.

## Results

Result: later takeover32 locally recovers139 held frames; aggregation improves pure training636->143mm but nominal PD59.48mm fails50mm collection gate, late pure diagnostic remains1186mm.
Decision: do not deploy this head; test frozen R on actual expert-only recovery hand futures before further actor/data scaling.

Collectionr1 completed74.83s: teacher/GT-R held138/139, expert takeover held139,
first-stable98 and clipping0; pure WAM0 held/clipping1.0417%. Partial local
calibration/recovery true, complete calibration/recovery null. Independent
raw audit3468 checks max0. All192 actual expert queries persist. Training
source has63 unclipped autonomous rows and128 recovered rows. Actor never
receives expert observation; mixed-policy hand futures remain forbidden.

Engineering smoke passed1 update/13.75s; live PD gradient1399.58, both
trainable modules nonzero and all five frozen hashes/cache-direct unchanged.
Fixed500 fit completed112.41s fromoriginalstartup; raw source label/row
identities retained. Pure training XYZ p95636.50->143.26mm; recovered
62.36->30.13mm. Late pure diagnostic1194.21->1186.27mm (22 unclipped rows),
recovered25.13->27.73mm (16 rows). Nominal nearest8 EPE17.22->19.70mm,
nominal PD25.56->59.48mm. Nominal retention and pure diagnostic halving fail;
collection admission<=50mm also fails. No learned-actor rollout run.
HeadSHA979621c5282a6e91cb96941546ca0b385235c2f432d34ded48ae066a44c86624.
This is UNCLEAR; training improvement does not prove on-policy improvement.

### Next Decision: physically measured expert future -> frozen R

The new source offers137 complete unmixed expert windows, role3 current
ticks32..168, futurecontrols[t:t+24] all actuallyexpert. Independent review
found action/PD equality and current-state reconstruction max0. Unlike
unconstrained tau inversion, actual future hand motion is physically measured.
Run audit_wam_expert_recovery_tau.py on these ALL137 windows and same ticks
teacher row0, frozen R/current q; report PD XYZ p50/p95, rotation/finger,
clipping and fixed time blocks. GPU2<=60s/<1MB, still within this iteration's
600s/100MB allowance. If recovered GT-R is bad while nominal is good, address
R physical-trajectory mapping coverage rather than blaming only actor inputs;
if both good, prioritize trajectory prediction/supervision. This is privileged
diagnosis only. Source actor_future_hand_training_allowed=false is preserved;
any later expert-only training requires a separate derived scoped admission,
not reinterpretation of mixed-policy future or overwrite of the source.
