---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref22-multitakeover
experiment_id: P-20261012-goal-point-wam-ref22-multitakeover
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: 28d3498d
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 31
seed_pool: probe
seeds: [292]
decision_changed_if_positive: adapt the retained anchored policy with true multi-takeover recovery futures
decision_changed_if_negative: inspect expert recovery feasibility and actual late-state support before further fitting
status: UNCLEAR
run_id: wam-ref2-multitakeover-20261012-r1
---

# ref2_2: actual student-state multi-takeover aggregation

## Decision Note

Probe30 retains R physical meaning and newly earns GT hold75/no clipping,
but pure actor0held persists: t12 PD103.44->21.98mm/wrist30.80->9.33mm;
t24 PD265.83mm/t32 wrist354.10mm. Stop loss-only optimization as requested.
The next hypothesis is whether the current learned student's late states
have usable actual expert recovery futures, necessary before fitting broader
closed-loop support. This serves the self-trained manipulation subgoal; no RL.

Cheapest decisive step: retain the already-completed128/t8 source, then TWO
fresh128-control collection groups using the SAME exact learned WAM/R pair,
takeover16 and32. Role3 applies the student's actual commands until takeover,
then queries/applies expert on ITS OWN measured state. Each has own observed
H4, current-q expert PD, actual applied action and measured consequence.
No hidden-state copying, nominal future substitution or role2 borrowed future.

Pair: wam-ref2-joint-anchored-20261012-r1, headSHAc30b48c6c28b37c14e541f23b7dcf28cf5b59a835df1858de83beb1a3a0ee094,
RSHAf1f6472ca967c34a4d0eeea9ace01290051fc3efeef57becaa90a08101db317e.
UseGPU2/execution292; identical reset is development TRAIN collection, not
fresh statistical validation. Each group<=600s/total<=20GPUmin, new outputs
<=100MB/cumulative ref2<=850MB/global400GB; stop on source drift, nonfinite,
foreign GPU, incomplete controls or budget. No new training under this card;
next fit requires an explicit objective and retained-model screens.

## Scoped aggregation and checks

Exact audited128 states129, controls128 and durable query records; all native
PD/applied/current-state/own-H4/reference-goal root checks must pass. Compare
same learned pair identities across8/16/32. Full24 future windows at actual
role3 currenttakeover..104 only when all causal controls are expert/unclipped.
These are expected97/89/73 future rows. Pure role2 expert queries that are
unclipped supply CURRENT PD only; their tau placeholder is zero and
trajectory_valid=false, so future geometry and R-supervision are denied.

Save source/role/tick, actual own future displacements/current87D R state,
own observed H4/known goal/current-q expert PD, actual applied PD/commands
and measured next hand/object. Original native admissions stay unchanged;
derived scope grants only these actual role3 expert futures. The t8 source
is the postprocessed complete physical trace from Probe30: original failed
terminal bookkeeping record remains immutable, full/partial snapshot match,
root2316 checks all0. The derivative does not represent another simulation.

Report whether each expert takeover actually recovers/holds and any excluded
or unusable futures. Failed takeover recovery limits teacher support; it is
not proof WAM capacity is insufficient. The next decision is dataset/model
coverage, not another loss coefficient sweep.

## Limitations / future evidence

Temporal windows overlap, one motion/reset and privileged expert training
labels. Same-policy independent live launches can vary. Aggregation is TRAIN
data and retires these queried states from independent evaluation. More data
does not establish successful learning; new trained-policy rollout and later
formal matched Cm-on/off remain necessary. No world/evaluator/RL started.

## Results

Collection and aggregation **COMPLETED**. Usable actual expert-future support
is locally **PROMISING**; autonomous learning benefit remains **UNCLEAR**.
No additional policy fit or rollout after aggregation has been run.

| Takeover | Physical run | Wall seconds | Expert held frames | Actual future rows | Pure current-PD rows |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 | `wam-ref2-execution-anchored-t8-20261012-r2-recovered` | reused Probe30 | 74 | 97 | 128 |
| 16 | `wam-ref2-execution-anchored-t16-20261012-r1` | 97.23 | 75 | 89 | 128 |
| 32 | `wam-ref2-execution-anchored-t32-20261012-r1` | 92.14 | 41 | 73 | 128 |

Every source contains128 completed controls/129 measured states. Same exact
head/R identity, unclipped expert applied controls and current-PD labels are
verified. Each independent native root audit has2316 checks/max0. Every pure
actor remains0held. Takeover32 shows41 held frames at the128-control cutoff,
with final object height.0705m; it does not reach45-frame sustained-hold
qualification here. This short trace cannot establish failed expert recovery,
placement success, autonomous success or generalization.

Original t8 FAILED terminal records remain preserved; its derivative reuses
the single completed physical trace. t16 runs code28d3498d, t32 f8ffd7e4;
aggregation records96de597cc42309f8e3fe8450ec0a6ed6527c3b32. Those commits
include concurrent video-line work; the exact retained robot checkpoint pair
and input hashes, rather than Git HEAD alone, define this collection identity.

Artifact: `outputs/trajectory-policy/wam-ref2-multitakeover-20261012-r1/`
contains `aggregation.npz`, manifest and `root_aggregation_audit.json`.
Dataset SHA256:
`9c6f4b30afda0d559d0544e19e4bb1906e7d1cef0c88e1d3e34989bedef5a8b7`.
643 samples=259 actual expert-future rows+384 current-PD-only pure rows.
Independent per-point matrix-product reconstruction (without producer feature
helper) performs7073 checks: state/tau maximum1.19e-7 float-rounding delta;
PD/q/H4/goal/applied action/next state all0. All input hashes match, all values
finite and every pure tau is zero with trajectory_valid=false.
Auditor: project-local `tmp/ref22-audit/audit_aggregation.py`, hash recorded
in the audit; CPU used for file/label arithmetic, without model inference.

Decision: actual t8/t16 and partial t32 recovery provide admissible broader
supervision. Next fixed anchored fit should mix these actual futures and pure
current PD with nominal/startup retention, preserving GT-only R gradients;
after unchanged screens, one new student rollout must test coverage benefit.
No loss coefficient search, model-capacity expansion or RL. The queried
collection is TRAIN and cannot serve as independent evaluation of that fit.
