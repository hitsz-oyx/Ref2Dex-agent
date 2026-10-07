---
schema: ref2dex.probe.v2
probe_id: P-20261007-consequence-oracle-headroom
experiment_id: P-20261007-consequence-oracle-headroom
date: 2026-10-07
task: consequence-evaluator
branch: consequence-evaluator
git_commit: 7c85d5d
claim_id: C3
hypothesis_family: HF-consequence-oracle-headroom
probe_index_in_family: 1
seed_pool: probe
seeds: [291, 292, 293, 294, 295]
decision_changed_if_positive: adapt PointWorld to native robot physical futures and measure recovery of oracle ranking gain
decision_changed_if_negative: audit local labels, coverage and evaluator fit before further WM integration
status: UNCLEAR
run_id: oracle-headroom-20261007
---

# Does observed physical future add ranking information to recorded robot actions?

Result: Implementation and44synthetic engineering tests pass; no real six-expert data or matched fit yet.
Decision: Qualify the newly trained parent, rebuild six distinct experts, then collect continuous episodes and run the bounded matched Probe.

## Purpose and decision

Follow userref1: fixedK24/Kexec8 and inherited native policy H. Compare
E0(H,A) with Eoracle(H,A,Z_GT), with identical architecture/initialization,
data/labels/split and optimizer budget. Z is anchor-relative24-step rigid
object motion only, directly compatible with PointWorld's physical outputs.
Reward, contact/drop/success flags, phase identity and perturbation metadata
are not model inputs. No oldY target, forked branches or online proposal.

The decision is whether additional future physics information justifies
robot-domain PointWorld adaptation. Recorded reactive A already depends on
future feedback; this measures conditional association, not same-state
counterfactual candidate selection. A Probe cannot prove final policy utility.
Primary-source boundaries are in `../../research/PRIMARY_SOURCES.md`.

## Preconditions, coverage and resources

Blocker: old experts/robot outputs are missing. The separately bounded parent
rebuild must first pass its45-frame/no-later-drop gate. Rebuild six genuine
distinct self-trained roles with new weight hashes and route config before
collecting this dataset; do not put one parent into six route slots.
Further expert fit recipes/budgets are recorded after parent qualification.

On GPU0 only, each split collection uses24env/2waves, <=600steps/episode,
<=900s/2GiB. Seeds292/293/294 define train/val/test collection groups before
windows are formed. At most144initial episodes, with clean plus approach/
contact/grasp/lift/hold assignments and one smooth24step residual per episode.
Retain untriggered assignments as such; they do not prove perturbation coverage.
Task/stage/event counts must support both successful and bad local comparisons.
The cheapest follow-up fills a missing stage, not a broad unbounded rollout.

Labels/preparation are CPU file/statistics work, <=600s/2GiB and20000windows.
GPU matched fit seed291:1000updates,32pairs and32separately drawn reliable
expert windows/update, <=1800s. Independent frozen-weight evaluation seed295:
<=300s. Total GPU use remains at most3including PointWorld1/2, under4hard cap;
20GiBdisk reserve and300GBoutputs cap apply. No other jobs are stopped.

## Supervision and implementation invariants

`label_continuous.py` verifies native six-weight/source provenance and raw
episode/sidecar hashes. It preserves actual H/A/poses/clock and split groups.
Sample cadence8 does not change the24step horizon. A clean episode qualifies
for absolute progress only after45consecutive elevated-contact-proxy frames
and no later drop (<2cm or6lost-contact frames). Progress is original episode
time divided by verified completion time, saturated after completion; never
renormalize each window. Failed/suboptimal/perturbed progress stays masked.

Local preferences use only t..t+24: maintained_hold > unrecovered_drop and
lift_achieved > grasp_lost, with same split/task/current phase, different
episodes and initial relative heights within1cm. At most2comparisons per
unordered episode pair and64per split/task/phase/event stratum. Ambiguous
approach/miss/recovery examples abstain and may need explicitly grounded
annotations. Event rules adapt sustained-lift semantics; they are not
Robometer's literal video labels or DenseReward's reward recipe.

No eventual episode quality is consulted for local preference direction.
Missing split preferences or clean train-progress anchors disallow training.
The native contact quantity is hand+object net-force proxy, not identified
hand-object collision pairs. Physical diagnostics stay outside Z.

Both evaluators use shared train-only normalization and the same frozen pair
draws plus independent but shared expert-window draws. The separate expert
draw prevents progress training from disappearing when all ranking endpoints
are perturbed/suboptimal. Scalar Bradley–Terry ranking is the documented
Robometer-inspired adaptation; the null future module remains present.
Fixedval strict preference accuracy selects each arm's first bestcheckpoint.

## Evaluation and next action

Before test inference, freeze weight hashes and protocol. Report strictpair
accuracy (ties count wrong), paired oracle-minus-baseline gain, task/phase/
quality and episode-pair macro summaries. Progress MAE uses only reliable
masked frames. Replace Z with same-task/same-current-phase other-episode
futures under fixedseed295; keep H/A/labels unchanged. This diagnoses future
alignment dependence, not executable action candidates or a permutation test.
Overlapping windows are not independent evidence; do not claim significance
from rawpair count or tune on test. Freeze best weights onval only.

PROMISING means the oracle's ranking gain warrants PointWorld adaptation;
UNCLEAR means coverage/labels/fit need a targeted check. A negative result is
not sufficient to refute all world models. No formal Validation claim here.

## Outputs and limitations

Raw, labeled, window, fit and evaluation outputs belong under
`outputs/consequence-evaluator/`, each fresh run ID and runtime source hashes.
Real paths/PIDs/counts/metrics will be recorded after the expert gate. Current
44tests are synthetic engineering checks, not empirical oracle headroom.
Expert coverage, original backup recovery, PointWorld native-control to hand
point-flow adaptation, EWM, a deployable24step proposal policy and multi-seed
matched trained-policy Cm-on/off utility remain pending.
