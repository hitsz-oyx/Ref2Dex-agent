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

# Does physical future add ranking information to decision-known residual plans?

Result: Ref2 contract and73synthetic engineering tests pass; no real six-expert data or matched fit yet.
Decision: Parent50/64 and s3 endpoint36/64 qualify; rebuild the remaining five experts, then collect continuous episodes and run the bounded matched Probe.

## Purpose and decision

Follow userref1/ref2: fixedK24/Kexec8 and inherited native policy H. Compare
E0(H,delta), Eoracle-E(H,delta,Zobject), Eoracle-EI(H,delta,Zobject,Zinteraction),
with identical architecture/initialization, data/labels/split and optimizer budget.
Delta is the24step18D requested smooth residual schedule known BEFORE execution.
The expert remains feedback-driven: u=clip(pi(Hfuture)+delta), with native
noise/coupling audited separately. Actual reactive future controls and
actual_residual are NOT evaluator inputs. Before a pending state trigger, its
future starting time is unknown, so such windows cannot enter the dataset.
Clean has an all-zero known plan; triggered plans shift their remaining schedule
and append zeros after the one chunk. Old v1 data/fit contracts are rejected.

Zobject contains12values/step from Tcurrent^-1 Tfuture. Zinteraction contains
11 measured hand rigid-body keypoints relative to EACH future object pose,
33values/step. All arms retain the same45D future module; zero unused channels
AFTER train-only normalization. No reward/contact/drop/success flags, phase
identity, quality or perturbation metadata enter the model. Neither18D residual
plans nor the interaction output are yet connected to PointWorld hand-flow.
The result motivates robot-domain physical-future adaptation, but cannot prove
that the current object-only PointWorld recovers both oracle gaps. No forks,
oldY targets or online proposal; observational matching is still limited.
Primary-source boundaries are in `../../research/PRIMARY_SOURCES.md`.

## Preconditions, coverage and resources

Blocker: old experts/robot outputs are missing. The separately bounded parent
rebuild must first pass its45-frame/no-later-drop gate. Rebuild six genuine
distinct self-trained roles with new weight hashes and route config before
collecting this dataset; do not put one parent into six route slots.
Further expert fit recipes/budgets are recorded after parent qualification.

On GPU0 only, each split collection uses24env/2waves, <=1200steps/episode,
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
for absolute progress only after45consecutive elevated force-proxy AND <=1cm sampled hand/object surface-gap frames
and no later drop (<2cm or6lost-contact frames). Progress is original episode
time divided by verified completion time, saturated after completion; never
renormalize each window. Failed/suboptimal/perturbed progress stays masked.

Local preferences use only t..t+24: maintained_hold > unrecovered_drop and
lift_achieved > grasp_lost, with same split/task/expert/motion/current phase, different
episodes and initial relative heights within1cm. Additionally current world
object translation<=3cm/z<=1cm/rotation<=15degrees and hand11point RMS<=2cm. At most2comparisons per
unordered episode pair and64per split/task/phase/event stratum. Ambiguous
approach/miss/recovery examples abstain and may need explicitly grounded
annotations. Event rules adapt sustained-lift semantics; they are not
Robometer's literal video labels or DenseReward's reward recipe.

No eventual episode quality is consulted for local preference direction.
Missing split preferences or clean train-progress anchors disallow training.
The native contact quantity is hand+object net-force proxy, not identified
hand-object collision pairs. Force and gap diagnostics stay outside Z. Keypoints are measured geometry.
Before expanding collection, inspect force-proxy/geometry agreement on a bounded
actual native trace, including proxy-far/table examples. This is not pairwise GT.

All three evaluators use shared train-only normalization and the same frozen pair
draws plus independent but shared expert-window draws. The separate expert
draw prevents progress training from disappearing when all ranking endpoints
are perturbed/suboptimal. Scalar Bradley–Terry ranking is the documented
Robometer-inspired adaptation; the null future module remains present.
Fixedval strict preference accuracy selects each arm's first bestcheckpoint.

## Evaluation and next action

Before test inference, freeze weight hashes and protocol. Report strictpair
accuracy (ties count wrong), object-minus-baseline, interaction-minus-object and interaction-minus-baseline gains, task/phase/
quality and episode-pair macro summaries. Progress MAE uses only reliable
masked frames. Replace Z with same-task/expert/motion/phase and matched-current-object/hand other-episode
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
73tests are synthetic engineering checks, not empirical oracle headroom.
Expert coverage, original backup recovery, PointWorld native-control to hand
point-flow adaptation, EWM, a deployable24step proposal policy and multi-seed
matched trained-policy Cm-on/off utility remain pending.

## Ref2 implementation and current execution boundary

User authorized contract repair before further experiments. Raw/windows schemas
are v2; collect logs residual_plan before env_step and stores measured11point
hand geometry. Event labels require independent sampled proximity; reports
retain force-near/far/elevated agreement counts. Pairing and future donors both
match controller/reference/current geometry. The1200step cap replaces600 only
because restored references include1062frames; <=900s/2GiB remain unchanged.
Qualification intentionally retains its historical net-force operational gate,
with training_allowed=false. Parent50 and s336 are NOT geometry-audited evaluator
labels. Six experts, native geometry smoke and actual contact audit remain
pending; GPU0 unavailable. Current73tests are finite CPU engineering checks,
not a CPU substitute for real evaluator fitting.
