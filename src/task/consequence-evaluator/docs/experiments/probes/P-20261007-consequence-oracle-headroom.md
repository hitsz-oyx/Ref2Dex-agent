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

Result: Six-route continuous collection completed, but strict state-matched labels are insufficient for fitting: train/val/test contain 0/0/2 local preference pairs. The physical geometry audit passes; no evaluator fit was started.
Decision: Repeated waves make the label gate READY at 2/1/5 train/val/test pairs, but this is too sparse for a useful fit; stop before evaluator training and design paired current-state branches.

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
74tests are synthetic engineering checks, not empirical oracle headroom.
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
pending; GPU0 unavailable. Current74tests are finite CPU engineering checks,
not a CPU substitute for real evaluator fitting.

Blocker engineering check: while GPU1 is temporarily idle during three-source
file/statistics preparation, run native_reset_probe --reset-mode batched
--geometry-steps128 on8env/originals1 self-trained parent; <=180s/1MiB.
This checks actual measured body/keypoint/surface wiring and records bounded
force-near/far/elevated agreement, before any data expansion. No PPO/collector
fit, no expert substitution, no success or scientific gain claim. GPU2 may
compute file-derived physical statistics concurrently; max2ownedGPU. Preserve
all raw/checkpoint/input hashes and stop on nonfinite/teleport/early completion.

First native geometry smoke r1 exposed tensor-pipeline env_step returning a
raw observation tensor whereas env_reset returns an obs dictionary. The new
continuous loop previously assumed both were dictionaries (fake tests modeled
that assumption). Preserve failed r1; repair by wrapping the already measured
tensor without env_reset or re-normalization. Add both raw-tensor/dictionary
driver regressions; retry same128step engineering check in fresh r2. This is
an implementation issue before data collection, not evaluator negative evidence.

Native r2 then exposed the custom run loop omitting get_batch_size before
policy inference, yielding1×11536 vs1442×1024 instead of an8-env batch.
Repair both probe and collector to call the released batch initialization,
with explicit env-count assertion; driver regression now requires it before
get_action. Preserve r2 and retry r3 under the original bounded geometry scope.

Native geometry retry r3 at9e1da03 completes24.24s on GPU1:8env×128steps,
1024 measured frames, finite11keypoints and sampled gaps. Forceproxy712frames
all have<=1cm gap; elevated force568frames likewise all near; proxy-far>3cm=0.
Median gap.772mm. Reset persistence, FK and repeated subset checks pass.
Output: outputs/consequence-evaluator/ref2-native-geometry-20261007-r3/.
This establishes original-s1 wiring and bounded consistency only; no table
false-positive examples, other objects, full episodes or collision-pair truth
were tested. Full collection should retain per-route audits; six-expert fit
and genuine three-arm data remain pending.74Task tests pass after both native
loop fixes. Failed r1/r2 are preserved as engineering records.

## Duck geometry precheck and probe contract correction

Duck r1 fails before geometry at a post-physics FK velocity residual gate:
4.084mm/s and11.340mrad/s, despite body position2.429micrometers,
quaternion6.97e-7 and object first-step2.864mm. These are measured after
a full PhysX/TGS step, not at submitted reset. There was no calibrated
contract requiring solver body twist to equal differential FK at the
new integrated q; this gate cannot establish a reset bug. Preserve r1.

Independent read-only review identified this timing distinction. Root
keeps strict post-step pose/persistence/subset checks and gates velocities
at the submitted reset cache. Post-physics velocity residuals remain in
the report as diagnostics, without claiming their cause resolved. Fresh
r2 repeats the same8env×128duck geometry check,<=180s/1MiB; no physics,
reset implementation, expert checkpoint, labels or PPO recipe changes.

Collector native input inventory also repairs immediate directory links
(the old rglob missed every staged tensor), with broken/missing/duplicate
sequence rejection. Frozen dependencies now include shared geometry,
planner, oracle table alignment, native common_player and surface loader.
This is engineering readiness, not oracle headroom evidence.

Duck geometry r2 at4e17253 COMPLETED25.625s:1024frames,533forceproxy
frames all corroborated by<=1cm sampled gap, median2.238mm; no elevated
frames and no proxy-far>3cm examples. Reset cache velocity errors0, strict
post-step pose/persistence and subset checks pass. This confirms wiring
for another actual object, not reliable hold/contact truth or a full audit.

Independent review also exposed global env-index modulo6 phase assignments
aliasing native object layouts: default24env×2waves gave cup/waterbottle
no clean episodes, and changing seeds did not change assignments. New
collector rotates clean+5phases within actual motion across waves, with
seeded env permutations and explicit intended/triggered coverage by
expert/motion. Missing reached stages remain missing; no outcomes choose
assignment. This protects clean progress anchors and directs later
coverage completion without adding future feedback to requested plans.
The first real collection will choose env/wave counts from measured cost
and recorded motion coverage; the old144episode cap is not evidence that
every motion/phase was covered. No collector or evaluator has run yet.

Source inventory keeps the actual runtime alias paths, not just resolved
old targets: retargeting a native motion/assets link is detected even if
its previous file still exists. The real linked-input regression checks
that drift, missing/broken tensors and duplicate aliases.83Task tests pass,
with14collector tests; real six-expert collection remains pending.

New route artifact entry prepare_expert_route.py requires six actual distinct
owned trained roles and their completed64frame0 qualification traces. It
freezes ancestry/qualification/rawtrace hashes and preserves failed-role
readiness explicitly. Default airplane readiness remains required; failures
may be observational suboptimal examples, not reliable progress experts.
Collector rechecks/fixes these generated provenance dependencies, while
labeling still refuses missing actual clean progress or per-split pairs.
Six new route-contract regressions pass; this does not yet produce a real
six-expert route, as four further role checkpoints remain missing.

## Six-role route and first continuous collection (2026-10-08)

The fresh specialist endpoints are frozen in
`outputs/consequence-evaluator/expert-route-20261008-r1/route.json` with six
distinct endpoint hashes. Fixed qualification counts are airplane_base36/64,
duck8/64, cup63/64, mixed12 0/64, train5 5/64 and balanced5 4/64. The route
therefore remains `training_allowed=false` and the three weak roles are kept as
observational candidates only; this does not recreate the old six-expert
Validation substrate.

The route nevertheless passed the native provenance checks for a bounded raw
collection. Three runs (`continuous-20261008-{train,val,test}-r1`) completed
48 episodes each (144 total; 24 environments, two waves, 30 Hz, 1200-step
cap). Raw episodes retain only H, executed A, decision-known residual plans,
measured hand keypoints, object poses and clocks; physical contact/gap fields
remain diagnostics. All source manifests completed and their hashes were
accepted by the labeler.

The label output is
`outputs/consequence-evaluator/labeled-continuous-20261008-r1/label_report.json`.
It has 144 episodes and 11,733 selected windows, but status
`INSUFFICIENT_PREFERENCES`, with pairs train/val/test = 0/0/2. The event
windows contain 2,530 unambiguous local events (maintained hold 2,005,
grasp lost 404, lift achieved 74, unrecovered drop 47). The independent
geometry audit reports 99,894 valid frames, 31,297 native proxy frames and
29,115 proxy-and-near frames; only one proxy frame is farther than 3 cm. The
immediate blocker is state coverage rather than a failed geometry wire.

The reproducible matching audit is
`outputs/consequence-evaluator/pair-coverage-diagnostic-20261008-r1/coverage.json`.
At the declared object/height/rotation/hand limits, pair counts are 0/0/2;
diagnostic hand limits of 3/5/8 cm give 1/1/3, 2/1/4 and 2/1/5. These relaxed
counts are not labels and do not change the protocol. No matched evaluator
fit or test evaluation was launched, because `training_allowed` is false.

### Decision Note: repeated-wave state coverage

Question: can the missing train/val pairs be obtained by repeating the same
motion/phase assignments while leaving the physical-state contract unchanged?

Evidence: the first collection has ample raw windows and corroborated gap
measurements, but only 2 test pairs under the fixed current-state match. A
larger hand threshold would be a semantic change and still leaves train/val
nearly empty.

Root action: run a separate bounded Probe with three waves per split (seeds
296/297/298), the same route, motions, residual amplitude, 24-step plan and
2 cm hand RMS/object-state thresholds. This adds repeated current states while
preserving the old 144-episode artifact. Do not merge the new sources into
training unless the labeler produces nonempty train and val pairs under the
unchanged rule.

Cost and stop: at most 72 episodes per split, one GPU at a time, 900 s and
2 GiB per collector, 600 s CPU labeling, and the existing 2 GiB/20 GiB output
caps. Stop without fitting if either train or val remains empty, if source
provenance drifts, or if geometry diagnostics become invalid. A positive
coverage result only permits a fresh fit decision; it is not an oracle-headroom
claim.

Implementation checkpoint: queue smoke sizing now accounts for hard-object
oversampling and minibatch divisibility (`3b4305b`, `e62dd08`).

## Repeated-wave coverage result (2026-10-08)

The follow-up card `P-20261008-consequence-pair-coverage` completed its three
bounded collectors (seeds296/297/298, 72 episodes per split). The unchanged
label rule now produces `READY` with train/val/test = 2/1/5, and the sampled
geometry audit remains corroborated (149,841 valid frames; one proxy-far
frame). This resolves the original empty train/val gate as a data-contract
check, but not as useful evaluator supervision.

The pair count is still too small for the declared 32-pair/update fit: every
training update would resample two comparisons and validation would select on
one. Root therefore stops before `prepare_windows.py`/`train_matched.py`, keeps
the 216 new raw episodes and label output, and records the result as UNCLEAR.
No E0/Eoracle ranking or world-model conclusion is drawn. Future work needs
explicit twin current-state branches rather than more generic waves or a
post-hoc hand RMS relaxation.
