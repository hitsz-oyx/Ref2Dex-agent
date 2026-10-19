---
schema: ref2dex.probe.v2
probe_id: P-20261019-v5-contact-definition-audit
experiment_id: P-20261019-v5-contact-definition-audit
date: 2026-10-19
task: ptv3-shared-wam-v5
branch: main
git_commit: e3f7db18
claim_id: C1
hypothesis_family: HF-v5-contact-definition
probe_index_in_family: 1
seed_pool: probe
seeds: [177]
decision_changed_if_positive: inspect balanced BCE for scarce positives or normal-aware teacher for localized backside errors before v5 training
decision_changed_if_negative: retain the current distance-only teacher and unweighted contact BCE
status: RUNNING
run_id: contact-audit-20261019-r1
---

# v5 contact prevalence and normal-direction audit

Classification: Decision. [User specification](../../user/ref1.md) restricts
this change to data and C, before any executor/PPO integration or CEDex grasp
admission. Shared semantics remain palm plus3 segments per finger. MANO uses
its actual778 vertices; Shadow uses its actual collision/FK surface samples.
No robot is fitted into a fictional MANO hand.

Questions: are source-specific valid contact labels so sparse that ordinary BCE
needs balancing? Does an object-normal backside penalty change a meaningful
number of close queries, especially on thin/opposing surfaces? If not, keep the
current teacher/loss. Cheapest discriminator: a streaming cache census plus at
most324 TRAIN frame/object pairs, without any neural training or new datasets.

## Frozen protocol

Work version `v5.contact-prevalence-normal-candidate.v1`.
The remote root is
`/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v5-20261019-r1/`.
Inputs are the existing `data/plan/plan.json`, `data/hands/`, `data/contacts/`
and `data/manifest.json` (geometry assets under `data/geometry/`). No held-out TEST is read.

Prevalence: stream all completed sequence caches in bounded16-frame chunks,
exclude invalid frame/object labels, accumulate counts and soft confidence by
source for thresholds0/.25/.5. Also count repeated TRAIN/DEV future-target
exposure from each admitted window's tick+1 through tick+24. Report coverage
against all39526 planned sequences. Available-cache results are explicitly
partial until complete. Window rates are uniform within source, not an exact
source-probability/motion-stratum-weighted optimizer distribution. Very small
rates motivate a later matched loss probe; no automatic balancing is applied.

Normal panel: seed177, at most8 available TRAIN sequences/source, quota36
frame/object pairs/source. Half uniform valid targets, half contact-enriched
(any c>.25); fill without replacement. The fixed panel contains no DEV-only
target frames and is not used to estimate population prevalence. Source/cache
availability and exact frame IDs/selection are saved.

Compare the current nearest-vertex teacher with the same nearest vertex/part
using outward-normal candidate:

`s=(nearest_hand_vertex-object_query) dot unit_object_normal`

`d_eff=d+4*max(-s-0.001m,0)`

Both use `c=exp(-max(d-0.003m,0)/0.010m)`, zero beyond0.033m, with d_eff
substituted for the candidate. This is a specified diagnostic, **not** an exact
CEDex capsule formula. The1mm penetration tolerance and penalty4 are audit
parameters, not fitted hyperparameters. Invalid normals are excluded from
comparisons and leave baseline confidence intact. Rotate normals with the
object; translate only points. Recomputed baseline must match every sampled
cached label exactly before interpreting differences.

Report per-source mean suppression, fraction with suppression>.1, positive
queries crossing.25, backside counts, and changes on a nearby opposing-normal
surface proxy (sample separation<10mm, dot normals<-.5, residual at least50%
along normal,64 nearest object samples). The proxy is not proof of thinness or
concavity. Save six source-distinct examples with strongest changes as3D PNG
and NPZ plus explicit identities for user review.

## Decision limits

Neither normal winding nor watertightness is certified for all CAD assets.
Normal sign is not inferred/flipped using hand labels. A large candidate delta
alone does not establish correctness: inspect object normals and examples,
including penetration/noise, before changing any labels. A nearly unchanged
field supports retaining the simpler teacher for now. Continuous
triangle/primitive distances could unify teacher approximation quality later;
they are not silently substituted into this comparison. Increasing2048 sample
count alone does not imply a lower positive fraction than128.

Loss stays ordinary masked BCE and positive-contact part CE during this audit.
C stays confidence plus16 parts; tau stays21 points/K24 with TRAIN-only horizon
normalization. No KNN enters the model or training reader. No physical-contact,
cross-morphology invariance, WAM quality or PPO gain is claimed by this probe.

## Execution budget and artifacts

Remote `graspenv`, Python3.8.20; CPU1 thread per audit (array reductions/KD-tree
geometry only, no repeated neural inference). Existing GPU jobs stay running.
Normal audit<=900s/256MiB; prevalence watcher<=10800s/128MiB; both stop for
STOP/deadline/free NAS<20GiB. Outputs stay in separate run directories under
`outputs/contact-audit-20261019-r1/` of the remote root; no label overwrite.
Per-run manifests record code commit, runtime versions, PID and actual inputs.

## Upstream correctness repair prompted by ref1_1

The [additional audit](../../user/ref1_1.md) identified candidate blockers;
the actual remote environment confirms Trimesh4.12.2 capsules already have
Z bounds[-28,+28]mm for radius8mm/half-length20mm. The old extra translation
moves these to[-48,+8]mm. Both source4 input surfaces and downstream C are
affected despite aligned joints. Old contact generation stopped at4547
sequences; the old finalizer and prevalence watcher stopped gracefully.
Source4 must be rebuilt into fresh `data/hands-r2`/`data/contacts-r2` before
interpreting its normal audit. Human geometry/labels are explicitly inherited,
not universally regenerated. No model training is launched by this repair.

The initial normal attempt failed before comparison because --geometry pointed
at the assets subdirectory instead of the registry parent. Corrected input
run `normals-r2` finished324 samples in16.38s including figures, baseline
regeneration matching cached labels. On this enriched panel, suppression>.1
affected3.29–13.43% of queries by source; this is not negligible, but neither
outward winding nor correctness is proved. Source4 numbers are invalid for
the repaired teacher. Retain examples as diagnostics and rerun on repaired
inputs. Do not automatically enable normal penalties or BCE balancing.

Engineering fixes also align H4 admission with the reader; bind immutable
cache runs to protocol/upstream dependencies; reject PARTIAL CAD; expose same-ID
PTv3 features to C; add fine-grained DEV metrics; implement checkpoint retention
and the hand-in-PTv3 switch. Array digests are computed at new-cache publication,
not every training batch; metadata/contract checks do not claim detection of
manual payload edits in an otherwise immutable run. Old human inheritance has
explicit scope. Global-only C architecture/config cannot resume as local-ID C.

Result: remote capsule blocker confirmed and repair required; preliminary normal differences warrant review, not immediate teacher changes.

Decision: repair Shadow geometry, refreeze with exact H4 admission, rerun both audits; keep distance-only teacher/BCE until corrected evidence is available.

Verification limitation: repository-wide `tools/verify.py --changed` also
validates historical cards and currently fails on missing v2 metadata in six
pre-existing trajectory-policy cards. New v5 unit tests and diff checks run
separately; unrelated historical cards are not rewritten here.
