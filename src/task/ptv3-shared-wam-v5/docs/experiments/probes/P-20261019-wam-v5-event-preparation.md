---
schema: ref2dex.probe.v2
probe_id: P-20261019-wam-v5-event-preparation
experiment_id: P-20261019-wam-v5-event-preparation
date: 2026-10-19
task: ptv3-shared-wam-v5
branch: main
git_commit: e8d995c9
claim_id: C1
hypothesis_family: HF-v5-planned-proximity-events
probe_index_in_family: 1
seed_pool: probe
seeds: [177]
decision_changed_if_positive: prepare complete candidate event sidecars and an explicit checkpoint expansion before a conditioned continuation probe
decision_changed_if_negative: revise annotations before any event-conditioned optimization
status: COMPLETED
run_id: events-candidate-20261019-r2
---

# Planned CONTACT/RELEASE data preparation

Result: v2 candidate39526 sequences complete;32768 inherited draws98.7% annotated; real step10049 weights/564 optimizer states preserved, nine-source tau/C expansion deltas0.
Decision: keep current production training unchanged; proceed next to an explicitly bound short event-conditioned continuation, not a new long run yet.

Classification: Decision. The user confirmed the new WAM input is the next
CONTACT/RELEASE type and relative30Hz frames, then asked to prepare it. Test
whether existing frozen proximity caches yield sustained, source-diverse
transitions with explicit censored/unknown coverage. Cheapest discriminator:
16 fixed random sequences/source, persistence2/3/4 sensitivity and timelines;
no geometry reconstruction, KNN or optimizer update. Seed177 is reused for
exploratory diagnostic selection, not an independent replicate. Current
production training continues with its frozen ac951370 code and original data.

## Candidate protocol

Work version `v5.proximity-event-candidate.v2`, schema
`ref2dex.wam-v5-proximity-events.v2`. Object-level any-hand union follows the
existing2048-query contact teacher. Enter if at least3 queries have c>=.8;
leave if fewer than3 have c>=.5; retain state in the hysteresis band. Require3
consecutive valid observations and backdate to that run's onset. Initial
state acquisition creates no onset event; gaps reset state. Given the teacher
formula, thresholds correspond approximately to5.23mm and9.93mm, not force.

HOI4D/source10 is explicitly sampled at its observed even frames (stride2),
because odd frames are synthetic and masked by the teacher. Three observations
span5 WAM frames there, versus3 elsewhere. Missing even observations still
break continuity. All stored event times/deltas remain on the30Hz timeline.
Between adjacent known real frames, a synthetic midpoint may query the right
confirmed next-event packet (real event delta+1, NONE delta0); physical C/state
stay masked. Unknown/censored neighbours prevent the query. This is a task
annotation time query, not interpolated contact truth.

Keep next confirmed events beyondK24 with true delta. Known NONE means no
event inK24 and requires K24 plus persistence confirmation margin to be
observed; truncated tails/gaps are unknown. The event at the current frame
can have delta0. These offline demonstration task annotations are an input
condition for imitation; deployment must supply the scheduler's planned event,
never extract it from the future realized rollout. No stable-grasp, per-hand
handover or force-ground-truth claim is made.

## Execution boundary

Remote root `/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v5-20261019-r1`.
Parent `outputs/finalize-training-20261019-r1/recipe/manifest.json` binds all
39526 sequences, original TRAIN/DEV windows, source probabilities, motion
strata and unchanged horizon stats. Sidecars read each sequence's own bound
contact_path, including repaired Shadow contacts. The old unsafe caches are
not selected by a global path assumption.

Outputs under `outputs/event-panel-20261019-r1` and, if useful,
`data/events-candidate-20261019-r2` (r1 preserved as diagnostic). Per-sequence atomic publication, no
overwrite, <=8 pure CPU spawn workers, <=7200s, <4GiB output, free disk>=20GiB,
STOP/deadline checks. Pure cached reductions do not need a GPU. Any neural
checkpoint inference uses a GPU with checked headroom. Record Python/NumPy,
git commit, parent metadata identity, protocol and source coverage in manifests.

Candidate manifests deliberately have training_allowed=false. Full candidate
preparation is not approval of label quality or activation of a new long run.
Next: inspect timelines, certify named optimizer expansion and exact initial
predictions, then decide a short matched conditioned continuation. Original
K24 normalization,768 backbone and21-joint tau remain unchanged.

## Results

The fixed144-sequence panel completed in7.42s on remote Python3.8.20/NumPy
with8 CPU workers, using f68057ee. All9 sources have CONTACT onsets. Sampled
Shadow has17 CONTACT/1 RELEASE; sampled DexYCB16 CONTACT/0 RELEASE, so source
imbalance and recording boundaries require reporting rather than synthetic
release labels. HOI4D yields35 CONTACT/33 RELEASE using observed even frames.
The first full39526-sequence candidate census completed in211.67s using
f68057ee, output `data/events-candidate-20261019-r1`, PID3058882. However,
b3e3bd82's32768 inherited-mixture TRAIN draw audit found all1803 HOI4D samples
unknown because all inherited anchors are synthetic odd ticks. The original
source-level raw-frame coverage therefore did not certify usable conditions.
The v2 protocol above fixes only annotation-time querying; no physical C,
windows, source probabilities, target data or horizon stats change. Rebuild
to a new directory and rerun the actual sampler/loader check before admission.

56664364's real checkpoint expansion (step10049, same remote Torch2.0.1+cu118)
preserved all564 original parameter optimizer states and steps10049 exactly;
added4 tensors/50304 parameters with fresh optimizer state. All9 source
singleton tau/C comparisons and baseline repeats were exactly equal, peak
GPU1 allocation1.14GiB,40.997s including load/save. Candidate checkpoint
`outputs/event-migration-20261019-r1/event-expanded-candidate.pt` is1185966912
bytes; its distinct schema/false admission prevents accidental generic resume.
This is migration evidence, not event-conditioned training/effectiveness.

No training model/data configuration has changed. No event-conditioning gain claimed.

## Completed v2 preparation

Frozen e8d995c9 in `code-events-e8d995c9/`, remote graspenv Python3.8.20,
Torch2.0.1+cu118,8 pure CPU spawn workers. `data/events-candidate-20261019-r2`
completed all39526 sequences in193.41s. The full pipeline (census,144-sequence
panel, actual reader/loader audit, geometry export) completed in231.68s,
supervisorPID3060563. No preparation process remains to restart.

The same58444 CONTACT/26929 RELEASE onsets as v1 are retained; only source10
midpoint event-time queries change. In32768 fixed inherited-mixture TRAIN
draws (seed177,step10050,rank0),32345 have valid annotations (98.7091%); after
independent15% condition dropout,27411 remain (83.6517%). HOI4D improves from
0/1803 to1776/1803 valid (98.5025%), with355 CONTACT/1288 RELEASE/133 NONE and
27 unknown. These are sampled optimizer exposure estimates, not uniform-frame
coverage or model accuracy. Source8 still has no inherited DEV and sampled
release exposure0; do not claim learned release performance there.

Actual EventV5Windows reads preserve every original input/target field on a
nine-source panel exactly;32768 source/stratum/window choices match the base
sampler exactly. Two spawn workers materialize a real batch128 with106 valid
conditions after dropout. TRAIN3218740/DEV545459 and all original stats remain
unchanged. `outputs/event-dataset-audit-20261019-r2/candidate-recipe.json` is
reviewable but training_allowed=false; stats.json is an immutable parent link.
The production launcher still cannot silently activate this recipe/model.

Inspection artifacts: `outputs/event-panel-export-20261019-r2/` contains actual
object queries, hand surface samples/joints at onset±6frames and timelines.
Local receipts/PNGs are under
`outputs/ptv3-shared-wam-v5/event-preparation-20261019-r2/`. Three geometric
examples (OakInk2 CONTACT, Shadow CONTACT, HOI4D RELEASE) expose actual
surfaces and threshold counts. OakInk2/Shadow approach strongly increases
c>=.8 queries (0→10→158 and0→20→162 over onset±6frames). The HOI4D RELEASE
example is marginal: c>=.5 query count3→0→3, with no c>=.8 queries at those
three snapshots. Do not call this a certified intentional release; compare
event stability/conditioning effects before upgrading proximity to task phase.
Timelines show residual
short transitions in some ARCTIC/HOT3D examples; persistence2/3/4 sensitivity
does not certify force contact or every annotation. This is sufficient to
prepare a short conditioning decision probe, not to declare label perfection.

44 v5 tests pass. Scoped Python/structured/seed/link checks pass; the full
repository verification gate remains blocked only by six existing unrelated
trajectory-policy cards missing v2 metadata. No unrelated cards were edited.
Current four-GPU production run remains ac951370 and unconditioned. New long
training, new strict event recipe migration entry, matched continuation and
event-effect validation remain separate next work.

Final live check: production RUNNING at12400,4 GPUs100%,1.118updates/s,
remaining60440s (about16.8h). Preparation did not pause or replace that run.
