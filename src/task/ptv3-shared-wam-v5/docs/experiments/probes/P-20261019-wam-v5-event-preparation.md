---
schema: ref2dex.probe.v2
probe_id: P-20261019-wam-v5-event-preparation
experiment_id: P-20261019-wam-v5-event-preparation
date: 2026-10-19
task: ptv3-shared-wam-v5
branch: main
git_commit: pending
claim_id: C1
hypothesis_family: HF-v5-planned-proximity-events
probe_index_in_family: 1
seed_pool: probe
seeds: [177]
decision_changed_if_positive: prepare complete candidate event sidecars and an explicit checkpoint expansion before a conditioned continuation probe
decision_changed_if_negative: revise annotations before any event-conditioned optimization
status: PLANNED
run_id: event-panel-20261019-r1
---

# Planned CONTACT/RELEASE data preparation

Classification: Decision. The user confirmed the new WAM input is the next
CONTACT/RELEASE type and relative30Hz frames, then asked to prepare it. Test
whether existing frozen proximity caches yield sustained, source-diverse
transitions with explicit censored/unknown coverage. Cheapest discriminator:
16 fixed random sequences/source, persistence2/3/4 sensitivity and timelines;
no geometry reconstruction, KNN or optimizer update. Seed177 is reused for
exploratory diagnostic selection, not an independent replicate. Current
production training continues with its frozen ac951370 code and original data.

## Candidate protocol

Work version `v5.proximity-event-candidate.v1`, schema
`ref2dex.wam-v5-proximity-events.v1`. Object-level any-hand union follows the
existing2048-query contact teacher. Enter if at least3 queries have c>=.8;
leave if fewer than3 have c>=.5; retain state in the hysteresis band. Require3
consecutive valid observations and backdate to that run's onset. Initial
state acquisition creates no onset event; gaps reset state. Given the teacher
formula, thresholds correspond approximately to5.23mm and9.93mm, not force.

HOI4D/source10 is explicitly sampled at its observed even frames (stride2),
because odd frames are synthetic and masked by the teacher. Three observations
span5 WAM frames there, versus3 elsewhere. Missing even observations still
break continuity. All stored event times/deltas remain on the30Hz timeline;
odd-frame conditions remain unknown rather than fabricated contact labels.

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
`data/events-candidate-20261019-r1`. Per-sequence atomic publication, no
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

Pending panel and complete preparation. No event-conditioning gain claimed.
