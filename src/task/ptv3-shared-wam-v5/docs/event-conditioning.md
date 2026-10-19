# WAM-v5 next CONTACT/RELEASE condition — proposed contract

Source: the user's [shared discussion](https://chatgpt.com/share/6ad61d8a-d46c-83ea-b4e6-9977926fc3c2),
read from the newest substantive exchanges backward; user confirmed that
the requested WAM addition is event type plus relative frames. The recent
agreement is `(H4, object goal, next event) -> tau`, with CONTACT/RELEASE
event planning supplied by the scheduler. Force/state expansion discussed
later belongs to executor sensing and is outside this WAM change.

## Why this fits the current model

An almost-static object goal can accompany approach, hold or release.
Object geometry/goal alone does not identify the intended hand transition.
Use one next-event slot: type NONE/CONTACT/RELEASE, delta frames on the30Hz
WAM timeline, and event_valid. Unknown annotation must remain distinguishable
from a known absence of an event. A late CONTACT plan can remain CONTACT at
delta0 until the scheduler confirms acquisition; switching phases solely
because the nominal deadline passed would hide failed execution.

Keep the768 backbone,21-point K24 outputs,778 hand samples,2048 object/C
queries and TRAIN-only horizon statistics. A small event encoder can add a
zero-initialized residual to the existing object-goal tokens. This adds no
point search or KNN and does not lengthen the shared token sequence. Verify
old predictions at migration before any event-conditioned update.

The optional model interface is implemented in `events.py` and enabled only
with event_conditioning=true plus event_encoding=
`next-contact-release-plan.frame30.v1`. Inputs are per-sample event_type
(0 NONE/1 CONTACT/2 RELEASE), event_delta_steps, event_valid and event_hz30.
It encodes the type, relative time divided by24 (capped at240 frames), and
an out-of-horizon flag. A far event retains its type; invalid padded fields
are cleaned and gated out. A one-sample condition cannot broadcast across
an unrelated batch. Production configs still leave this branch disabled.

Four tests exercise the actual WAM forward: exact old trajectory/scene/C and
shared-token equality after zero-residual weights-only expansion, a nonzero
event-branch gradient and subsequent event sensitivity, unknown/far event
handling, and rejection of incomplete/wrong-rate/broadcast input contracts.

This requires an explicit new input/config identity. Model weights can be
reused; optimizer migration must retain moments for old parameters by name
and initialize the additional parameters separately. Shape-compatible weight
loading alone is not a certified full optimizer resume. Do not bypass the
new effective-batch/data/config checks with a generic non-strict load.

## Data and deployment boundary

Large-scale MANO/robot demonstrations mostly lack an external planned event
table. Existing proximity caches can supply a compact candidate event
sidecar, without rebuilding surface/contact geometry. However, a transition
in nearest-surface proximity is an annotated proximity event, not measured
force contact or proof of stable grasp. Thresholds, hysteresis/persistence,
invalid gaps, left/right censoring, dual-hand contact and multi-object slots
must have a recorded contract and a small visual audit before freezing it.

Training may treat demonstration-derived events as annotations of that
demonstration's task intent. This is explicitly event-conditioned imitation,
not evidence of predicting future contact without a supplied task. During
online execution, get the event from the task plan/scheduler, not from the
future outcome of that very rollout. Keep the current explicit object-goal
conditioning as well. Raw task-control frames must be converted to30Hz WAM
time before computing delta; Shadow/Inspire native rates can differ.

Do not substitute NONE for truncated or invalid future segments. Annotation
coverage, censored intervals and availability beyond K24 must be explicit.
Drop some event conditions during training to retain the unknown-event path;
the event-conditioned branch should receive a real gradient after migration.

## Cheapest useful next decision

First annotate and visualize a bounded source-diverse panel with genuine
contact/release transitions. Freeze its semantics, then test the zero-residual
migration and short continuation on the same starting checkpoint. Compare
event-conditioned and event-dropped predictions, especially before/around
CONTACT and RELEASE; always include root-relative, tips and short horizons.
Do not claim success from total loss alone or from a wrong-event-only test.

The engineering-resume revision is independent. Event-conditioned production
training is not yet activated; no future-event payload has been inserted into
the currently frozen training data or model.

## Preparation implementation

`event_labels.py` and `tools/run/prepare_events.py` produce immutable candidate
sidecars from each frozen row's own contact_path. No new contact distances,
geometry, normalization or window/sampling changes are introduced. At least3
queries above.8 establish proximity; fewer than3 above.5 establish clearance.
Confirm3 valid observations, keep hysteresis-band state, reset on missing
observations, and do not fabricate an onset at the recording's left boundary.
Known NONE requires a fully observed K24 plus confirmation margin. Far events
keep their actual time; censored tails/gaps remain unknown.

HOI4D uses its real even-frame observations (3 observations span5 WAM frames).
Between two adjacent known real observations, the event table may be queried
at the synthetic odd-frame time: use the right packet with delta+1 for a real
future event, or NONE with delta0. Do not fabricate a contact measurement or
state there. A missing real observation, unknown state or censored right packet
prevents that query. Event deltas remain30Hz. This matters because inherited
HOI4D windows are anchored on odd ticks; first-pass frame coverage alone hid
that all sampled HOI4D event conditions were unknown. The v2 annotation schema
records this explicit time-query contract; v1 remains diagnostic evidence only.
Candidate manifests have training_allowed=false pending annotation inspection.
The source-diverse panel, full census and limitations are tracked in the
[event preparation card](experiments/probes/P-20261019-wam-v5-event-preparation.md).

`EventV5Windows` is a separate probe reader that requires a complete sidecar
bound to its parent. Its15% TRAIN condition dropout uses an independent
seed/step/rank stream; original source, stratum and window choices are retained,
and spawn workers reopen bounded mmap caches. Direct DEV indexing has no
dropout. This reader is not silently substituted into production training.

`event_migration.py` maps original AdamW group positions through parameter
names, preserves every original state/moment/step and group setting, and
leaves only event_goal parameters fresh. Tests compare the next original
parameter update exactly. `audit_event_migration.py` verifies this against a
real checkpoint and can save a separate candidate payload. Its candidate
schema deliberately cannot pass generic strict resume: the new event recipe
identity and an explicit migration entry still need binding before launch.
Conditioned continuation and event-effect validation remain outstanding.
