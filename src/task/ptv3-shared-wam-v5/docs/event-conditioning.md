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
the currently frozen training data or model. The code supports the optional
interface, but data sidecars, named optimizer expansion, conditioned continuation
and effect validation remain before an event-conditioned training run.
