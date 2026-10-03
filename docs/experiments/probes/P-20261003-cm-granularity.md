# Fixed-data query and neighborhood granularity Decision Probe

User: "那你试试看", approving the proposed fixed-data comparison. Bounded
research Probe in the existing isolated surface-prior branch, same MISSION.
Question: does sparse object context or neighbor averaging limit prediction?
Decision: a >=10% same-hand held-parent gain prioritizes that representation
for causal execution-input qualification; absent useful gains stop this exact
granularity matrix and return to the command-to-motion blocker. Cheapest
discriminating test is an offline2x2factorial, without new physics collection.

## Prospective design

For each MANO/Inspire source independently:64/256object context points crossed
with mean/detail4neighbor inputs, eight fits total. Train on the SAME first
2048windows of the completed scale Probe's immutable ordered selections.
Retain ALL original held-parent windows (MANO test, Inspire val), no new split,
no outcome filtering or target-based point sampling. Original64object IDs
are first in each256point packet; add192uniform points from the remaining
4032using seed3902. All inputs/outputs preserve original source correspondences.

Mean repeats each neighbor group's mean four times; detail keeps all four
current nearest-neighbor distance-rank slots. Common49->128->64point encoder,
global64mean Cm,128->64->3decoder, same23107parameters for ALLarms. This is
a new matched baseline, not a direct reuse of the old22feature19651parameter
model. Geometry, normalization, global hand mean flow and previous object flow
are identical. The original2048/10135point hands are not resampled or aligned
by joint index. Larger context supplies more spatial observations only.

Train and score ONLY the SAME original64object target points, including in
256point arms. No extra supervised dense targets. Each arm1500AdamW updates,
batch32windows, lr3e-4, weight decay1e-4, clip10, output zero initialization.
Common initialization3903, exact common batch schedule3904, no dropout,
checkpoint selection, adaptation, capacity/seed/threshold rescans. Equal data,
parameters, updates and supervised targets;256context is4xpoint computation,
so FLOPs and wall time are NOTmatched and are explicitly reported.

Primary metric equal-weight mean per-parent64point flow EPE/mm. For eachhand,
each of256mean/64detail/256detail versus64mean has a prospective10%gain gate.
PROMISING if any gate passes (identify which hand/variant, no universal claim),
UNCLEAR if none pass but any gain>=5%, otherwise UNPROMISING. Report all six
gains, all eight models on both held domains, zero/persistence baselines,
per-parent errors, training costs, and factorial contrasts descriptively.
Single initialization and correlated windows are Probe evidence only;
multiple comparisons are not a formal statistical test or future method choice
without independent validation. No favorable subset rescues a failed gate.

## Scope and integrity

Hand flow is realized t->t+1 motion: still OFFLINE, unavailable at native policy
decision. Future object target is isolated; no policy or pure morphology-cause
claim. Legacy Inspire cache physics is explicit. Unsigned2cm masks do not
establish physical contact, force closure, fingertip/patch coverage or absence
of penetration. Changed pooling/input slots test this bounded representation,
not every possible fine model or a converged scaling law.

One freshly idle GPU for all main fitting/prediction; CPU file extraction and
independent NumPy/geometry audits (no main CPU model substitution). Budget
1800s/2GiB new evidence, fixed commit before run, unique run_id. Guard input
SHA/source metadata and GPU contention; stop only owned process groups. Retain
all packet rows/selection, models, optimizer/schedules/losses, predictions and
audit. All selected raw rows must reconstruct from source exactly. Verify
old64packet inheritance, split/schedule/init/optimizer counters/changedweights,
independent float64features/NumPyforwards, future-object isolation, SE3-frame
invariance and distance-rank slots. Numeric bounds features atol2e-5/rtol2e-6,
network<2e-4normalizedflow, metrics<1e-4mm, nearest distances<1e-5m.
No complete optimizer-trajectory replay claimed. Stop after this fixed matrix.
