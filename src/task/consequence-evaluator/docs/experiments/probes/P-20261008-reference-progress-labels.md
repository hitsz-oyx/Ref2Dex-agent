---
schema: ref2dex.probe.v2
probe_id: P-20261008-reference-progress-labels
experiment_id: P-20261008-reference-progress-labels
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: e99d02e
claim_id: C3
hypothesis_family: HF-consequence-reference-progress
probe_index_in_family: 1
seed_pool: probe
seeds: [261]
decision_changed_if_positive: retain causal delta-progress supervision and prepare an independent scalar evaluator Probe
decision_changed_if_negative: diagnose reference alignment before fitting rather than extending old S/P/M supervision
status: RUNNING
run_id: reference-progress-labels-20261008-r1
---

# Can original-reference temporal progress label executed candidates causally?

Result: Implementation and six engineering tests pass; real reference/rollout
alignment not yet evaluated. No evaluator training.
Decision: Follow user ref4_1; label validation precedes model fitting.

## Decision Note and purpose

This Decision Probe serves C3/Cm policy utility by checking whether a short
physical consequence can carry reference-conditioned task advancement. It
does not demonstrate policy utility or prove a same-state candidate ranking.
Hypothesis: geometry with causal temporal context and a bounded historical
phase prior can distinguish progress, stagnation and regression, including
the repeated object state at approach and normal placing.

The user explicitly replaced old airplane-specific S/P/M supervision with
`Y_t=P_(t+24)-P_t` and chose **original successful motion**, reconstructed with
the robot URDF, as R. Preserve all old trajectories/labels/cards; do not fit
an S head or redefine the Mission claim. Successful-reference provenance is
the user's motion specification, not a binary outcome fitted from rollouts.

The cheapest discriminating experiment is eight preserved train episodes:
four clean and four fixed-dose contact interventions, selected by recorded
assignment/order, not outcomes. No fresh rollout, policy training, held-out
inspection or reference selection from test. Proceed to evaluator design only
after nominal/reference and observed failures' traces support useful labels;
otherwise fix alignment or record insufficient geometric phase information.

## Fixed implementation

Original reference: complete s3_airplane_lift, 543 frames at30Hz, retargeted
robot q373:391 plus object position198:201/quaternion201:205. Reuse native
URDF FK with explicitly named Gym joint order and identity actor root; wrist
world pose is carried by q[:6]. Preserve original q without PD conversion or
human keypoint substitution. Audit12assigned-clean train state frames against measured11
native link positions; coordinate error must be<=10micrometres. Also check
initial actual/reference object pose matches and freeze input hashes.
Separately retain the first assigned-perturbed trajectory's full FK-versus-
measured discrepancy. FK mapping agreement in clean states does not guarantee
PhysX constraint agreement in a violently perturbed articulation; all actual
label features always use independently measured hand points.

Common90D features: initial-object-registered position and rotation matrix,
11 current-object-frame hand points, object translation/rotation backward
motion and backward hand/object relative motion. Six equally weighted groups,
reference-only scales with fixed physical floors. No frame clock/phase/outcome
input; actual registration uses only frame zero. The full reference is a fixed
available condition, not an actual-episode future.

Reuse pinned Google XIRL helpers for squared L2 and dimension scaling; adapt
its explicitly causal trailing-context sampling to eight frames. First-stage
soft match is combined with an unbiased stay/forward/backward historical
transition, offsets[-4,4], exp(-abs(offset)/1.5). Hard posterior support within
four frames of the previous soft index prevents rare distant tails from
creating a far expectation jump. Temperature0.01; P=q·index/(N-1). Start at
reference0 because these raw episodes start at reference0. No episode clock,
monotonic clamp, reverse cycle, OT smoothing or episode-wide normalization.
[Source/code boundaries](../../research/REFERENCE_PROGRESS_SOURCES.md).

Known24requested residuals retain frozen-controller closed-loop execution.
Each window gets exactly P[t+24]-P[t]. Difference deadzoneepsilon0.01;
ties abstain. No cross-state preferences are automatically fabricated.
The signed-value helper requires an independent current-context check before
candidate comparisons. Labels contain no inherited success/stage/margin.

## Controls, resources and stops

One idle GPU1 for batched untrained geometry distances, <=180s/eight episodes,
<=0.5GiB outputs. Reference reconstruction uses CPU: small batched URDF
geometry/file audit without a neural model or simulation. All artifacts under
outputs/consequence-evaluator; scratch/cache under repository tmp. Commit code
before execution, freeze actual code identity, preserve failed runs. Stop on
source drift, nonfinite data, occupied GPU, deadline or causal/bound violation.

Compare reference against itself; a static initial clip must not advance with
elapsed time. Recompute actual prefixes truncated at candidate t+24; posterior
must match the full run's prefix within1e-9, with |deltaP|<=4/(N-1). Plot actual
object heights and P for manual/physical interpretation, retaining uncertainty
and fit costs. Reference self/index-clock is **only a diagnostic target**; no
t/T actual labels. Default matched-cost<=1 is an engineering diagnostic, not a
calibrated confidence or validated success gate.

Probe source: outputs/consequence-evaluator/official-value-dose-train-20261008-r1
(seed261, frozen original collection98ceb76). Prepared reference expected at
outputs/consequence-evaluator/reference-progress-original-20261008-r1.
Output labels: outputs/consequence-evaluator/reference-progress-labels-20261008-r1.
`training_allowed=false` throughout this label validation. Existing trainer
schemas remain unchanged and must not accept these labels by accident.

## Limitations / future evidence

Only one object/reference and one Probe group. Original robot-retargeted motion
can differ from the physical expert's repeated lift/place execution; useful
geometric alignment is an empirical question. Flat/repeated clips may remain
unidentifiable, and distant failures can produce uncertain progression rather
than a meaningful negative. Feature floors/temperature/transition/deadzone are
exploratory defaults, not tuned on held-out data. No multi-seed scientific
conclusion, same-state action intervention comparison, evaluator headroom,
learned temporal embedding or online Cm-on/off evidence yet.
