---
schema: ref2dex.probe.v2
probe_id: P-20261008-reference-progress-labels
experiment_id: P-20261008-reference-progress-labels
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: 43e7bc0
claim_id: C3
hypothesis_family: HF-consequence-reference-progress
probe_index_in_family: 1
seed_pool: probe
seeds: [261]
decision_changed_if_positive: retain causal delta-progress supervision and prepare an independent scalar evaluator Probe
decision_changed_if_negative: diagnose reference alignment before fitting rather than extending old S/P/M supervision
status: UNCLEAR
run_id: reference-progress-labels-20261008-r2
---

# Can original-reference temporal progress label executed candidates causally?

Result: Causal geometry labels pass self/static/prefix checks, but corrected
eight-episode replay still misaligns all four nominal trajectories. No fit.
Decision: Preserve both label runs; test learned TCC alignment before Gate1.

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
transition, offsets[-4,4], exp(-abs(offset)/1.5). The original r1 hard support
around the soft index trapped matching in an early local minimum. The corrected
filter preserves reachable alternatives in log space and projects posterior
mean steps to<=4frames by exponential tilting (minimum KL change), without
retroactive smoothing. Temperature0.01; P=q·index/(N-1). Start at
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
Corrected replay uses reference-progress-labels-20261008-r2 with the same
eight assigned train episodes and unchanged reference/features/temperature.
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

## Initial execution and implementation diagnosis

Reference reconstruction at96366c6 completed543frames in0.245s; the12assigned-
clean FK comparisons have max coordinate discrepancy2.228micrometres. The
first assigned-contact trajectory's full FK diagnostic reaches25.509mm over
159frames above10micrometres. The measured actual points are retained; these
states do not justify loosening the clean mapping gate or replacing actual
measurements with ideal FK. A prior preflight spelling mismatch for the owned
motion symlink was fixed by requiring canonical identity **and** frozen hash.

R1at96366c6 completed8episodes/496known-plan windows in7.738s, GPU1peak
allocation24.18MB/reservation44.04MB, allowned processes exited. Reference self
ends0.999845with indexMAE0.000073; stationary initial ends0.000160. Truncated
actual prefixes match exactly, and bounded progress checks pass. Nevertheless,
all4clean episodes end near0.032; only3.1%of their windows pass even the loose
fit-cost diagnostic. This is **not usable supervision**.

Single clean replay distinguishes geometric data absence from tracker trapping:
global8frame matches reach late reference phases, reference-frame hand/relative-
motion median standardized distances are0.024/0.029, and global best clip cost
median0.259. The filter loses alternative phase paths at early approach. Raising
only temperature0.01→0.03/0.1/0.3/1does not recover nominal progress and degrades
the original filter's self-control. Preserve this diagnostic, do not select a
favorable temperature or fit the failed labels.

Correct the identified irreversible-support-pruning defect, keeping the same
transition bound. Log-domain filtering preserves small alternative phase
probabilities; exponential tilting bounds the posterior mean without deleting
paths. A late-evidence regression test now requires reachable-phase recovery
as well as the step bound. Same clean trajectory ends0.319at unchanged0.01;
self/stationary remain0.999845/0.000161. This fixes a real implementation defect
but does not yet validate full-reference geometry progress. Further temperature
checks on this one train example reach at most0.516and do not justify a change
to the fixed replay or held-out evaluation. Diagnostic artifacts are in
outputs/consequence-evaluator/reference-progress-diagnostic-20261008-r1.

## Corrected replay result and next decision

R2at43e7bc0 completed8episodes/496windows in9.799s, with the same GPU1peak
allocation24.18MB/reservation44.04MB and exact prefix invariance. All owned GPU
processes exited. Clean finals are0.319/0.417/0.262/0.437, valid-cost window
fractions86.2/86.2/78.5/70.8%, and backwards totals0.790/0.774/0.828/0.871.
Thus numerical/causal correctness improved, but nominal phase semantics remain
**UNCLEAR** and training remains disallowed. Four contact intervention endpoints
have invalid fit confidence; a failure's positive delta cannot be treated as
validated task advancement. Neither Gate1 nor evaluator fitting was run.

Bounded one-example diagnostics changing only context16/32yield finalP0.344/
0.486. Larger max step8/16 yields0.385/0.828but more backwards total1.037/1.300;
this is not a fix or a frozen setting change. Discounted history and separate
latent/raw versus projected state also do not restore valid nominal progress.
Independent review verifies original object references within0.4micrometres,
while actual intermediate held states and end hand/object placement differ
from literal R. Reference native-clamp/coupling mismatch is recorded rather
than silently changing original R. No more geometric/prior sweeps are planned.

Next: [TCC alignment encoder Probe](P-20261008-tcc-phase-alignment.md), using
independent nominal source230train views but original R as the only progress
anchor. The user's full chain moves direct GT-value Gate1 before evaluator
fitting. This does not retroactively turn old23/32→27/32rolling evidence into
new reference-progress evidence.
