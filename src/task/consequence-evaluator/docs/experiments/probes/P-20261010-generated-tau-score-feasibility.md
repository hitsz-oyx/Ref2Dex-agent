---
schema: ref2dex.probe.v2
probe_id: P-20261010-generated-tau-score-feasibility
experiment_id: P-20261010-generated-tau-score-feasibility
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: 1352105
claim_id: C3
hypothesis_family: HF-history-tau-proposal
probe_index_in_family: 2
seed_pool: probe
seeds: [297]
decision_changed_if_positive: prioritize a bounded native generated-tau execution and real outcome collection
decision_changed_if_negative: repair the specific scorer or geometry contract before native execution
status: UNCLEAR
run_id: generated-tau-score-feasibility-20261010-r1
---

# Does generated tau preserve score information and native hand geometry?

## Motivation / Decision Note

Continue the active user goal through ref8. Prior goal turn is progress:
1352105 retains a pure measured-history displacement predictor with test
.21266m versus persistence .26214m, and train-only K8 coverage .17191m.
Final chain is H -> candidate tau -> Y -> selector -> tau* -> A -> Z; Mission
still requires eventual Cm utility. Current result proves neither ranking nor
execution, and the goal remains active. No known current task process is running.

Decision/Blocker: before paying for new physical candidate rollouts, distinguish
scorer distribution transfer from unreachable generated hand geometry. Freeze
all model weights. Query futures have no measured Y for their *execution*;
do not attach an old observational candidate's Y to a generated alternative.

## Fixed protocol

Use the existing held18x7 approximate-H panel and frozen tau-only T evaluator
from history-panel-trajectory-utility-20261010-r4; replay its .7007 accuracy and
saved scores before interpreting new scores. H/effect input to T is exactly
zero as in training. Use the declared measured-history displacement checkpoint
from measured-history-tau-proposal-20261010-r1, with its train-only transforms.
Each recorded query reads t-3:t only, no actor observation/ref/clock/force/action.

First diagnostic: substitute each recorded row's predicted future tau into the
frozen T score. Report original observed-tau versus forecast-substitution
pair/regret/shuffle metrics. This tests expected-future score information under
the recorded actor+random-residual distribution, **not same-H generated-candidate
ranking**, and cannot assign labels to newly generated alternatives.

Then generate ten alternatives at each of all18 anchor queries: persistence,
learned displacement, eight train-only nearest-history candidate displacements
from distinct episodes. No label or query future participates in generation,
neighbor ranking or score argmax. Observed anchor future is an eleventh separate
comparison, never an eligible selector candidate. Retain all18 anchors, including
uninteresting/failed ones; old panel composition is recorded, not a fresh sample.

URDF projection uses current live q/hand, candidate hand points and static URDF;
never future q/object/command/force. Batched same300iteration/two-start bounded
coupled-FK fit as existing tau geometry. Report raw and projected scores, palm
rigidity error, FK coordinate RMSE, projection displacement and q finite/range
checks. Geometry screen: coordinate RMSE<=5mm and palm point RMSE<=3mm; first
require observed comparators pass>=80% or the calibration/evidence is UNCLEAR.
Generated score-selected candidates pass geometry on>=90% anchors and forecast
substitution>=.70 with shuffle drop>=.03 before calling this joint screen
PROMISING. FK projection passing does not establish contact/dynamic feasibility.

Projected forecast rows receive only their recorded-policy *diagnostic* labels;
projection itself changes the future, so those labels are not real projected
execution outcomes. No U32/GT label influences fitting/projection/selection.

## Resources / stop

One idleGPU2, new total<=15GPUmin/1GiB, one audit<=840s; no policy/proposal/scorer
training, model sweep, simulation or PointWorld use. GPU inference and batched
FK fit; CPU file transforms/statistics. Tiny geometry contract smoke may use
CPU because CUDA startup exceeds two-iteration/synthetic fixture cost.
Fresh outputs, source hashes/commits, no branch/push/checkpoint overwrite;
preserve foreign processes. Stop on identity/replay/nonfinite/projection budget
or input contract mismatch. If specific defect is identified, record the next
bounded repair Decision Note instead of spending automatic extra seeds.

## Results

Pending. Full scope remains active; this Probe only resolves the next blocker.

## Limitations / future evidence

Approximate-H historical panel,7informative anchors, repeated exploratory held
split, single motion, owned oracle-trained executor history. Scorer did not see
new generated-candidate execution outcomes. Projection static geometry does
not establish contact stability, dynamic reachability, tau->A->Z or Cm benefit.
