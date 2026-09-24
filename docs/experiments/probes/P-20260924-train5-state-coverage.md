# P-20260924-train5-state-coverage

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Blocker/Decision Probe — self-trained state support.

## Question and cheapest discriminating test

Before building a denser Cm representation, does the frozen self-trained
train5 e320 actor visit enough distinct load-bearing and failure states across
object identities to supervise an object-general interaction model? The
existing evaluation saved episode outcomes but not transitions, so replay the
same checkpoint, split and seeds solely to export first-episode transitions:
train5 seed178 and held-out alarmclock seed174, 64 environments each. No
training, reward change, Cm, official actor action, or new evaluation seed.

For every complete per-environment trajectory, construct current-contact H10
windows. A positive load-bearing window has future object dz >=10 mm and mean
hand/object contact >=0.5; a failure window begins in contact and either has
future contact <0.5 or future dz <=0. Count both by object and episode, and
report held-lift provenance from the unchanged evaluator.

The representation route has minimally adequate support only if at least four
of five train identities each contain >=32 positive and >=32 failure windows,
with positives spanning >=4 environments per identity. If fewer identities
pass, do not fit a larger Cm to this distribution: policy state coverage is
the blocker. If the gate passes, proceed to an object-LOO dense-interaction
representation Probe with raw, action-blind and action-shuffled controls.
Alarmclock remains diagnostic and cannot rescue a failed train gate.

## Budget and stopping rule

Use one idle GPU sequentially, <=10 minutes per replay, <=100 MB per export;
CPU analysis <=10 minutes and <10 MB. Stop on checkpoint/split hash drift,
GPU conflict, incomplete first episodes, non-finite tensors, object mapping
ambiguity, or output-budget breach. This Probe tests data support only and
cannot establish Cm accuracy or policy utility.

## Result

Status: `BLOCKED_BY_POLICY_COVERAGE`. Both replay exports completed with
the pinned e320 checkpoint, split and original seeds. The train replay had
17/64 held-lifts; alarmclock remained 0/64. Of five train identities, only
three passed the fixed H10 support gate:

* airplane: 979 positive windows from 7 environments;
* cubesmall: 252 from 10 environments;
* mug: 2,974 from 10 environments;
* toothpaste: 34 from only 2 environments;
* waterbottle: 28 from 5 environments.

All identities had abundant failure windows. Thus the missing support is
object-diverse positive load-bearing coverage, not absence of negative data.
Only 3/5 identities passed versus the required 4/5, so do not train the
larger Cm representation on this distribution. Alarmclock contained brief
positive H10 windows despite 0/64 held-lift, confirming that short upward
motion alone is not sustained grasp success and cannot override the train
gate.

The sampler duplicates `cubesmall` because the upstream hard-object substring
list contains `small`, assigning it 21/64 environments while the other
identities receive 10–11. The next cheapest decision test is one bounded,
object-balanced continuation from the same checkpoint. If that does not
raise positive support to four identities, stop treating sampling imbalance
as the blocker and redesign the policy representation/curriculum.

Artifacts: `outputs/CmResidual/agent_train5_state_coverage_s178174/` and
the two `eval_*_coverage_*` directories below the train5 run.
