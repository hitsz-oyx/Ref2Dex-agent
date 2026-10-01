# Decision: resolve task compatibility before further predictor tuning

Question: is the reference-tracking substrate suitable for45-step hold plus
retention-to-episode-end evaluation?

Evidence: randomized controller gate fails on3072complete episodes. Frozen
source references contain at most36/28/25consecutive lifted frames and return
to initial height at termination. Full-episode nonreturn demands departing from
their intended behavior. This is a task-specification limitation, independently
supported by read-only labels, rather than grounds to salvage the old gate.

Action: preserve UNPROMISING and stop the tested one-step controller. First
construct one deterministic synthetic hold-plateau variant from reference labels,
then evaluate existing self-trained actors without model corrections or training.
This is a new holding task, clearly separated from original reference-tracking
evidence. Do not lower45-step hold criteria to match the short source lift cycles.

Cheapest decision test: insert90stationary raw reference rows at the maximum
height within the FIRST>=3cm lift interval of each source tensor, with all598
pose/contact/table/joint columns copied. Native loader recalculates velocities
from the resulting trajectory. Initial row/assets remain identical. Selection
uses source labels only, never current trial success examples. Evaluate a fixed
75-step continuous contact-supported hold within the inserted phase (45hold
plus30followup), avoiding the later deliberate release. This is stricter than
five-step historical held-lift and remains proxy based.

Cost: label generation CPU<60s/10MiB; actor-only new-task Probe on one admitted
GPU<=1800s/256MiB, four192-env panels, fresh seeds498/499. If baseline coverage
passes the fixed gate, freeze the task and design a fresh matched utility trial;
if it fails, establish a holding baseline/curriculum separately before more Cm
policy work. No PPO is authorized by the current negative predictor/controller
gates. A separately designed baseline training Probe can be considered only after
this diagnostic, within the same user's autonomous research authorization.

Stop input drift, incomplete cohorts, budget overruns; preserve all old artifacts.
External data/checkpoints remain read-only. No new external permission is needed.
Journal objective remains active: no novel method or consequential utility yet.
