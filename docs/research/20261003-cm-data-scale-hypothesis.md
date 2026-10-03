# Cm as a transferable prior: scale and coverage remain untested

User raised data scale as an explanation for weak Cm utility on3October2026,
specifically treating Cm as knowledge learned before target policy training.
This note is a route-selection hypothesis, not a measured scaling result.
No new training or data collection is launched by this note.

## What our experiments tested

Recent independent experiments mostly learn task-local physical predictors
from the same airplane and three synthetic reference motions. The actual
successor model uses1536FIT trajectories; the latest budgeted auxiliary model
uses2304physical pairs. Continuous learning uses15360training trajectories,
but they do not supply broad object/contact diversity. The impulse fit has
398224eligible transitions, correlated within trajectories. These counts
cannot be pooled into one compatible training set: input definitions, clocks,
actions, target horizons and behavior policies differ. All these collections
precede the independently confirmed native collision-filter correction.

Thus failed utility gates close specific small/task-local recipes, not the
broader hypothesis of a transferable action-conditioned interaction prior.
We have no controlled data-size learning curve or matched diversity study for
that hypothesis. Historical mixed MANO/Inspire transfer records are separate
evidence, not a successful broad pretraining experiment on this independent
track. A demonstration frame or geometric retargeting alone does not provide
the target simulator's action-conditioned physical transition label.

## Primary-source context and limits

[RoboCat](https://arxiv.org/abs/2306.11706) reports that growing and diversifying
its multi-task/multi-embodiment action-labelled data improves adaptation. It
studies a generalist policy, not our Cm definition; it motivates testing data
coverage but does not identify our failure cause.

[Learning Transferable Dynamics Priors from Action to World Modeling](https://arxiv.org/abs/2606.29501)
reports broad action-conditioned world-model pretraining and downstream
simulator/policy adaptations. This is closer to the prior interpretation,
but its visual diffusion representation and data differ from our small SDK
predictors. Neither paper proves that simply increasing our current samples
will improve manipulation.

Our inference: scale and diversity are plausible unresolved causes; changing
decoder targets repeatedly under narrow data is an incomplete test of a
knowledge-prior claim. Useful physics knowledge still needs compatible action
semantics, correct labels, coverage of consequential contact transitions and
a policy interface capable of using it.

## Minimal discriminating design to prepare

First qualify the corrected common physics and prospective compatible data.
Freeze one Cm target/action contract and trajectory-disjoint held-out set.
Use nested training subsets at three sizes (for example1k/4k/16k independent
interaction windows, subject to feasible collection and resource design),
with fixed capacity and a declared optimization schedule. Distinguish sample
growth from extra optimizer computation; report both. Evaluate held-out
action-sensitive predictions and action ranking, not train loss alone.

Then compare narrow and diverse training at the same sample count, to separate
more repeated experience from object/contact/action coverage. Diversity should
include approach, establishing contact, slip, maintaining grasp and releasing,
as well as suitable object/physical variation. Keep target-policy task fixed
to airplane initially, consistent with MISSION; pretraining diversity is not
an assertion of policy generalization. Actual labels must follow compatible
native execution; immutable legacy physics must not silently enter corrected
physics data. Assess source mismatch explicitly if reuse is proposed.

Only after a decision-worthy model signal, test prior transfer with the same
target-policy initialization, interaction budget and evaluation. Include no
pretraining, correct physical pretraining and an information-destroying
pretraining control with matched architecture/compute. A concrete permutation
control must preserve marginal statistics and destroy the intended action-
outcome association; its validity must be specified before collection.
Pretraining data/compute are separately charged, never described as free
sample efficiency. Offline improvement alone is not Cm policy utility.

This changes the next question within the same mission, not a formal claim:
does additional compatible interaction experience improve a fixed prior and
then help low-data target learning? Exact budgets, sources, gates and stops
require a prospective experiment card before any execution. No large blind
collection, all-recipe replay or claim of inevitable scaling gain is justified.
