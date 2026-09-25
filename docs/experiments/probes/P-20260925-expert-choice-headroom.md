# P-20260925-expert-choice-headroom

- Classification: Decision Probe.
- Cm: off; all five PPO experts are self-trained.
- Question: Is there state-specific choice headroom beyond the frozen
  object route large enough to justify fitting a Cm expert selector?

## Protocol and decision

Evaluate each of the five frozen checkpoints as the sole expert across
the same twelve converted motions on seed219, 64 first full episodes,
early termination disabled. Also evaluate the fixed object-ID route
twice on seed219 with fresh processes to estimate repeat variability.
Align outcomes by environment ID, motion ID and start frame. For each
initial state, count whether any checkpoint succeeds when the fixed
route fails; this is an *observed best-of-five ceiling*, not an attainable
policy score. Count successes in fixed-route repeat B when repeat A
fails as a minimal noise reference.

If the best-of-five ceiling exceeds fixed-route A by >=12/64 and exceeds
the repeat-only gain by >=8/64, and at least three object identities
have complementary winning experts, proceed to offline Cm option-value
modeling with action-aware, action-blind and shuffled controls. Otherwise
prefer improving expert coverage or the baseline; do not fit an online
Cm router without a meaningful choice opportunity. One idle GPU,
<=35 minutes, <200 MB output. Stop on checkpoint/config drift, GPU
conflict, incomplete evaluation, or invalid state alignment.

## Results

All seven sequential seed219 evaluations completed, with matching
environment IDs, motion IDs and start frames in all 64 episodes. The
fixed route scored **20/64** in run A and **23/64** in independent repeat
B. Individual uniform expert scores were source e260 **4/64**,
mixed12 e300 **6/64**, train5 e320 **10/64**, balanced e360 **17/64**
and duck e340 **9/64**.

The observed best-of-five ceiling was **33/64**. Fourteen states failed
under fixed run A but succeeded under at least one uniform expert run;
three states failed in fixed run A but succeeded in fixed repeat B.
The exploratory excess over this one-repeat noise reference is
**11/64**. Airplane, alarmclock, apple and cup each had different
experts uniquely succeed on different states, satisfying the
predeclared three-identity complementarity condition. The Probe gate
**passed** (`PROMISING`) for offline expert-option value modeling.

The ceiling is optimistic: it selects after observing the outcome of
five separate simulator executions. Fixed route A even succeeded twice
on cubesmall where the five uniform runs had only one success in total.
The one-repeat noise subtraction is not a formal correction. A Cm
selector must beat object-only and action-shuffled controls on fresh
states before any policy-utility claim.

Artifacts, frozen route configs and per-run manifests:
`outputs/CmResidual/agent_expert_choice_headroom_s219/`.
