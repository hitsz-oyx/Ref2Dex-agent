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

Pending.
