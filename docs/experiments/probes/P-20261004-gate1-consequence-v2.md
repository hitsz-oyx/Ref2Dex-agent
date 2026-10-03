# P-20261004-gate1-consequence-v2 — corrected physical timing probe

**Type.** Decision Probe.

**Question.** After correcting the physical transition timing and coordinate
contract, do temporal history plus future object consequence and interaction
(`V_HEI`) improve exact return-to-go prediction over `V_H`?

**Protocol.** A fresh pre-`env_step` collection used six environments and
produced 6,577 transitions over 12 episodes. Horizon was swept over
`{3, 5, 10, 16, 32}` with history length 10. The six predeclared bridge arms
used episode-grouped splits, exact Monte Carlo return targets, 30 epochs, and a
shared initialization seed. This remains a Probe, not a Validation.

**Result.** Relative `V_HEI` versus `V_H` changes were `-16.2%`, `+5.4%`,
`-18.9%`, `-81.7%`, and `-14.8%` for horizons 3, 5, 10, 16 and 32. The test
set contained only two episodes, so the result is not Gate-level evidence.
No horizon met the predeclared 10% improvement with a defensible independent
episode interval. Status: `UNPROMISING` for this small corrected sample,
with implementation repair complete and broader data coverage still required
before any formal conclusion.

**Code and artifacts.** See
[`20261004-gate1-consequence-v2-results.md`](../../research/20261004-gate1-consequence-v2-results.md)
and the `tmp/fresh_pre_s86d*` outputs.

**Follow-up.** Two additional corrected runs expanded the probe to 56 episodes.
The h16 direction stayed positive across five composite splits, but future-action
control and episode composition remained sensitive. With the corrected
episode-balanced control, `V_HFEI` versus `V_HF` was `-9.0%`, `+10.9%`, `+28.9%`,
`+19.7%`, and `+23.2%`; only two intervals excluded zero. The held-out episode
audit found that the first two split gains were about 94% and 96% attributable
to one episode each. This is still a directional `PROMISING` probe, not a Gate
closeout or policy-utility result; the next useful probe needs broader
success/drop coverage and actor-level cluster uncertainty.

**Coverage extension.** Two additional independent e260 runs added 56 episodes.
On that held-out cohort, direct h16 `V_HEI` versus `V_H` was
`+25.9%`, `+32.2%`, `+8.3%`, `-1.0%`, `-2.1%` across the five fixed splits, with
all intervals crossing zero. Combining all four e260 runs (112 episodes) still
gave `+12.1%`, `+15.0%`, `-25.8%`, `+29.8%`, `+5.4%`; only two intervals excluded
zero. The larger same-actor probe therefore did not meet Gate 1 stability and is
closed without Cm training.
