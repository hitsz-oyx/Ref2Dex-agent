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
