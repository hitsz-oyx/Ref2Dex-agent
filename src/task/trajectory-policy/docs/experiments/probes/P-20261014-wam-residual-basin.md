---
schema: ref2dex.probe.v2
probe_id: P-20261014-wam-residual-basin
experiment_id: P-20261014-wam-residual-basin
date: 2026-10-14
task: trajectory-policy
branch: main
git_commit: 883464be
claim_id: C3
hypothesis_family: HF-wam-residual-ppo
probe_index_in_family: 3
seed_pool: probe
seeds: [292, 296]
decision_changed_if_positive: prioritize targeted finger exploration or a warm-start bias on the actual startup-contact states
decision_changed_if_negative: do not close residual RL from four directions; prioritize actual visited-state coverage and controller geometry
status: RUNNING
run_id: wam-ref2-execution-basin-20261014-r1
---

# Frozen finger authority screen after the corrected PPO budget/course probe

Serve Objective A and decide whether a few explicit finger corrections on
the exact frozen WAM+old Chunk-R pair can form startup hold45. Corrected
PPO reached100 updates under the contact course without a strict frame0
hold. Its teacher-contact starts improve, but self-visited contact poses
remain different and initial thumb-pitch exploration is only .0056 rad.
This is a Decision Probe, not a convergence or RL-impossibility claim.

Reuse the saved measured prefix/provenance from contact B2 final.pt at100.
The learned residual actor mean is bypassed; candidate0 is exact zero
residual on the frozen WAM+R. No gradient update, expert rollout intervention,
World input, R swap or model-weight change occurs. The launcher reuses the
audited native runner, captures native PD/command, and independently asserts
the complete recorded residual schedule. final.pt is only an unchanged copy
of the parent network/training state, not a learned intervention policy.

| candidate | latent correction during controls40:160 |
| --- | --- |
| base | zero |
| thumb_pitch_minus | coordinate11=-4 |
| thumb_pitch_plus | coordinate11=+4 |
| finger_close | coordinates6:12=+2 |

Execute clip(base+.15*delta) on12 active native coordinates. Corrections to
thumb pitch are ±.165 rad before clipping; finger-close adds .24 rad to four
fingers, .1725 rad thumb yaw and .0825 rad thumb pitch. These bounded native
commands intentionally test directions beyond the initial Gaussian's usual
finger samples. Wrist residuals remain zero. The intervention ends at160;
all copies then continue the frozen base to the full542 endpoint.

Decision Note: spend one four-env forward-only native run, GPU6 if free,
734 batched RPCs including expanded zero-residual diagnostics, expected3–5
minutes and<30 MiB output. Native hard cap900s plus120s server readiness.
Abort on the existing context/native/GPU safety checks or a residual schedule
mismatch. If a finger candidate has hold45 while base does not, retain a
PROMISING authority/exploration direction and test it on actual visited
states before another long PPO block. Otherwise record UNCLEAR for these
four directions; this sparse screen cannot exclude other wrist/finger/time
corrections. No external authorization boundary is expanded.

Limitations/future evidence: one reference, one copy per candidate, GPU
physics variation, potential clipping, a fixed intervention window and no
claim of learned-policy success or controlled placement. The parent actor
is deliberately bypassed, so this is an exact-base intervention screen,
not an A/B evaluation of the100-update learned residual. Matched repeats
and broader feasible-direction scans are deferred unless a positive signal
changes the next decision.
