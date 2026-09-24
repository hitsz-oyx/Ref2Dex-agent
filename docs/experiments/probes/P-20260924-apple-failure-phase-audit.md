# P-20260924-apple-failure-phase-audit

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe / mechanism diagnostic.

## Question

The self-trained multi-object actor achieves only 4/64 held-lift on
held-out apple, while the read-only official actor achieves 61/64 from
its own rollouts. Is the main visible failure phase missing first contact,
losing contact after initially reaching it, or retaining contact without
lifting?

## Minimal protocol and decision

Export the full first episodes of the official actor on the exact same
apple motion and simulator seed 174 already used for the self-trained
e320 apple evaluation; keep the official checkpoint read-only and outside
the final policy. Verify identical environment initial motion IDs and
start frames, then compare per-environment first-contact time, longest
contact run, contact fraction, peak contact-supported lift and held-lift
success. Stratify self-trained failures by whether first contact occurs
and whether contact ever persists >=20 steps.

If failures mostly never contact, focus future Cm on approach/affordance.
If they contact but lose it quickly, focus on geometry-aware contact
retention and the policy's closed-loop action distribution. If they
maintain contact but cannot lift, focus on force/kinematic lift dynamics.
This audit cannot prove Cm utility and apple is now exploratory; final
cross-object validation must use new object identities.

One idle GPU <=20 minutes, <=100 MB output; stop on checkpoint hash,
motion split hash, non-finite export or seed/initial-state mismatch.

## Result

Pending.
