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

The same-seed export passed the initial motion-ID/start-frame match on
all 64 environments. Official diagnostic actor: 56/64 held-lift,
median first contact step 27.5, median longest contact run 354 steps,
mean peak contact-supported lift 293 mm. Self-trained e320: 4/64,
median first contact step 52, median longest contact run 40 steps,
mean peak contact-supported lift 10.4 mm. All 60 self-trained failures
contacted the object at least once; 57/60 even had a contact run >=20
steps, yet did not lift. The failure is therefore **not chiefly absent
binary contact**. Contact arrives later, is much less persistent, and
rarely becomes a load-bearing grasp. These are exploratory phase
diagnostics, not Cm utility evidence.

## One follow-up mechanism check fixed before reruns

The current binary contact definition only checks *any* configured hand
contact link plus object net force. Export the five configured contact
link force magnitudes for both policies on the same seed/motion. Compare
the fraction of contact frames with >=2 and >=3 active links, and the
mean active-link count. If the official actor exceeds the self-trained
actor by >=20 percentage points on >=3-link frames, treat grasp topology
as a leading Cm target; otherwise investigate force direction/leverage
before choosing another contact-count architecture. Link force is a
proxy: it may include hand-table contact, so do not label it as exact
per-finger object contact. No policy training uses apple labels.
