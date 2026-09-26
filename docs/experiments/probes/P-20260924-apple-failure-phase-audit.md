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

### Matched apple topology result

Both topology reruns completed on seed 174, with all 64 initial motion IDs
and start frames matched. Among binary object-contact frames, the official
actor averaged 4.39 active configured links versus 2.32 for the
self-trained actor (paired difference +2.08, bootstrap 95% interval
[1.90, 2.25]). The >=3-link fractions were 91.2% versus 41.9%, a
+49.3 percentage-point difference [44.9, 53.4]; the predeclared 20-point
gate passed. Official/self-trained held-lift counts in these reruns were
58/64 and 2/64; simulator outcomes changed modestly across identical
seed/checkpoint reruns, so only the matched initial states are asserted.
This is a diagnostic association, not an intervention or Cm result.
Report: `outputs/CmResidual/agent_apple_contact_topology_s174/report.json`.

## Next cheapest decision check: train-object topology relevance

The apple gap alone does not establish that link count is task-aligned.
Before changing Cm, export the same topology tensor for self-trained e320
on train3 seed 178. Within each train object, compare the per-episode
fraction of object-contact frames with >=3 configured links between
held-lift successes and failures. If success exceeds failure by >=20
percentage points on at least 2/3 train objects, retain multi-link
topology as a candidate Cm target; otherwise drop link-count topology
and investigate load direction/leverage instead. This is observational,
single-seed Probe evidence only. Do not train on apple or claim causality.
Budget: one idle GPU, <=20 minutes, <=100 MB; abort on split or
checkpoint mismatch.

### Train-object result and next temporal check

The self-trained e320 train3 topology export completed on seed 178.
Success-minus-failure >=3-link contact-frame fractions were +21.8
percentage points on airplane (11 success/10 failure; bootstrap interval
[12.4, 32.5]) and +27.8 points on mug (16/6; [17.3, 38.0]). Toothpaste
had 0/21 successes on this topology rerun, so it cannot provide a
within-object comparison; this variation from the previous seed-178
evaluation (3/21) further limits precision. The predeclared 2/3-object
gate passes, but this may simply reflect extra multi-link contact *after*
an object is lifted. Report:
`outputs/CmResidual/agent_train3_contact_topology_s178/report.json`.

Before training a Cm target, make one no-simulation temporal check on
these same transitions: compare successes versus failures on the fraction
of >=3-link frames in each episode's **first 30 object-contact frames
while object z is <10 mm above its initial z**. Require >=10 percentage
points on both airplane and mug to retain pre-lift topology as a
plausible action-predictable precursor. If either fails, do not train a
link-count Cm merely on the full-episode association. Toothpaste has no
success stratum in this run; it is not silently counted as positive.
This remains observational and the configured force proxy can include
table contact.

### Temporal result

The predeclared early/pre-lift gate **failed** on both comparable train
objects. Success-minus-failure >=3-link fractions in the first 30
object-contact frames below +10 mm were effectively 0 points on airplane
and -8.2 points on mug; toothpaste had no success stratum. Thus the
full-episode topology gap must not be promoted into an action-predictive
Cm target. A saturated early multi-link count and possible hand-table
contacts are plausible reasons. Report:
`outputs/CmResidual/agent_train3_contact_topology_s178/report_temporal.json`.
The route decision is to **drop link-count topology** and seek a
pre-lift, object-specific load-bearing signal instead.
