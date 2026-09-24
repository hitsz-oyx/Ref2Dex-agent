# P-20260924-expert-option-availability

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe.

## Question

The self-trained actor's local wrist/finger options do not create lasting
cross-object lift, and one-step-action Cm does not improve H20 contact
forecasting over action-blind geometry. Is there *any* useful alternative
action sequence at the same self-trained pre-contact states? If the
read-only official actor can recover those states over a short window,
that supplies data for a future Cm option model; if not, the state
distribution is the dominant blocker.

## Minimal protocol and boundary

Run the pinned self-trained e260 policy as the base trajectory. Load the
read-only official checkpoint as a **candidate-action generator only**;
it is not a final-policy initialization, reward teacher, or inference
dependency in a completed method. At train-object pre-contact states,
randomize 10 policy steps of official actions versus 10 steps of the
self-trained base actor. Both candidate networks see exactly the same
current simulator observation, each with its own checkpoint's input
normalization. Follow ten more steps under the self-trained actor, total
H20. Use 128 envs, global intervention steps 50..190 stride20, seed191,
train objects only. Record both candidate actions, executed source,
object identity, contact and object displacement. No apple data or
checkpoint weight enters model fitting or route selection.

Continue only if official-option minus base-option contact-supported H20
object-z point effect >=10 mm pooled, H20 contact fraction >=+5 pp, and
at least two of three objects have positive lift effects with >=50
treated rows each. If passed, replicate before a new Cm model or
distilled candidate generator. Any future Cm utility claim must compare
matched Cm-on/off **with the same candidate generator**; importing the
official actor itself cannot count as Cm gain.

One idle GPU, <=30 min, <=100 MB. Stop on either checkpoint/input hash
drift, nonfinite action, missing per-network RMS, GPU conflict, incomplete
followup or budget overrun.

## Result

Status: `UNPROMISING` for the ten-step expert-action option from states
visited by the self-trained actor. An 8-env wiring smoke and the full
128-env `agent_expert_option_train_s191_h20_n128` run completed; the
source actor, official candidate and object split hashes were pinned.
767 eligible randomized states were treated. The two policy actions
differed in mean L2 norm by 1.11, so the treatment was not a trivial
identical-action comparison.

Official-option minus self-trained-option H20 contact-supported object-z
effects: airplane +24.25 mm (278 treated), mug -1.61 mm (228),
toothpaste -4.99 mm (261). Object/step-stratified pooled +6.61 mm,
environment-cluster 95% interval [-0.64,+14.44] mm. H20 contact fraction
fell by 8.44 pp pooled; mug alone fell 16.37 pp. The predeclared
>=10 mm, >=+5 pp contact, two-positive-object gate failed. Do not
replicate, train an expert-option Cm, or treat the official actor as a
transferable candidate generator on self-trained states. This result
does not show the expert policy itself is poor: full-episode official
evaluation from its own states remains strong. It reveals a relevant
distribution/coupling mismatch for this ten-step splice.
