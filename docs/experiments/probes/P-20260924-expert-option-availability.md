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

Pending.
