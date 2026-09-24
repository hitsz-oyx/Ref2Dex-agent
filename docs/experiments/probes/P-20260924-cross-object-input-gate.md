# P-20260924-cross-object-input-gate

Date: 2026-09-24. Branch: `agent/cm-cross-object`. Starting commit: `44f0982`.
Classification: Blocker / engineering smoke, not a Cm-effect experiment.

## Question and decision

Can the already validated GRAB-to-DExplore reconstruction path produce a
physically usable lift reference for a *different object identity*? The
existing self-trained and Cm results use airplane and do not answer this.

If a new object passes tensor, alignment and simulator smoke, make an
object-disjoint split and run the cheapest frozen-policy transfer Probe before
training a cross-object Cm. If it fails, repair data/scene alignment or switch
to a documented alternative source; do not interpret policy failure as lack of
generalization.

## Fixed minimal protocol

Start with `s7_apple_lift` (available in legacy GRAB, raw GRAB and geometric
Inspire references). Reuse the existing reconstruction adapter without editing
external projects. First convert the uncorrected motion, infer per-frame body
translation from the geometric reference using the existing
right-hand/object-relative method, then convert the corrected motion. Audit
finite `[T,598]` tensor, left/right contact,
hand-object relative alignment, table/object geometry and provenance.
Only then run a short frozen actor simulator smoke on apple. The frozen actor
was trained on airplane; apple must not enter its training data.

Data conversion: at most one idle GPU, <=60 min and <=2 GB local output.
Simulator smoke: at most one idle GPU, <=20 min and <=1 GB output.
Stop on asset mismatch, non-finite tensor, empty/invalid contact, input hash
drift, GPU conflict, or a large unexplained hand-object alignment error.

## Result

The first uncorrected conversion used the pinned upstream official converter
as a diagnostic. It produced finite `[481,598]` data, but all 32 hand-contact
columns were `-1`, unlike the current self-trained policy's `0/1` conversion.
Therefore it is *not* eligible for policy evaluation. Continue with the local
DExplore converter matching the training data; preserve the upstream output
only as evidence of this producer mismatch. Engineering smoke cannot establish
Cm policy utility or cross-object grasp generalization.
