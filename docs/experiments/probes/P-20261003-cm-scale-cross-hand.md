# Cm scale and cross-hand knowledge-prior Decision Probe

User authorized scale experiment and explicitly added hand morphology. Same
MISSION, independent clone, no original project mutation. Read-only existing
OICM native-RL/MANO cache has509train,58val,63test parents, no parent overlap.
Train MANO254parents/50objects, Inspire255parents/50objects. Metadata-only
current-contact inventory gives7946/7329eligible windows respectively, so
fixed nested sizes512/2048/7168 fit both sources. This is not newly collected
corrected-physics data. No learned policy benefit is claimed.

## Question and decision

Distinguish (a) small training bank limits a fixed motion-conditioned prior,
(b) MANO-to-Inspire execution/hand domain shift limits transfer, and(c) this
bounded representation lacks useful knowledge. Cheapest first step is offline
held-parent prediction plus matched low-data adaptation, before native policy
training. It cannot isolate pure morphology from different data sources.

## Fixed data and input

Use original train entries for fitting/adaptation; MANO test63parents and
Inspire val30parents for fixed diagnostic evaluation, never checkpoint/model
selection. Existing val is not a new independent test corpus. One30Hz step,
current t>=1 with current-only full object contact-candidate mask, candidate
frames1,5,9,... (each prev/current/next triplet does not overlap); max64windows
perparent. These are correlated observational windows, not independent trials.
Random ordering seed3801, prefix nested sizes, no outcome-based filtering.
Inspire first256train windows form fixed adaptation bank; disjoint evaluation
parents, same for every adapted arm.

64object surface points (seed3802), closest4 cached current hand neighbors
from area-sampled surfaces (MANO2048, Inspire10135), current object coordinate
frame, positions/normals/local relative geometry, distance, local and global
hand surface flow, previous measured object flow. Common22-dimensional point
features; geometry/.05m, flows/.01m, fixed scales, no evaluation normalization.
Target is next rigid object point displacement/.01m. No hand joint-number
alignment. Surface input reflects different geometry; hand identity is not
an input. Four-neighbor neighborhoods and representation capacity are fixed.

Hand flow is REALIZED t-to-t+1 movement: this is a motion-conditioned prior
and explicitly an offline diagnostic unavailable as a native decision input.
No future object pose/flow enters input. Transfer to a command-conditioned
policy requires a separately validated causal hand-execution bridge. Existing
Inspire physics provenance predates the filter fix and is not silently merged
with corrected physics. No force closure, causal counterfactual, calibration,
pure hand-shape effect or policy gain follows from this experiment. Unsigned
contact distances do not establish absence of penetration.

## Fixed learning

Shared point encoder22->128ReLU->64ReLU; mean latent pooling gives64-dimensional
Cm. Decoder concatenates point latent and shared Cm,128->64ReLU->3. Common
initialization3803, output layer zero. Each of MANO/Inspire512/2048/7168 gets
1500AdamW updates, batch32windows, lr3e-4, weight decay1e-4, gradient clip10.
Fixed update schedule/generator3804 perarm; equal updates, sample counts and
reuse rates separately reported. MANO7168motion-off control zeroes local/global
hand-flow features at fit and evaluation, same architecture/data/updates.

Adapt each MANO size, MANO7168motion-off and fresh common-init scratch using
the same256Inspire windows,600updates, identical batch indices/optimizer.
Motion-off remains off during adaptation, isolating its information condition;
it is not a pure pretraining-target placebo. Pretraining/fitting costs separate.
A MANO7168label-permutation pretraining control uses permutation seed3805,
same inputs/target marginal distribution/1500updates, then the SAME full-input
Inspire256adaptation. This isolates informative initialization from extra
pretraining computation, without asserting a physically plausible null model.
No extra epochs, capacity, target, checkpoint, seed or model selection scan.

## Gates and reporting

Primary metric: mean of parent-level point-flow EPE in mm, eachparent equal
weight. Report all window/parent counts, objects, motion magnitude, zero-motion
and previous-object-flow persistence controls, zero-shot and adapted errors.
Scale signal perhand:7168error<=.90*512error and2048error<=1.02*512error.
Cross-prior signal: adapted MANO7168error<=.90*scratch adapted error,
<=.95*adapted motion-off error and<=.95*persistence error on Inspire heldparents.
It must also be<=.95*adapted label-permutation control error.
PROMISING if cross-prior gate passes; UNCLEAR if only either scale signal passes;
UNPROMISING if none passes. Single initialization, descriptive decision only,
no formal supported/refuted claim. Results do not explain all prior failures.

Positive transfer -> qualify command/motion bridge and corrected native target
data before matched policy experiment. Scale-only -> investigate domain adapter
and coverage, not blind volume growth. No signal -> stop this exact cheap
representation; do not infer all priors fail. Stop after this fixed matrix.

## Budget and audit

One freshly idle GPU for fitting and all batch predictions; CPU mmap/file/
geometric extraction and independent source/prediction reconstruction. Bound
3600s/1GiB new evidence, no external caches/writes, original input statistics
and selected-source bytes guarded. Fixed git commit before run, unique run_id,
own-process-only termination, no unknown GPU interference. Retain packet,
selection/provenance, checkpoints, optimizer/generator state, predictions,
perparent metrics and all failures. Independent audit must rebuild selected
raw rows and labels/features, split/nesting, real updates/checkpoint forwards,
all gates, and input/target separation. Full optimizer replay not claimed.
Prospective numerical bounds: source packet bytes exact; independently rebuilt
features atol2e-5/rtol2e-6, network forward<2e-4normalized flow (2micrometers),
parent metric<1e-4mm, current closest4distances<1e-5m, rigid point correspondence
after own-frame transforms<1e-4m. These are engineering bounds, not task gates.

## Actual outcome

r1 COMPLETED/UNPROMISING, codebc6b576,458.194s/273459235bytes,15600updates,
all own PIDs absent. All18161source rows and independent NumPy/geometry audits
pass, inputs unchanged. Scale improvements8.015%MANO/6.560%Inspire<10%;
adapted prior improves2.143%over scratch/2.330%over shuffled<10%/5%.
All three complete gates fail. Zero-shot4.411876mm and adapted4.726920mm are
retained; no favorable subgroup rescues status. [Full result](../../archive/2026-10-04-root-research/research/20261003-cm-scale-cross-hand-results.md).
