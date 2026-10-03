# Fixed-data object context and neighbor detail result

P-20261003-cm-granularity-r1 COMPLETED/UNPROMISING at code3c19e61. All eight
fixed models trained and all independent audits passed. Both increasing context
from64to256object points and preserving four neighbor slots gave less than0.3%
same-hand error change, with no prospective10%gate reached. This is a bounded
offline representation result, not a formal rejection of finer priors.

## Matched comparison

Eachhand used its original first2048training windows and original held-parent
evaluation (MANO2002windows/63parents/32objects; Inspire884/30/25). Training
MANO254parents/50objects, Inspire247/50. No new outcomes, masks, parents or
supervised points selected. Old64queries remain the first64of each256packet,
and all inherited fields matched exactly. All arms train and score the SAME
64target points. The extra192points provide context only.

All models have49input slots/23107parameters, common3903initial weights and
identical3904batch schedule:1500updates x32windows,48,000window draws and
3,072,000supervised point targets each. Mean repeats four neighbor means;
detail retains four distance-rank relative-geometry/normal/motion slots.
This is a new matched baseline, not the older22input19651parameter model.

Primary equal-weight per-parent64point-flow EPE/mm (lower is better):

| Variant | MANO model on MANO eval | Inspire model on Inspire eval | MANO gain vs64mean | Inspire gain vs64mean |
| --- | ---: | ---: | ---: | ---: |
|64queries, mean|3.456930|4.392716|baseline|baseline|
|256queries, mean|3.457202|4.390056|-0.007878%|+0.060556%|
|64queries, detail|3.465854|4.399568|-0.258159%|-0.155999%|
|256queries, detail|3.464157|4.397399|-0.209067%|-0.106622%|

Six gates all fail. Best primary gain0.060556% is below both10%PROMISING and
5%UNCLEAR gates: UNPROMISING. No subgroup, seed, larger bank, longer training,
threshold, capacity or checkpoint sweep rescues this matrix.

All held-domain scores, including descriptive cross-hand predictions:

| Training arm | MANO held EPE/mm | Inspire held EPE/mm | Fit seconds |
| --- | ---: | ---: | ---: |
|MANO64mean|3.456930|4.327226|7.294|
|MANO256mean|3.457202|4.331850|6.615|
|MANO64detail|3.465854|4.342681|6.324|
|MANO256detail|3.464157|4.336344|6.694|
|Inspire64mean|3.768416|4.392716|6.038|
|Inspire256mean|3.761819|4.390056|6.369|
|Inspire64detail|3.769807|4.399568|6.425|
|Inspire256detail|3.767168|4.397399|6.453|

Original64target zero-motion/persistence baselines unchanged: MANO15.153214/
4.769115mm; Inspire14.999536/5.429061mm. Cross-domain scores cannot isolate
pure morphology from source/action/object-distribution and sampling differences.

## Integrity and costs

156.272s and371854792bytes at terminal, within1800s/2GiB. Smoke3.301s,
file extraction9.496s, GPU fit/evaluation95.215s, independent audit44.726s.
GPU6 freshly idle at admission, no unknown-process intervention. All12,000
actual optimizer updates completed. Dense context has4xpoint observations per
update (12,288,000vs3,072,000context-point draws perarm), with equal supervised
targets/updates; FLOPs were not matched or measured. Near-equal fit wall times
are observations on this small GPU workload, not a compute-efficiency claim.

All6982selected raw rows exactly reconstructed from read-only sources. Both
feature conditions match independent float64geometry (max3.814697e-6), all
rows have monotonic neighbor rank, and pergeometry independent KD distances
match within2.182613e-8m. Saved predictions match independent NumPy forwards
(max4.768372e-6normalizedflow), parent metric max1.119400e-7mm. Future object
perturbation cannot change inputs. Retained smoke checks SE3invariance,
old22feature mean equivalence, neighbor distinctions and real weight updates.
All eight initial states, batch schedules, parameter counts and optimizer
1500step counters agree. Full optimizer trajectory replay is not claimed.
Protected inputs unchanged; parent and all four owned child PIDs absent.

## Decision and limits

Stop this exact granularity matrix; adding query points or retaining these
four neighbor slots gives no useful signal at the fixed budget. The original
global64mean pooling remains; finger identity, attention, higher neighbor
counts, all-hand encoders, much denser queries and converged training were
not compared. A low whole-window2cm proximity miss rate is also not complete
local contact-patch coverage. These results do not establish which alternative
representation would work, or a principal cause of historical policy failures.

Keep the next main blocker: causal command-to-realized-hand-motion input
qualification under corrected physics. This prior still uses realized future
hand movement; prediction performance is not deployable command-conditioned
utility. Legacy Inspire cache physics, reused held evaluation, correlated
windows and one initialization remain explicit. No learned policy, formal
statistical claim, universal hand-granularity sufficiency or journal-ready
result follows. Independent supervision also reviewed gates and aggregation.

Raw evidence: `src/task/CmResidual/research/contact_response/output/`
`P-20261003-cm-granularity-r1/`: guarded manifest/logs, provenance and all raw
packets, eight model/optimizer/schedule checkpoints, all held predictions,
per-parent reports and audit. [Prospective protocol](../experiments/probes/P-20261003-cm-granularity.md).
