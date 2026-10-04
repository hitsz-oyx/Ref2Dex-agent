# Raw hand points and actual prior granularity

P-20261003-surface-granularity-audit-r1 completed at code927da81. CPU-only
current-geometry statistics, zero model calls, 2.169s. No dense-representation
training or prediction-performance comparison was performed.

The cached MANO hand surface has2048points; Inspire has10135. Both sources
also contain1538decoder points, which the recent surface-motion prior does
not use. Its4096object-point pool is reduced to64fixed query points. Each
query uses4nearest raw hand points; local relative geometry, normals and
motion are averaged, then a64dimensional globally mean-pooled latent is used.
Whole-hand mean motion additionally uses the full raw cloud. Thus raw point
count differs from effective local representation granularity.

All18161retained windows were checked with the SAME current masks, object
query IDs and closest4hand IDs as the completed scale Probe.

| Group | Windows | No selected near-hand query | Fraction | Median selected near-hand queries | Median distinct local hand IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| MANO train |7946|132|1.661%|23|98|
| Inspire train |7329|43|0.587%|28|138|
| MANO eval |2002|17|0.849%|26|105|
| Inspire eval |884|14|1.584%|29|132.5|

Every full4096point mask has at least one near-hand point by the original
selection rule. Near-hand means unsigned distance below2cm; these statistics
do not measure physical contact, penetration, fingertip coverage, individual
contact-patch retention, or force closure. No selected near-hand query is a
whole-window absence test, so a low miss fraction cannot establish complete
local contact coverage.

The prospective >=10% any-group coverage gate was not reached: UNCLEAR.
There is no strong whole-window coverage blocker by this criterion. Averaging
four neighbors and globally pooling can discard local distinctions; whether
this limits accuracy is unresolved. The recent scale Probe's negative gates
are conditional on this representation and do not reject finer prior models.

Decision: retain command-to-realized-motion qualification as the next active
blocker. Before claiming geometry granularity is sufficient or insufficient,
use a separate fixed-data representation comparison that distinguishes object
query density from neighbor aggregation. This audit alone does not justify
another bulk-data collection, a new model fit, or a morphology-cause claim.

Evidence: `src/task/CmResidual/research/contact_response/output/`
`P-20261003-surface-granularity-audit-r1/{results.json,counts.npz,run_manifest.json}`.
Saved arrays independently reaggregate to reported counts/medians. Protected
input SHA hashes remain unchanged, source file metadata checked before/after,
and the owned run PID is absent. Protocol is retained unchanged in the
[experiment card](../experiments/probes/P-20261003-surface-granularity-audit.md).
