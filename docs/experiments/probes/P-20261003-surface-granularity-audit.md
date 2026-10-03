# Current geometry coverage audit

User asks whether point granularity limits the recent prior. Decision:
distinguish sparse object-query coverage from raw hand point count. Cheapest
existing-data check: all18161retained window selections, full4096current
object candidate masks and same64sampled object IDs, actual closest4hand IDs.
No model calls, fitting or new native data. Source/index/provenance/code hashed;
external files remain read-only. CPU file/statistics audit <=180s/10MiB.

Report full/selected near-hand object counts, fraction with no selected near-
hand query despite full candidates, and unique local hand IDs perwindow,
separately by source/split. Mask means unsigned distance<2cm, not actual
contact, support or penetration-free interaction. PROMISING for a coverage
blocker if any group misses >=10% of windows; otherwise UNCLEAR. This gate
does not establish an accuracy/performance cause or justify threshold changes.
High miss rate prioritizes a separately controlled contact-aware sampling
comparison; low miss rate leaves feature-aggregation/capacity questions open.
No dense-vs-coarse model training performed by this audit.
