# 轻物体原始接触力工程核对

Engineering blocker diagnostic, not a utility Probe or grasp validation.
Decision: [force reporting](../../decisions/D-20261002-contact-force-reporting.md).

`r1` COMPLETED on admitted GPU0, code d5c3f1d, 96 own environments,
150 ticks at30Hz; hand translated2m away. All final5760 observations satisfy
linear speed<.005m/s, angular speed<.05rad/s, hand distance>1m.
Actual airplane mass.002593613kg, gravity9.81, weight.02544335N,
native4substeps/CC_ALL_SUBSTEPS2. Force median.02552922N, force/weight
median1.003375 (10–90%.587–1.413); old force>.1N never triggers.
Resting source-mesh clearance median.00036766m. This rejects interpreting
the old threshold as contact presence for this light object. It does not
identify hand-object pairs or validate grasp labels. The HF13 geometric
pose-hold failure and all historical thresholds/results remain unchanged.

42.186sec/4.632MB, no updates; inputs verified before any source change,
raw telemetry SHA e39ddeae430600a043a3cd361f9d6b57132add3b38c541b8ad7d0b053717d8ea,
quiet mask/quantiles independently recomputed, both owned PIDs exited.
[Exact terminal audit](P-20261002-contact-force-units-r1-audit.json).

`r2` COMPLETED, same supported protocol plus airborne negative condition.
5760/5760 quiet supported observations trigger normalized presence;0/480
airborne observations trigger it (all raw forces exactly0N, clearance>=.361m,
hand distance>=1.761m). This verifies these two controlled force conditions,
not sensitivity/specificity on actual grasp transitions or identified pairs.
Both runs total86.191sec/9.303MB; inputs/hashes/raw telemetry replay and owned
PID exits verified. [Exact terminal audit](P-20261002-contact-force-units-r2-audit.json).

Executed check: same setup, after resting measurement lift object
.5m, zero its velocity, observe first5ticks of free fall while full-mesh
clearance>.2m and hand distance>1m. Save all raw forces; test a new
weight-normalized presence proxy (>.1 of static weight) against both known
conditions. Preserve r1 and charge both runs; no backfill of old boolean data.
