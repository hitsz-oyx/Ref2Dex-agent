# Decision: one fresh on-policy aggregation step

Question: does exposing the same actor to its own visited states and corrective
teacher labels remove the observed wrist drift and produce independent holding?
Baseline9c0a218failed0/192. Posthoc audit shows ~171mm late wrist target error
on motion1 despite no normalization clipping. This suggests a covariate-shift
hypothesis worth one cheap test; the diagnostic does not establish causation.

Choose a single DAgger-style aggregation, an established baseline technique.
Collect513/514,96env each,202ticks, with frozen baseline policy actions ONLY.
Query the fixed teacher at each saved current state, using the same available
planned reference and label-only schedule. Never execute query actions during
collection. No outcome filtering, no511/512fit reuse. If expert targets exceed
action bounds, fail the design instead of clipping labels or dropping rows.

Fit all58176rows (original510teacher19392 + two visited cohorts38784), uniformly,
2000Adamupdates,lr.001,batch512,weighted action scales unchanged,gradclip10.
Warm-start baseline final734weights, preserve original FIT mean/std; optimizer
reset, seed735 for batch sampling. Keep architecture and normalization fixed.
Final-only checkpoint. New515/516evaluation, same192trajectories and physical105
gate (motion1>=50%pooled and>=25%eachseed), allmotions reported. Historical
strictforce75 secondary. No teacher fallback in testing. Failure stops this
exact aggregation; no additional rounds, update changes or seed search.

One admitted GPU,<=1800s/1GiB; reversible isolated outputs. This establishes
at most initializer feasibility, not Cm utility or the journal-level objective.
If promising, move budget to matched model-assisted policy training rather than
continuing baseline-only optimization. No external authorization required.
