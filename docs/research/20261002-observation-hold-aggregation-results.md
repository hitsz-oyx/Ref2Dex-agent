# One on-policy aggregation: failed independent gate

P-20261002-observation-hold-aggregation-r1, codee2a38c2, COMPLETED184.698s,
80,681,979bytes before separate posthoc diagnostic. All217protected inputs match.
Two fresh frozen-baseline behavior cohorts513/514,38784current-state teacher
queries; no query actions executed. Original510teacher19392rows retained;
all58176fit rows, no old evaluation reuse. Baseline weights and FIT normalization
preserved,2000new updates only. Final checkpoint7781a96e1fc2284350be5d9c9f424b35be084915564f57cf8dfeacf7cd7d9add.

Fresh515/516: physical105 **0/192**, motion1 **0/64**,0/32eachseed; all other
motions0too. Strictforce75also0/192. Both fixed gates fail, **UNPROMISING**.
Independent causal-context/native-PD/physical labels pass; clearance error2.25e-8m.
Five focused context/query/support tests pass. No source actor acts, no Cm orPPO.
End this exact aggregation, no more rounds or update/seed/model selection.

Separate posthoc target diagnostic shows late motion1wrist RMSE69.513/64.733mm,
finger-target RMSE0.07021/0.06855rad; maximum motion1root rises13.988/14.820mm.
Wrist error is smaller than the earlier baseline diagnostic, but these are
different physical cohorts, not a matched causal claim or successful grasp.
New distinct test changes output coordinates to reference-relative absolute PD
targets, trained from originalteacher510only and evaluated on new517/518.
This is an established control parameterization, not a Cm contribution.
