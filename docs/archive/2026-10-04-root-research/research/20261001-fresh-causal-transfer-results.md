# Fresh contact-conditioned acquisition: matching precondition failed

Collection source59de1cf, source checksums frozen before simulation. Later
commits only add independent scripts/docs; original collection sources stay fixed.
Run `P-20261001-fresh-causal-transfer-r1` completed four new references and
32/32physical arms, then FAILED its mandatory analysis precondition. Total1152.21s.
Original manifest and analysis.log remain unchanged; no automatic rerun.

All four new panels (actors286/287,environment seeds490/491) have96/96current
geometry+force-proxy triggers,32per motion,384total. Full1538x1024sampled geometry
maximum gaps per panel16.60,15.57,18.83,18.12mm. This fixes acquisition coverage
for these panels, not pairwise contact-force attribution or hidden-state matching.

The frozen comparison requires object position RMS<=0.05mm, joint max<=1e-4and
quaternion-component max<=1e-4before treatment. Across the seven nonreference
arms in panelt286_s491, the last two requirements fail. Position RMS remains
within its limit. The first detected failure is zero_b, demonstrating that this
matching problem precedes any action perturbation.

| Panel | Max over arms: position RMS mm | Joint absolute max | Quaternion absolute max |
|---|---:|---:|---:|
|t286_s490|0.003554|0.0000732|0.0000670|
|t286_s491|0.010859|0.0002204|0.0006188|
|t287_s490|0.007110|0|0.0000416|
|t287_s491|0.013398|0.0000982|0.0000881|

The complete engineering audit is `failure_analysis.json`; its scientific
screen label is UNCLEAR, reason INVALID_PRE_INTERVENTION_MATCH. The original
execution status stays FAILED. No arms/windows are removed and no threshold is
relaxed. **No pooled frozen-model causal score or model winner is reported.**
This does not refute the nominal model: its intended fresh causal test is unresolved.

Decision: stop chasing exact cold-replay matching. A randomized trial can target
conditional effect-risk differences without claiming cloned warm solver state.
It requires genuinely new random assignment independent of all pretreatment
states, known propensities and a frozen statistical design. Retrospective arm
selection in this failed dataset is not a substitute for such a trial.
