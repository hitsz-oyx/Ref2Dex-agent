# P-20261002-reference-target-policy

Decision Probe; r1. Question: is the same teacher information useful for an actor
when its learned output is an absolute reference-relative target rather than an
incremental native wrist command? Fixed design in D-20261002-reference-target-policy.
Source originalteacher510only19392rows; no held-out evaluation reuse. Scratch736,
2000updates, finalonly; new517/518test. Primary physical105motion1>=50%pooled,
>=25%eachseed. Reportall3motions. Native exactPD inverse and bounds at execution;
no scriptedcurl/fallback or sourceactor actions. GPU fit/inference/simulation;
CPU independent trajectory/label audit. Stop on target/action-range/provenance
failure, nonfinite tensors,1800sor1GiB. Failed earlier gates remain unchanged.
