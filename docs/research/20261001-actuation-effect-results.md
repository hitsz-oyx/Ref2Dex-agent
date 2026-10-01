# Actuation/effect factorization: completed exploratory screen

Code `bbd5ba4`; output
`src/task/CmResidual/research/contact_response/output/P-20261001-actuation-effect-factorization-r1`.
Completed in53.64s on GPU4. Five feature tests pass. All30 held RMSE metrics
were independently recomputed from saved predictions; audit PASS. Sources and
read-only shards were checksum verified before and after the run.

Fit32902frames; test32896frames from disjoint historical960-episode collections.
The test-near stratum has1756frames from639episodes. These reused episodes are
not new independent Validation data. Three seeds are optimization repeats, not
three separately acquired physical datasets.

| Effect input | Near vector RMSE, mm (seed mean) | All RMSE, mm |
|---|---:|---:|
| State only |17.1797|7.3278|
| Native command |16.4363|7.3303|
| PD-target palm/tip motion |14.5985|7.0929|
| Predicted realized palm/tip motion |15.0663|7.2060|
| Measured future hand motion, privileged |13.9941|7.0006|
| Passive constant velocity |23.4004|7.5873|

The primary learned-motion predictor improves8.33%over native commands, failing
the fixed10%gate. It passes the other four gates. Label remains UNPROMISING;
the nominal variant cannot replace the predeclared primary method retroactively.
Actuator near wrist-position errors11.75–12.07mm show a substantial prediction
gap. Neither future-hand diagnostics nor factual RMSE establishes a causal mediator.

The secondary, available nominal variant improves11.18%over native commands
and is better in each seed. This warrants a separately frozen *fresh causal
transfer screen*, not a claim of novelty or an upgrade of this failed screen.
The label-free nominal no-slip transport baseline is much worse (133.64mm near).
Motion conditioning needs learned effects; simply transporting the object by
the desired hand flow is inaccurate here.

No policy benefit, robustness across objects/hands or journal readiness follows.
