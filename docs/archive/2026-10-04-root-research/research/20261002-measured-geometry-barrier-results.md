# Actual measured geometry and barrier forecast result

Designbd67bdb/ce1a6cd, fixed P-20261002-measured-geometry-barriers; native runsr1/r2, final fitr3code4ed78a9. COMPLETE / **UNPROMISING**, not Validation.

Fresh578/579FIT120960rows and580TEST60480rows/576episodes, three stochastic u20 heads, all3motions, original105timing and no outcome filter. Four matched162->64->64->2models1500updates EACH, same CPU batch stream after all8GPUoccupied; architecture/weights/data/target/gates unchanged. CPU stream differs from hypothetical CUDA RNG3212; not a bitwise GPU replay. This exception and all admission/contention failures are preserved in separate records. No native trajectory or completed optimizer replay.

| Forecast | MSE of barrier increments/.005m |
|---|---:|
| Cm geometry+action |0.034464663369277|
| Geometry state-only |0.034477385564067|
| Geometry-off action |0.041374244883606|
| Motion/phase+live geometry/action |0.046152166358035|
| Observed barrier persistence |0.075183582358728|

Cm-state difference95%CI[-0.000236461,0.000188113], improvement0.03690%<1%; BOTH state-control gates fail. The other3controls pass. Retain overall UNPROMISING. Descriptive geometry-off gain16.70% and global gain25.32% suggest value in observed geometric state/history, not sufficient action-conditional information, novelty, policy-training benefit or a causal explanation of previous failures. No subgroup rescue or coefficient/step/seed/target scan.

All three old native physical/PD/request/fullmesh audits pass. Separate held-out Torch64 SDK transform/previous-force/flow/barrier audit passes; max rounding error9.537e-7. FIT stats/templates and all control features exactly reconstructed; paired192cluster/bootstrap/gates rebuilt. FIT decoder reused primary reader; model checks are the fixed512sample NumPy forward checks per variant (max3.105e-6), NOT full optimizer/model replay. Explicit verification limits remain fixed.

Conservative native-failed-execution+fit total475.180s/948,381,673bytes within1200s/1GiB, plus independent audit21.587s. All owned children terminal. Source/TEST/model/raw hashes retained; original worktree untouched.

Decision: close the EXACT one-tick shallow action/geometry barrier regression. Do not launch its direct model-gradient actor from the failed gate. Next higher-level question is whether a coherent multi-tick execution contract exposes controllable physical consequences, rather than choosing another readout/lag on the same independent one-tick noise. This changes actual control semantics and must be preregistered/audited. Journal objective ACTIVE/NOT READY.
