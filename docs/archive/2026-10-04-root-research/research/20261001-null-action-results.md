# Exact-null action recovery: negative mechanism screen

Code33d1085; completed
`src/task/CmResidual/research/contact_response/output/P-20261001-null-action-recovery-r1`.
Twelve fixed1000-update fits (four methods,three seeds), GPU5. All32896held
states pass bitwise nativePD-target invariance under independently randomized
six overwritten commands. Input and source hashes match after the experiment.
Saved predictions/recovery values were independently audited on CPU; PASS.

| Variant | Near physical RMSE, mm | Null prediction RMS, mm | Null inverse MSE |
|---|---:|---:|---:|
| Factual-only |16.5567|1.1179|—|
| Full inverse |16.5343|1.0794|1.0019|
| Effective inverse |16.5313|1.0689|1.0232|
| Hard quotient+effective inverse |16.5638|0|1.0230|

The independent-null population inverse-MSE floor is1. Full inverse does not
beat it and its null response is lower than factual-only in each seed. Three
of five interference gates fail; label UNPROMISING. The conjecture that this
specific inverse objective systematically encodes unavailable null information
is not supported by this screen. Do not raise lambda, increase steps or pick a
more favorable representation to salvage this fixed hypothesis.

Raw-input MLPs still predict about1.1mm change under exactly identical PD
targets. Hard canonicalization removes this by construction, without a material
near factual-RMSE improvement. This is useful implementation hygiene, not a
novel empirical method result or a critique of the published AD-WM system.
The omitted inverse-head null outputs in effective/quotient variants are
untrained and their errors are diagnostics only. No controller was evaluated.

Decision: abandon inverse-induced null hallucination as the current explanation.
Use physical canonicalization where appropriate; return to the ongoing fresh
causal transfer result to decide whether factual effects support action choice.
