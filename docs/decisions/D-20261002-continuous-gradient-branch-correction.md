# Correct a witnessed floating-point ReLU branch in the independent audit

The fixed policy comparison stops at u17's first-minibatch audit, before final
evaluation. All 17 training panels and actual u17 optimizer/checkpoint outputs
are retained; no scientific utility label is available. Original audit threshold
gradient2e-5 rejects maximum1.023578773e-4, despite forward1.545e-6 and
Adam1.166e-7. The original failed parent/log remain unchanged.

Saved-batch engineering replay on the same admitted GPU4 reproduces ALL actor
and critic gradients, forward outputs and losses exactly in all three variants.
There are no optimizer updates or native simulation. A single no-aux actor
first-layer ReLU, sample477/unit30, is positive3.813118607e-8 in independent
float64 arithmetic and negative-2.980232239e-8 in the original float32 GPU
arithmetic. Both fall within an independently computed FP32 dot-product rounding
enclosure. Its upstream derivative4.904055582e-5 explains the discrepant first
layer weight/bias gradients; it is not a clipped-PPO or model-wiring discrepancy.

A SEPARATE correction uses the witnessed GPU branch after checking every
first-layer preactivation against the independently computed rounding enclosure.
All gradients, returns, joint norm and first Adam updates are then reconstructed
in NumPy using the SAME scalar tolerances. Corrected maximumgradient1.2223e-6;
all original limits pass. The correction is explicit, restricted to the diagnosed
u17 single branch, with source/replay hashes. Original auditor stays byte-identical,
and its failure is retained rather than changed to a pass. This is a witnessed
floating-point branch audit, not an assertion that float32 and float64 derivatives
must agree at nonsmooth boundaries, and not a novel scientific method.

Continue EXACT u17 checkpoint/Adam state in a unique resume-r3. Retain native
547--563 and u00--u16 through protected links; u17's new read-only view links the
original actual checkpoint/packet/results and the separate corrected audit. No
old path is overwritten. Collect only564--566/u18--u20 and final-only568/569,
with unchanged actors, objectives, reward, native physics, cohorts, metrics and
seven gates. The original analyzer remains unchanged; final closeout and paper
must name the correction and GPU replay scope explicitly.

Reserve2550s for original/resume-r1/resume-r2 and engineering diagnosis/margin,
leaving1050s and the SAME combined6GiB budget for the remaining phases. GPU4
must be readmitted and uncontended; no automatic retry if another failure occurs.
Success permits complete final-only interpretation; failure retains an incomplete
comparison. No tolerance expansion, scientific redesign, unknown process action,
external checkpoint modification or new authorization boundary is involved.
