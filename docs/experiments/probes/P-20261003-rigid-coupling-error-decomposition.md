# Fixed-weight coupling error decomposition

Decision Probe, post-hoc diagnostic after completed learnability r1, before
any decomposition values are computed. Parent gates/UNPROMISING remain fixed.

Question: are the saved full model's coefficients already useful if candidate
selection is repaired, or is coefficient quality itself a blocker? Cheapest
answer: CPU NumPy reductions over saved fields/coefficients/scores, no forward
model/optimizer/native physics. No architecture or checkpoint selection.

Freeze parent full/state-only/shuffled outputs and capacity fields by SHA.
For all6144held windows, reconstruct full's15candidate flows with learnedw.
Report actual deployed winner; true-EPE-minimizing winner with those SAMEw;
parent outcome-fitted oraclew at the learned winner; full oracle; and full
model restricted to the2state endpoints. Aggregate equal episodes, plus
fixed near/far subsets inherited from parent. Selection regret is actual
deployed EPE minus best same-w candidate EPE; coefficient gap is best same-w
EPE minus full oracle. These oracle substitutions use future outcomes and
are deliberately nondeployable. They do not identify causal contact.

If best same-w candidates beat matched learnedstate-only by>=10% overall
AND>=5%near, prioritize a separate prospective candidate-risk/ranking design;
otherwise do not pursue a score-only repair. Either result closes this exact
coefficient-plus-confidence recipe, with no steps/seed/width rescue. This
diagnosis cannot authorize actor training or native qualification.

Limit60s/1MiB new output, no GPU: pure saved-array error/statistical reduction.
Save result and all source hashes in unique run directory; verify current
deployed metric agrees within1e-9mm and all candidate errors respect full
oracle up to1e-7m. Same previously viewed held panel, diagnostic only.
Goal ACTIVE, journal NOT READY.
