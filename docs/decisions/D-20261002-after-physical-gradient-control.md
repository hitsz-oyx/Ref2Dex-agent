# Revisit physical representation and distribution after both learning interfaces fail

Question: does it make sense to train another actor with these fixed physical
models, or must the source of useful physical task information change first?

Evidence: actual Cm actor172/off174/directQ222/P0129 per384. Fresh COMPLETE
actor-gradient trace Cm32.58 versus baseline8.25/off11.94/directQ14.05;
all original gates fail, full independent audits pass. Both model-driven
offline actor and this return-corrected derivative use give no Cm-specific
benefit. The direct-Q actor remains useful self-training evidence.

Action: close the fixed coefficient1/Gaussian derivative recipe; no small
coefficient/sigma/model/baseline scans. Do not launch corrected actor training.
Return to the physical representation/data distribution instead of optimizing
loss or code organization around failed interfaces. A deterministic normalized
mean successor need not preserve physically compatible task-valued uncertainty;
this is a hypothesis, not an identified explanation for current failures.

Next cheap Decision: before proposing an uncertainty-preserving physical
representation or a new interaction distribution, use already collected618
and FIXEDmodels to check whether its actual8-step physical successor gives
useful task-value headroom over directQ, and whether the learned mean-successor
composition misses it. No new fitting/physics or future-state deployment.
Freeze gates/audit before inspecting those predictions; declare reused-data
diagnostic scope. Positive permits a genuinely different uncertainty-aware
physical representation design; negative rejects recycling this continuation
value and prioritizes a different physical interaction distribution.

Neither branch changes Mission/claim. Both require an actual matched trained
actor comparison before utility. This review follows several failed gates and
blocks further local derivative refinement, while retaining improved self-trained
actors and every failure. No new experiment queued at this checkpoint; ordinary
single-GPU budget and read-only/source-protection boundaries remain.
