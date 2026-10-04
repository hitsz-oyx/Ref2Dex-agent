# Decision: separate geometry reconstruction from same-input model replay

Run34f78b6completes all12updates and all10752native trajectories. Every per-panel
physical/control audit and every update gradient/Adam audit passes. Parent exits
FAILED900.343s only at final model-input replay; its immutable log reports
maximum feature error2.491474e-5against fixed2e-5. No policy-utility label exists
in that failed run. Do not rerun collection or training, select checkpoints, alter
thresholds or overwrite its manifest/scripts/data.

Diagnosis: the final replay fed NumPy-reconstructed physical69states, whereas
the executed GPU actor used the saved native69states. The difference is in
full-mesh clearance column68: maximum2.491288e-8m for541and1.944136e-8m for542,
within the predeclared raw reconstruction tolerance1e-6. The FIT standard
deviation floor.001magnifies this25nm geometry discrepancy to25microunits.
It is inappropriate to mix a raw-state reconstruction check with a tighter
same-input neural arithmetic check.

New analyzer v2first compares independently reconstructed current69features to
the saved actual native inputs with the existing raw tolerance1e-6. Then it
recomputes neural physical features from those SAME saved native inputs using
independent NumPy math and the frozen weights, retaining feature2e-5/logit3e-5
tolerances and exact argmax checks. Diagnostic replay gives maxima<1.1e-6for
features when the input is the same. Keep actual observations and all original
data unchanged; no acceptance threshold is enlarged.

Write corrected analysis to a NEW `analysis_correction_r1` subdirectory, protected
by all original parent input hashes and explicit source-analysis hashes. The
parent remains FAILED; corrected analysis may be COMPLETED with its own scientific
label and the original utility gate, clearly distinguishing runtime status from
evidence. CPU independent NumPy arithmetic only, no training/physics,<=120s/5MiB.
No new permission or resource boundary is required.
