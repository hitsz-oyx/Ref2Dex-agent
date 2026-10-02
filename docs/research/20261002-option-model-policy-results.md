# Actual physical-successor option-policy learning: negative Cm utility

Probe `P-20261002-option-model-policy`, frozen design3cde51a, learning/native
implementation3ae0701. Parent r1 preserves all completed physics/learning/audits
and FAILED only at NumPy `bool_` JSON serialization. Separate correction54d3c2a,
r2 COMPLETED, executes the exact original counts/gates with scalar serialization
and separately reconstructs every count and decision using Python integers.
No native trajectory, optimizer step, label, tolerance or threshold repeated.

Actual new data:1536FIT603/604 trajectories and1536EVAL611/612 trajectories,
768environments/202native ticks per panel. Placements and treatment arms are
randomized; no exact same-state replay or retrospective oracle assumed. All
four panels retain full current/P0/PD/native geometry/physical105 audits.
Physical105 still requires root rise>=.03m and full-mesh clearance>=.02m on
every75plateau+30drop-checking tick; synthetic90plateau remains explicit.

Two131-dimensional8-tick physical residual models and two option-conditioned
task value models take1500updates each,6000actual model optimizer steps. Three
identically initialized152-observation actors take1000updates each,3000actual
offline policy optimizer steps. Same data/minibatches/actor regularization;
Cm and action-removed dynamics use the SAME option-conditioned continuation V.
Direct-Q is an additional strong control. Deployment uses trained actors only.

| Method | Successes /384 | Rate | EVAL611 /192 | EVAL612 /192 |
| --- | ---: | ---: | ---: | ---: |
| Unchanged self-trained P0 |129|33.59%|63|66|
| Cm successor -> continuation V -> trained actor |172|44.79%|88|84|
| Action-removed dynamics -> same V -> trained actor |174|45.31%|79|95|
| Direct task-Q -> trained actor |222|57.81%|107|115|

| Motion | P0 /128 | Cm /128 | Dynamics-off /128 | Direct-Q /128 |
| --- | ---: | ---: | ---: | ---: |
|0|0|0|0|0|
|1|107|110|110|105|
|2|22|62|64|117|

Frozen pooled>=5pp over EACH of P0/off/directQ: FAIL. Each evaluation seed
noninferior to ALL three: FAIL. Motion1 loss<=5pp versus P0: PASS. Primary
label **UNPROMISING**. Cm exceeds P0 descriptively by11.20pp but trails
dynamics-off by.52pp and directQ by13.02pp. This fails to establish a
Cm-specific policy-learning gain. Two EVAL seeds share one trained model;
none of this is formal multi-training-seed Validation.

The strong direct-Q result is useful new self-training evidence: motion2
117/128 versus P022/128, and same-input option learning can improve actual
task performance. It is not a Cm result, not universal all-motion competence
(motion0 stays0), and not distinctive methodology. Retain this actor and all
three alternatives as bounded, independently audited artifacts.

Engineering evidence: all current/successor72+SDK80 features independently
reconstructed, maximum discrepancy0; exact FIT statistics after shared primary
float32 decoding. All final FIT network outputs independently compared with
recorded GPU predictions: physical models<=1.619e-6, value/Q<=4.696e-7,
actors<=4.505e-7. Actual deployment observation error0, ALL actor forwards
<=4.707e-7; native full-mesh clearance<=2.329e-7m and native targets<=4.769e-7.
First actor gradients are nonzero(.4108/.1968/.3133), parameters actually
change(.3307/.3011/.3138 maximum). The tiny analytic seam independently
verifies the physical Cm gradient path and its removal. No independent
optimizer replay is claimed; counts and retained execution establish updates.

Parameter counts and per-step model calls are in fit/results.json; Cm/off
composition uses two model calls versus direct-Q one. Pretraining and actual
evaluation data costs are reported separately, not called free online samples.
Model loss and predicted return do not replace actual policy outcomes.

Conservative r1+r2 cost633.273s /997,392,487bytes within1200s/2GiB. GPU6 freshly
admitted for each native/model/actor phase; independent files/NumPy audits on
CPU. All protected inputs verify; original failed parent and log remain intact;
all owned processes terminal. Result SHA256:
`2fb20999d1eb90c7eb3dbcaf259def75c125b8f5404b2a0ede9e211c6cf474f3`.

Close this exact offline physical-successor actor recipe without penalty,
width, update, horizon, threshold, seed or checkpoint rescue. The next role
to assess is a return-corrected control variate, where a model can affect
gradient noise without substituting its prediction for measured task returns.
Its variance mechanism and prior-art boundary must be checked before another
actual policy-training comparison. Goal ACTIVE; journal readiness unmet.
