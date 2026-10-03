# Corrected causal surface-feature calibration

Classification: Decision Probe. Previous goal turn was progress: completed
causal execution qualification, rejected this frozen direct-transfer
combination and preserved the evidence. Mission and final claim unchanged.

Question: do pretrained surface features retain action-dependent information
after a new output head is calibrated to corrected native physics? A positive
screen permits fresh randomized native qualification before policy learning;
a negative screen closes this bounded transfer family and triggers a
representation/target review. Existing data and frozen features are the
cheapest discriminating test; no new native collection or policy training.

## Prospective design

Reuse P-20261003-surface-execution-input-r1 and its corrected655 source:
384 whole train episodes /384 disjoint held episodes, balanced within
3 motions x4 own fixed policy arms,16 fixed ticks each,6144 windows per split.
This held distribution has already been examined; it is a diagnostic screen,
not a new independent validation set or policy-seed test.

Same inherited 10135 area-uniform hand samples,64 fixed airplane object
queries, four current neighbors and22 normalized features. Use retained
train-only54-coefficient command/state actuator; do not refit it. Current
q/dq/sent-target predict hand flow. Next object pose is a label only.
Held features/labels are the exact earlier causal bank. Build train features
with the same FK/frame/codec and no contact/outcome selection.

Four fixed arms, all11200 encoder parameters frozen:

1. MANO7168 normal pretrained encoder.
2. MANO7168 shuffled-target pretrained encoder, same original init/data/
   1500-update pretraining schedule.
3. The original untrained common encoder initialization.
4. Normal pretrained encoder, with six predicted local/global hand-flow
   columns zeroed during both training and evaluation. This removes predicted
   hand flow, not all current-state information about past actions.

Local64-dimensional encoder outputs plus global mean yield128 features.
All receive an identical NEW128->64ReLU->3 prediction head (8451 trainable
parameters), output added to explicit previous-object-motion persistence.
Head seed4101, last layer zero, initial prediction exactly persistence.
No extra normalization, old decoder, residual gate, hyperparameter scan,
early stopping or checkpoint selection. Fixed1200 AdamW updates perarm,
batch32 with replacement; shared6144-row schedule seed4102,lr3e-4,
weight decay1e-4, clip norm10, normalized pointwise MSE.4800 actual updates.
Save complete checkpoints, optimizer states, common initialization,
schedule, all losses, first3 head states and all held predictions.
Precomputed frozen features accelerate identical head fits; extra pretraining
cost remains inherited rather than treated as free.

## Gates and diagnostics fixed before fitting

Primary: equal-weight whole-episode mean64-query Euclidean next-object-flow
error in mm, all6144 held windows. Let E be normal pretrained arm error.

Four gates: E<=0.90*persistence; E<=0.90*scratch;
E<=0.95*shuffled; E<=0.95*hand-flow-removed.
PROMISING iff allfour pass. UNCLEAR iff persistence gate and at least one
other gate pass. Otherwise UNPROMISING. No subgroup overrides a gate.
Report zero-flow baseline; all motion/actor groups and inherited current-query
near (<2cm)/far diagnostic, equal-weight available episodes within subset.
Geometric proximity is not measured contact. No policy benefit inferred.

## Engineering, audit and resource boundary

CPU tiny2-row/3-update engineering smoke avoids GPU startup cost: zero head
equals persistence, inherited held codec reconstructed, future-label
perturbation leaves causal input unchanged and encoder receives no gradient.
Main train geometry, encoding, fitting and held inference on one freshly idle
GPU (prefer6). CPU independent source/statistical/geometry/NumPy audit.

All12288 source rows/split/timing independently reconstructed; all6144 train
joint predictions independently reconstructed from retained coefficients.
Independent SciPy FK/KD-tree/frame/target reconstruction on the FIRST fixed
window of every384 TRAIN episode, plus five SDK body origins; held geometry
audit inherited by hash from previous run, not repeated or claimed full.
New train feature/target tolerances atol2e-5/rtol2e-6 normalized units;
KD distances<1e-5m; SDK origin error<2e-4m.
Independent NumPy frozen encoder/head forward for ALL four held banks,
atol2e-4 normalized flow; all parent/subgroup/baseline/gate metrics<1e-4mm.
NumPy gradients and AdamW independently replay first3updates of eacharm,
parameter max<5e-5, loss max<2e-5; remaining4788 updates are NOT replayed.
All encoders unchanged; same new-head initial state/schedule/update count,
positive head change and optimizer step1200 checked.

Fixed code commit before launch, unique run_id/output, inherited inputs,
prior/source/asset/Python/card protected hashes. OneGPU,<=900s/1GiB newrun,
within300GB overall and3October23:59Beijing deadline. Stop OWN child only on
budget/input drift/contention/nonfinite tensors or failed audit. Keep failed
evidence; no CPU fallback for main model work and no overwrite/relaunch on
observation timeout. Terminal evidence committed and separately archived.

After all gates pass: fresh randomized corrected-physics qualification before
policy fitting. Failure: close bounded surface-feature transfer family, no
calibration data/epochs/seeds/coefficient rescue. Formal utility Validation,
novelty and top-journal readiness remain unestablished.
