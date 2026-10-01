# Direct randomized supervision: completed negative screen

Design predates fitting in eff9c92; implementation2075314, independent auditor
f283bb9. All3072prior randomized windows (seeds492/493) became fit data.
Fresh test seeds494/495have3072/3072complete windows, no excluded rows. Assignment
is independently IID over seven arms, recorded propensity1/7. Maximum sampled
gaps are19.27/17.52/18.77/19.82mm (see native records for full precision).

Fit:30.31s,3.41MiB, GPU5. Test including analysis:134.49s,11.89MiB, GPU4.
Each seed611/612/613fits two1000-update nuisance models, one1000-update direct
contrast model and1000/3000-update factual controls. Nuisance folds separate
acquisition seeds and keep shared environment/actor blocks together. Six fold
input/target normalizations independently match their own fit rows.16saved
model/data hashes verified. Independent CPU predictions and NumPy pseudo-labels,
risk arrays, all12individual points and bootstrap quantiles agree.135test input
hashes verified. CPU audit provides an independent numeric path; the principal
training/collection/model scoring all use GPU.

Negative risk difference favors direct supervision. Units sum squared millimeter
error over three output coordinates and three intervention axes; absolute causal
RMSE is not identified by these statistics.

| Direct minus control | Point mm² | Central95% descriptive interval | One-sided95%upper |
|---|---:|---:|---:|
|Factual1000updates|23.9773|[2.0239,48.1064]|44.2011|
|Factual3000updates|24.2778|[2.3704,48.4344]|44.5863|
|Global residualized fit mean|20.8197|[-1.3111,45.1837]|40.8823|
|Zero effect|20.5414|[-1.2229,45.4365]|40.6384|

All five fixed gates fail: **UNPROMISING**. Direct-minus-factual seed points
32.3527/19.3497/20.2295all have the wrong direction. Motion0accounts for much
of the excess error; motion2has some favorable descriptive comparisons. Neither
subgroup changes the pooled gate. Stop this learner; no extra training steps,
seed choice or post-hoc pseudo-label clipping.

## Post-hoc candidate from the pre-existing controls

After retaining the primary failure, algebraically compare the already frozen
controls using the same saved risk/bootstrap arrays. These were not the
predeclared primary contrasts. They are **exploratory route-selection evidence**,
not a successful primary trial or independent Validation.

| Exploratory comparison | Point mm² | One-sided95%upper |
|---|---:|---:|
|Factual minus zero|-3.4359|-1.1063|
|Factual minus global|-3.1576|-1.2216|
|Compute-matched minus zero|-3.7363|-1.1587|
|Compute-matched minus global|-3.4581|-1.1754|
|Compute-matched minus factual|-0.3004|0.8133|
|Global minus zero|-0.2783|0.7562|

Factual-minus-zero seed points-3.4017/-3.4403/-3.4656; factual-minus-global
-3.0972/-3.1688/-3.2069. No robust advantage of3000over1000updates is apparent.
Choose the simpler factual model as a **candidate**, freezing all three seeds
with equal-weight predictions. The candidate is a standard supervised model,
not novel. Its contrast-risk signal does not prove better lifting decisions.

Next decisive question: does the candidate improve actual retained manipulation
success over native actor, state-independent effect selection and random local
correction under the same intervention budget? A new randomized closed-loop
task Probe is warranted. Its held states, full first-episode retention/drop
outcomes and gate must be fixed before launch. No PPO until task benefit appears.
The original journal-level objective remains active and unmet.

Artifacts: `src/task/CmResidual/research/contact_response/output/` under
`P-20261001-direct-randomized-response-fit-r1` and
`P-20261001-direct-randomized-response-test-r1`, including both audits.
