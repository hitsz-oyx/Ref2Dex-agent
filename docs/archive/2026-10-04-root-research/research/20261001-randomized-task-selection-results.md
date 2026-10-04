# Randomized task selection: negative gate, task specification mismatch

Design003ec5f, implementation c0ec955. Run
`P-20261001-randomized-task-selection-r1` COMPLETED on independently admitted
GPU5; GPU4was occupied when launching.465.87s/44.13MiB. All3072first native
episodes complete;768each actor/seed panel, seeds496/497, no exclusions. IID
four-policy assignments are fixed before physics, unchanged actors and RMS.

Every group shares current proximity gating, clipping masks, candidate inference
and at most10decision events separated by6steps. Actual nonzero correction
counts differ by selected action, as recorded. Conditional control averages the
three1000-update factual models; global control uses seven FIT-only IPW vertical
response means; random control draws valid candidates; native actor alwayszero.

Equal-weight six-stratum policy means (percentages for binary outcomes):

| Policy | Retained success |45-step stable success|Drop after success|Mean max hold s|Mean lift mm|
|---|---:|---:|---:|---:|---:|
|Actor|1.166|11.509|10.342|0.652|9.193|
|Random|1.332|10.588|9.256|0.675|10.364|
|Global|0.542|9.624|9.082|0.543|7.775|
|Conditional|0.803|14.004|13.202|0.815|11.797|

All four fixed gates fail: **UNPROMISING**. Conditional-minus-actor retained
success-0.364pp, central95%[-1.287,+0.575]; random-0.529pp, global+0.261pp.
Conditional-minus-actor drop+2.859pp, upper95%=+5.478, exceeding2pp.
The secondary45-step stable-success difference+2.496pp has lower95%-0.142pp;
mean lift+2.604mm is not retained utility. Conditional gains over random/global
in stable success occur alongside more subsequent drops. Do not replace the
primary gate or assert Cm policy improvement.

Audits: all3072height/contact trajectories independently reconstruct stable,
retained and drop labels, durations and mean lift;30440policy choices reproduce
their recorded actor/random/global/conditional rules. Candidate clipping and
decision spacing/budgets match.142protected input hashes verified; fit-only
global scores independently recomputed. Both audit JSONs are stored with the run.

## A protocol issue found during followup

Read-only source-label audit of the exact three reference tensors shows maximum
consecutive>=3cm lift intervals36/28/25frames; all end within0.50mm of initial
height and have final contact labelzero. These periodic lifting references include
return-to-table behavior. Source reader uses object positions at198:201and
contact label205. The task trained to track those references is different from
maintaining lift for45frames and never returning before episode termination.
Eight archived native runs also have zero retained successes despite70stable
success instances (repeat runs included). Their correlation is not independent
evidence for prevalence. Current fresh retained rates are small, not exactlyzero.

This mismatch should have been checked before freezing the task gate. It is
retained as a design limitation, not repaired using current outcomes. The fixed
gate still fails, but the experiment cannot reject all predictive control or
attribute all counted drops to accidental object loss; intentional reference
return is also counted. It also does not prove that reference return causes
every observed early drop. End this exact controller/protocol combination.

Next Blocker: define a task that actually requests stable holding and demonstrate
a usable frozen self-trained baseline before comparing model utility. A synthetic
reference hold plateau is a new explicitly labeled task variant, not a change
to the old evaluation or a human-data generalization claim. No coefficient,
dose, seed or metric is retuned on the failed current trial.
