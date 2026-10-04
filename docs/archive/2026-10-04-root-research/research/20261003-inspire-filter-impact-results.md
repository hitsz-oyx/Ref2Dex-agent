# Fixed-policy response to corrected native collision filters

P-20261003-inspire-filter-impact-r1 completed at codea3118de. One retained
seed655,768environments,202ticks, unchanged P0/actors/normalizers/configs,
zero optimizer updates. All768native hand/table filter checks and independent
actor/P0/PD/fullmesh105tick audits pass. Inputs unchanged; owned parent and
child PIDs absent. Wall70.962s,250472932bytes, GPU6 fresh idle admission.

| Successes /192 | Legacy physics | Corrected physics | Difference |
| --- | ---: | ---: | ---: |
| P0 |65|61|-2.083pp|
| Cm |96|81|-7.813pp|
| Physical-action-removed |101|92|-4.688pp|
| Cold task-Q |95|91|-2.083pp|

Each motion/arm has64episodes. Motion0 remains zero in all arms. Motion1
before[59,52,50,52], after[56,37,40,45]; motion2before[6,44,51,43],
after[5,44,52,46], in the table's arm order. Full105criterion unchanged.

The prospective absolute5pp sensitivity gate passes; the motion0>=4/64gate
fails. PROMISING refers only to material sensitivity of fixed old policies
to the corrected substrate, including degradation; it is not performance
improvement or Cm utility. One seed, no exact native paired-counterfactual,
no fresh learning. Therefore neither the old bug as the root cause of all
negative results nor the corrected-physics training outcome is established.

Keep the verified ownership fix. Treat old counts as legacy physics only.
Next decision is a bounded corrected-environment baseline/data qualification
before further policy-training comparisons. Do not replay every closed
auxiliary recipe, and do not transfer the old learned-policy ranking to a
formal corrected-physics conclusion. Fresh training requires a separate
prospective design with the corrected common substrate.
