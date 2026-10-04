# Test Cm as a training-data acquisition signal

Question: continue local physical prediction/rotation heads or test a different
mechanism for using short response information in policy training?

Evidence: matched head8.955%gain misses10%, rotation1.9174%flips misses10%,
and earlier trained actor auxiliary/successor/physical-feature recipes fail
full utility gates. The old randomized factual learner has a small identified
contrast-risk advantage over zero, but its frozen action selector did not
solve the task. No confirmed neural bug. Generic model-guided exploration and
contact replay are prior art, as documented in the accompanying source note.

Action: open `agent/cm-active-acquisition`, preserving all prior branches and
failed gates. Evaluate a fixed-label-budget acquisition mechanism: use model
disagreement about plus-minus physical response, excluding common state
uncertainty by within-model differencing. Compare with absolute prediction
disagreement and uniform selection. This changes which training labels are
used; it does not rescue the old action-ranking or head-loss experiments.

Cheapest screen reuses3072FIT randomized windows and3072independently acquired
IID TEST windows. All were previously viewed and use legacy source physics/
behavior; no fresh or corrected-physics claims.512initial+512acquired labels
per final learner, shared block-level selection/normalization and fixed fits.
4800new optimizer updates,0native ticks, one idleGPU, <=600s/64MiB. Bootstrap
whole paired actor/environment blocks; acquisition cannot access candidate
assigned arms/outcomes. Fix gates before any model metric. Passing licenses
separate corrected-physics/native acquisition-policy experiment; failure closes
this exact acquisition construction without seed/budget/model-size rescans.

Final objective remains self-trained policy plus matched causal Cm benefit;
offline predictor utility cannot substitute. No mission/claim change, new
permission or external irreversible operation. Goal ACTIVE; journal NOT READY.
