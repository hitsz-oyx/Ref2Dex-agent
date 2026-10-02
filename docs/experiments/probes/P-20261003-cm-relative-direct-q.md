# P-20261003-cm-relative-direct-q

Family: Cm decision interface  
Type: Decision Probe  
Status: COMPLETED — UNCLEAR (support failure)

This Probe tests the smallest residual interface suggested by the current
evidence.  Existing direct-Q is the baseline; Cm contributes only its score
relative to the Cup candidate.  A ridge head is fit on the uniformly random
candidate arm and evaluated on motion/start groups held out from fitting.

The run completed on the frozen record, but the random arm had 42 rows and the
fixed 4:1 group split left only 5 held-out rows across 4 groups.  This is below
the 8-row/5-group support contract, so no residual coefficients were fit and no
ranking or utility claim was made.  A new randomized collection would be
required to test this interface; the existing native randomized candidate
panel already failed its fresh utility gate, so this does not authorize PPO.

The target is local ten-step height retention and local reward.  The random arm
is the only source used for utility labels because its candidate is selected
independently of both score heads.  Other arms are used only for frozen action
coverage reporting.  This is an offline ranking Probe, not a counterfactual
native result and not a policy-training result.

Predeclared pass gates are recorded in
`docs/decisions/D-20261003-cm-relative-direct-q.md`: held-out positive ranking,
at least +0.05 Spearman over direct-Q for both targets, at least eight held
random top-action matches, and at least 24 prospective action changes.  A
failure or insufficient support closes this interface and forbids PPO under
this contract.
