# P-20261003-cm-exact-paired-panel

Family: Cm action ranking  
Type: Decision Probe  
Status: COMPLETED — UNCLEAR (engineering boundary)

This Probe executes all eight native candidate actions from each of 24 live
contact/lift states.  Every candidate is compared with fixed Cup using the
same-state `paired_sim_step` restore and a repeated base action.  Cm is frozen
and supplies only physical consequence predictions; there is no future-state
input, success classifier, policy update, or PPO.

The smoke reached one eligible state, but restoring root/DOF from that hot
contact state left the PhysX rigid-body tensor 8.4–40.5 units from the saved
state, above the fixed `1e-6` restore contract.  All non-Cup candidates were
therefore rejected before an effect could be measured.  The result is an
engineering boundary, not a Cm utility result; the tolerance is not relaxed.

The primary label would have been one-step object-height effect relative to
Cup.  The predeclared gates and stop condition are in
`docs/decisions/D-20261003-cm-exact-paired-panel.md`.  This Probe is intended to
decide whether a short real rollout is justified, not to establish final task
success.
