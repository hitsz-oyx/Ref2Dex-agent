# Probe: native uncertainty-guided transition acquisition

Date: 2026-10-03  
Experiment ID: `P-20261003-cm-uncertainty-acquisition-native`  
Status: `PROMISING`

This card records the first bounded Probe authorized by
[Decision Memo](../../decisions/D-20261003-cm-route-review-targeted-acquisition.md). It
will freeze the existing Cm consequence ensemble and score the six native expert actions
at eligible contact states using disagreement over object translation/velocity,
contact, reward and terminal outputs. A guarded high-disagreement action is allocated
against the baseline after the full current-state panel is available. The collector will
save the actual action, state/history/context, disagreement, propensity, ten-step future,
contact and terminal provenance.

The acquisition screen does not claim policy utility. It requires at least 32 complete
targeted windows across 8 motion/start groups, a higher targeted disagreement than the
baseline arm, and no more than 5 percentage points worse contact loss or drop rate. A
failure closes native acquisition. A pass permits one fixed fit on these new transitions
and the predeclared held task-value screen; it does not authorize policy training or
ordinary data expansion.

## Result

The completed native run produced `83` complete ten-step windows: `36` targeted and `47`
baseline, spanning `28` targeted motion/start groups. Targeted disagreement was `0.435071`
versus `0.431055` for baseline (`+0.004016`). Targeted contact loss was `16.94%` versus
`14.04%` (`+2.90pp`, within the `5pp` gate); both drop rates were `0%`. All five gates
passed, so the acquisition Probe is `PROMISING` at its local boundary.

This result authorizes exactly one fixed physical-consequence fit using the newly acquired
transitions and the existing held task-value screen. It is not a policy-utility claim. The
five earlier startup failures are retained as engineering provenance; only r6 is the
scientific collection run.
