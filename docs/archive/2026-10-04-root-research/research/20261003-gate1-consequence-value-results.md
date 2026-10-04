# Gate 1 consequence value bridge: current recipes do not pass

The Gate 1 contract was implemented on branch `agent/cm-interaction-oracle`
and evaluated before any Cm training or policy update. The exact target was the
recorded simulator Monte Carlo return-to-go (`gamma=0.99`); samples were split
by episode and compared with the predeclared `V_H`, `V_HA`, `V_HE`, `V_HI`,
`V_HEI` and `V_HAEI` models. The primary metric was held-out episode-cluster
MAE, with 1,000 episode bootstrap resamples.

The pinned e260 self-trained source produced 21,004 horizon-32 windows over 41
episodes. It had zero stable-success episodes, so its success/drop auxiliary
labels were constant. Across four independent grouped splits, `V_HEI` versus
`V_H` relative MAE changes were `-41.6%`, `+9.5%`, `-4.9%`, and `-4.0%`; no
positive split had a bootstrap interval excluding zero.

To test whether this was only a lack of outcome variation, a separately marked
self-trained `plain_off` e420 checkpoint was evaluated and collected under an
explicit diagnostic-source flag. The actor-only smoke had 2/6 historical
five-step holds but zero stable successes. The expanded diagnostic set had
14,081 transitions, 26 episodes, 1 stable success and 1 drop-after-success.
Four grouped fits gave `V_HEI` relative MAE changes of `-25.5%`, `-6.9%`,
`-11.3%`, and `+1.6%`; again no split passed the predeclared gate.

**Classification.** `UNPROMISING` for the current actor/data/bridge recipes.
This does not prove that all future interaction representations are useless.
It does close this Gate 1 route for further Cm training, threshold lowering,
or local bridge tuning. Any new actor/data distribution or target requires a
new Decision Memo and a separately identified Probe.

All raw transition shards, manifests, audits and fit JSON reports remain under
`src/task/CmResidual/research/contact_consequence/output/P-20261003-gate1-consequence-value/`.
