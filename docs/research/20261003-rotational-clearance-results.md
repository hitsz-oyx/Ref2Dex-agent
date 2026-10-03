# Full-mesh rotational clearance adequacy

P-20261003-rotational-clearance-adequacy-r1 completed **UNPROMISING** at
code `13cbd0bbf4a756cfab330e8b79cb0222b5431a24`. The prospective
[card](../experiments/probes/P-20261003-rotational-clearance-adequacy.md)
was fixed before analysis. This is a reused-data, observed one-step geometry
Probe, not causal intervention, learned forecasting or policy utility.

All 768 corrected655 trajectories were retained: 202 transitions per episode,
155136 transitions and 155904 current/future poses. Clearance uses the full
25002-vertex airplane mesh and the actual table upper plane. Eligibility uses
only current root rise >=30mm and current clearance in [15,25]mm. Equal weight
is given to each eligible episode; weighted rotation percentiles likewise
give each episode total weight one. No phase, motion or arm was selected to
rescue the primary result.

| Split | Eligible windows / episodes | Decision flips | 95th absolute rotation contribution |
| --- | ---: | ---: | ---: |
| Train | 148 / 123 | 2.9133% | 8.5026mm |
| Held | 144 / 113 | 1.9174% | 7.0397mm |
| All | 292 / 236 | 2.4364% | 7.6855mm |

Held coverage passes the fixed >=128 windows / >=32 episodes gate; rotation
magnitude passes >=2mm. Decision flips fail >=10%. There are zero unresolved
threshold windows. Motion0 has no eligible windows, so its rotational
decision relevance was not assessed. Other subgroup summaries are diagnostic.

For fixed world-up table normal, the translation oracle is
`Ct = Ccurrent + dz`; the observed rotation contribution is
`dr = Cnext - Ccurrent - dz`. These are exact geometric identities. The oracle
uses actual future translation and true current orientation at EACH step;
it is neither deployable nor a simulated trajectory with rotation suppressed.
In the ancillary full105 hold/drop criterion, actual and per-step translation
oracle labels agree for every episode: all325/768, train171/384,
held154/384 successes, zero discordances. This does not prove that a model can
predict translation or that rotation is generally irrelevant.

Independent CPU SciPy/NumPy reconstruction covers ALL155904 mesh poses and
ALL155136 transitions. Geometry differs from GPU float64 by at most
1.11e-16m and retained native clearance by 5.61e-8m. Derived fields differ by
at most1.11e-16 and metrics by2.09e-17. Episode split, source chronology,
all old6D targets/current contexts, sparse ancestor rows, all full105 task
labels and all gates pass. Four synthetic explicit hybrid poses separately
verify the clearance algebra. No subagent review is claimed for this run.

Actual cost:37.967316s,19509015bytes, one freshly admitted GPU6 geometry
phase6.950100s plus CPU smoke3.304308s/audit22.007172s. Zero new optimizer
updates or native simulation ticks. Protected inputs unchanged; owned parent
678695 and children678791/679008/679221 have exited.

The older6D target contains translation and linear velocity changes but no
future rotation. Current64-point flow already contains rotation. This screen
does not support investing in adding rotational response alone as the next
remedy for the current task. It does not reject longer-horizon effects,
unseen actions, other geometry, fine local contact representations or Cm in
general. Preserve the failed gate; return to an action-decision-level review
before another representation fit. Goal ACTIVE; journal NOT READY.

Raw evidence: `src/task/CmResidual/research/contact_response/output/`
`P-20261003-rotational-clearance-adequacy-r1/` (`results.json`, `audit.json`,
`run_manifest.json`, full geometry arrays and phase logs).
