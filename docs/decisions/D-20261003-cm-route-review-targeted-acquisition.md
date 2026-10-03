# Decision Memo: route review after the HF26–HF29 probes

Date: 2026-10-03

## Decision question

After three consecutive valid probes failed to improve the task-value target, is there a
new Cm responsibility worth one bounded experiment, or should the current policy-utility
search stop at the existing-data boundary?

## Key evidence

HF26 consequence-error memory worsened held direct-Q RMSE by `0.238`; HF27 object-orientation
consequence changed conservative continuation RMSE by `+0.019`; and HF29 high-uncertainty
retraining improved selected physical RMSE only `5.1%` against a `10%` gate while worsening
the conservative value-target RMSE from `26.622387` to `26.632624`. HF28 nevertheless showed
that frozen Cm disagreement is a real data-local signal: its top 20% had `7.03x` the physical
error of the bottom 50% and carried `32.5%` of absolute direct-Q residual mass. Existing-row
reweighting failed, so the unresolved question is whether actual targeted transitions are
needed rather than more fitting of the same rows.

## Chosen action

Open one new high-level family, HF30 `native_uncertainty_acquisition`. At eligible native
contact states, freeze the current six-expert candidate panel and Cm ensemble, score each
candidate by ensemble disagreement over translation/velocity/contact/reward/terminal, and
randomly allocate a guarded high-disagreement action versus the baseline action. Record the
actual ten-step transition, uncertainty, candidate panel, propensity and complete terminal
provenance. The Cm predictor remains a one-step physical-consequence model; uncertainty is
used only to choose where to acquire data, not renamed as a success predictor or policy
feature.

The first Probe is only an acquisition/executability screen. It passes if it obtains at
least 32 complete targeted windows across at least 8 motion/start groups, targeted actions
have higher frozen disagreement than baseline, and targeted contact loss and drop rate are
no worse than 5 percentage points. A failure closes native acquisition. A pass authorizes
one fixed fit on the newly acquired targeted transitions followed by the existing held
task-value screen; no policy training, ordinary data expansion, threshold scan or seed scan
is allowed before that result.

## Cost and boundary

One GPU, one native 96-environment collection, at most 96 ten-step windows and 12 minutes
including setup/audit; no unknown process is stopped and no external project is modified.
This changes the data responsibility after the required route review, not the MISSION claim
or the definition of Cm.

## Outcome of the acquisition screen

The fixed native run completed `83` windows, including `36` targeted windows across `28`
motion/start groups. Targeted disagreement exceeded its baseline by `0.004016`; contact
loss was `2.90pp` higher and drop rate was unchanged at zero, so every predeclared gate
passed. The local result is `PROMISING` and authorizes one fixed fit on these actual
targeted transitions followed by the existing held task-value screen. It does not authorize
policy training or ordinary data expansion.

## Outcome of the targeted fit

The fixed 600-update fit used the 36 targeted first-step transitions and adapted only the
Cm final output projection. Although fit loss decreased to `0.111830`, held physical RMSE
worsened by `17.2%` overall and `10.4%` in the high-uncertainty subset. The conservative
value-target RMSE also worsened by `0.091` (`26.622387` to `26.713272`). The Probe is
`UNPROMISING`; close targeted fitting and do not run a policy follow-up.
