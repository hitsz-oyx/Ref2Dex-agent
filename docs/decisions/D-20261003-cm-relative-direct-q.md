# Decision Memo: direct-Q plus Cm-relative residual Probe

Date: 2026-10-03  
Decision type: Decision Probe before any native policy training

## Question

Can Cm add decision information as a relative residual on top of the strongest
available direct-Q score, rather than replacing the baseline with an absolute
physical-value score?

## Evidence

The existing frozen MVE-style `cm_value` teacher already performs a short
dynamics rollout followed by a value continuation, but its native decision
Probe is below the fixed-Cup control.  The direct-Q teacher is the stronger
available baseline in the same candidate-panel records.  The remaining cheap
test is to use only rows assigned to the uniform-random candidate arm, fit a
small held-out residual interface, and check whether its action ranking carries
over across motion/start groups.

## Action

Run `scripts/audit_cm_relative_direct_q.py` on the existing decision-interface
record.  Fit separate local-height and local-reward heads from
`direct_q - direct_q(Cup)` and `cm_value - cm_value(Cup)`, with a 4:1
motion/start group split.  Use non-random arms only to report prospective
action coverage; do not use them as utility labels.

The completed run found only 5 held-out random rows across 4 groups, below the
8-row/5-group support gate.  It therefore returned `UNCLEAR` without fitting
the residual or making a ranking claim.  The existing fresh native randomized
candidate panel remains the utility boundary and had already failed, so this
support failure does not justify new data or PPO.

The Probe would pass only if held-out ranking is positive and improves direct-Q by
at least 0.05 Spearman on both targets, held random rows contain at least eight
top-action matches, and the frozen score changes at least 24/212 rows.  A pass
would justify a fresh native matched action panel.  Any other result closes
this residual interface without starting PPO or expanding ordinary Cm data.

## Cost and stop condition

This is CPU-only tensor analysis of an existing 212-row record.  It creates no
new simulator data and changes no checkpoint.  Regardless of the result, do
not tune the coefficient or scan seeds in this Probe.
