# Decision Memo: Cm decision-interface route review follow-up

Date: 2026-10-03

## Question

After the recent decision-interface Probes, should the current Cm candidate/value
family receive more data, policy training, or local tuning?

## Evidence

Three bounded Probes failed to improve the North-star Cm policy-utility evidence.
Object-projected direct-Q MVE changed native actions but had negative lower90 local
utility and weaker random-panel ranking. Candidate coverage then reached 411 rows
across 45 motion/start groups and passed the support gate, yet the frozen
state/action+Cm adapter still had held lower90 `-22.162 mm`; ordinary data
shortage is not the remaining explanation. The critic residual audit found a weak
episode-Spearman increase (`0.743` to `0.761`) while RMSE, MAE, and row Spearman
worsened.

## Decision

Pause this exact decision-interface family. Do not collect more ordinary candidate
rows, scan thresholds or Ridge settings, or start PPO from these signals. The
current evidence supports retaining Cm as a physical predictor and direct-Q as the
strong task-value baseline, while the Cm policy-utility claim remains open.

If work continues, it must begin with a new representation or decision target whose
held criterion is written before execution—for example a task-conditioned,
action-relative value-equivalence target with an explicit conservative fallback.
That is a new Probe design, not a continuation of the closed selector, MVE, or
linear residual variants.

## Cost and safety

This review changes no core mission claim, checkpoint, external project, or running
process. It records a route decision after the existing bounded evidence.
