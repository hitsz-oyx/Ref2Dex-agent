# Oracle action-bank capacity after qualified r2

Decision: r2 state/E/I/joint all8/12, with all8motion1/2subjects solved and all
4motion0subjects unsolved. Valid queries and independent checks eliminate the
previous semantic blocker. Joint incremental utility gates fail. Remaining
uncertainty is whether ANY existing alternative can solve those4subjects.

First fix a confirmed implementation issue: float32 sigmoid erases ordering
of distinct high logits. Locked r2 probability choices remain preserved.
Re-select with frozen models/input packet and raw logits, verify CUDA/NumPy
choices, and run only2new selected actions (subjects1/7 options3/4). Same
architecture, weights, normalizer, features, options, physics and success
criterion;0new updates. Raw-logit ranking has exact-real monotone equivalence
and avoids artificial probability ties. Do not claim the bug caused the failed
utility gate: these subjects already succeeded in r2.

Capacity Decision Probe: after the stable choices are locked, collect only
16missing full202tick branches (options1/2/4/5 for subjects0/3/6/9). Reuse
r2baseline,84short queries,26actual branches and all audits by SHA. Entire
scene/seed763/fixed-background protocol remains unchanged. All other8subjects
already attain success, so their per-subject capacity is exactly1 without
collecting unselected outcomes. Compute observed finite-bank best success
for the4remaining subjects from all8actual alternatives. This is a bounded
retrospective capacity diagnostic, never an oracle policy-gain count.

Hypotheses: no finite-bank headroom vs useful unselected alternatives. If
capacity remains8/12, close this eight-option/single-decision control bank and
change the control class before more Cm fitting. If capacity>8/12, quantify
which failures were selection vs missing control opportunity, then consider
matched fixed-background oracle training. No new density/horizon/option/model
seed tuning in this run; no global infeasibility or universal Cm claim.

Budget<=900s/2GiB new data, one admitted idle GPU, CPU artifact audits. New
simulation18full worlds maximum (2ordering correction +16capacity),0new
optimization,6000inherited. Stop on protected-input drift, resource contention
or query/deployment contract failure. Current mission/claim/campaign unchanged;
no new external authorization boundary.
