# Reference-relative target policy: promising initializer

P-20261002-reference-target-policy-r1, code596c5a0, COMPLETED117.158s,
35,089,617bytes. All226protected inputs independently match. One scratch seed736,
2000fixed updates on19392originalteacher510rows, final checkpoint only:
15ed218fda6b8569e002df3e824369d4cafad1ff027eb115a7ef05d11e34f79a.
No official/source actor weights, no old evaluation fitting, no teacher fallback.

Fresh517/518tests: primary motion1 physical105 **56/64 (87.5%)**,28/32eachseed.
Both frozen gates pass, **PROMISING** initializer only. Allmotions pooled56/192;
motion0andmotion2each0/64. Historical strictforce75is50/192, all50motion1.
This is not full multi-motion manipulation, hardware evidence or Validation.
The reference plan is explicitly available to the trained policy, as in earlier
baselines; this is a reference-conditioned controller, not reference-free control.

Learned network predicts bounded target residuals; native PD inverse computes
commands from current q. No curl schedule is executed during tests. The exact
target map and native joint coupling reconstruct independently from saved
model residuals, planned reference and native scales; maxPD inversion error
1.19e-7. Current-state/planned-context and physical105labels reconstruct;
independent fullmesh clearance error3.66e-8m. Four relevant tests pass.
Two physical cohorts share the same trained model; they are not two independent
optimization seeds. The earlier0/192results remain failed; different output
coordinates, losses and physical cohorts preclude a matched superiority claim.

Decision: freeze this initializer and stop baseline-only tuning. Next work must
test action-conditioned physical information and matched policy-training utility.
The positive Probe does not establish novelty: absolute-target PD control,
imitation and residual-to-reference policies are established techniques.
The journal-level objective remains active and NOT READY.
