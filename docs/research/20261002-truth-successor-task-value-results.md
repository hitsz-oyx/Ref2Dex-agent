# Fixed-policy truth-successor task value: primary gate fails

Frozen design c483ed5; executed code 0ed0a78. Run
`P-20261002-truth-successor-task-value-r1` is COMPLETED / UNPROMISING.
Previously inspected TRAIN547 supplies 64,174 FIT rows; fresh native panel595
supplies 63,175 TEST rows from576 episodes under the SAME fixed u00 stochastic
behavior. Eligibility uses only observed task history and the known deadline.
The actual terminal physical105 criterion remains unchanged.

| Input / control | Held-out Brier | Oracle relative gain |
| --- | ---: | ---: |
| Current-state value | 0.0576045566 | 1.4285% |
| Current state and executed action, direct Q | 0.0573178164 | 0.9354% |
| Actual next-state oracle | 0.0567816915 | — |
| FIT-only motion/time control | 0.0992000784 | 42.7604% |

The frozen primary criterion requires at least1% oracle gain over EACH control
and a negative upper endpoint of each paired95% interval. The direct-Q gain
fails1%; all paired intervals are negative. Oracle-minus-control intervals are
[-0.0010732244,-0.0005723365] for current value,
[-0.0007765378,-0.0002727913] for direct Q, and
[-0.0528185050,-0.0319721213] for motion/time. These are exploratory intervals,
clustered over192 shared-noise episode groups retaining their three replicas.
The small descriptive oracle advantage does not rescue the primary label.

Three matched networks receive1500 updates each,4500 actual GPU updates total.
Known absorbing task boundaries are analytic; NEXT outcome is already known
for0.8976% FIT and0.9118% TEST rows. Only the oracle receives the actual physical
successor. This is privileged lookahead under factual policy continuation,
not an executable Cm, a policy-training gain, or proof that all possible
physical representations lack useful information.

Native context, commands, probabilities, PD, full-mesh geometry and physical105
labels pass the independent panel audit. A separate audit rebuilds current and
actual successor states, causal task boundaries, FIT-only controls, full TEST
NumPy network forwards, cluster statistics and all gates. Maximum forward
errors are8.806e-7/1.1115e-6/9.553e-7; optimizer steps are NOT independently
replayed. All protected inputs remain unchanged. Completed210.929s and
222,641,171bytes within600s/512MiB; all owned phases are terminal.

Result SHA256: `9cf09c43d07948f2c1ff8169834155bd0af6ac9a98c33476385e18bd02d9079e`.
Model SHA256: `97cca61bb6f03101fe80430286fa02c5ab0fbe0cb6ad8a6ecc7d695160038bdc`.
Prediction SHA256: `335e389e952313830f2443ac985fc8543ee87531681dd4daf639888a82a64bea`.

Close this fixed one-step value contract without lowering its threshold or
scanning width, updates, horizon, time bins, targets or seeds. No actor is
trained from this failed gate. The user is starting a new route; retain this
track as a committed research record. Distinctive methodology, positive
matched Cm policy-training utility, formal Validation, generalization and
hardware evidence remain unmet. Manuscriptv11 is an evidence draft; these
latest results live in the research records and are not yet integrated into
that PDF.
