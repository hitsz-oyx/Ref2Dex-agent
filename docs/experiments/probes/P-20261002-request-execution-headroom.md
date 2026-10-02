# Request innovation execution headroom

- Classification: Decision, reused TRAINING data only; not policy utility.
- Hypothesis: native projection prevents a material fraction of sampled request
  innovations from producing meaningful executable target changes on motion2.
- Decision: change feasible residual parameterization if execution gate fails;
  if it passes, prioritize physical representation/policy integration instead.
- Cheapest information: use saved547--566training traces and exact known actuator
  map; zero model fits, zero physical rollout, zero optimizer updates.
- Model execution: none. CPU is for file/label/command audits and statistics.
- Budget:120s/5MiB new data, one fixed screen; no scans/retries/seed selection.

For all20training panels and all three learned arms, use ALL64motion2episodes
per arm/panel and ALL105original criterion ticks `stop-74..stop+30`, inclusive.
Native post-step progress identifies the same tick's pre-physics request; current
q is initial committed q on tick1or the preceding post-step q thereafter.
The pre-physics counterfactual replaces the actual Gaussian request with its
saved conditional mean and holds q/base target/limits/PD scale fixed. It is an
execution-map counterfactual only; no subsequent physics is inferred.

Independently reconstruct tanh-scaled requests, native wrist interval clipping
and all finger-child bounds/coupling in NumPy. Saved actual target must match
within1e-6absolute per coordinate (stricter than no additional physics). Input
manifests/full raw trace hashes and balanced denominators must verify first.

Meaningful innovation: at least one independent coordinate differs from the
mean-request executable target by>=0.0001m for wrist translations0--2,
>=0.001rad for wrist rotations3--5, or>=0.005rad for finger parents6/8/10/12/14/15.
These thresholds and motion/window are fixed before processing data. Primary
each of Cm/state-only/no-aux >=50%meaningful transitions on motion2in the105window.
Each arm denominator20*64*105=134400, total403200. PROMISING only means material
execution headroom under this operational gate; otherwise UNPROMISING. No
grasp, physical response, Cm policy benefit, statistical harm or novelty follows.

Report per-arm meaningful fraction, exact/null-at1e-7fraction, all12coordinate
fractions, episode distribution and projection max error. Coordinate/episode
breakdowns are diagnostics, never replacements for the primary pooled gate.
Do not inspect/reuse final568/569or modify any frozen continuous recipe/gate.
