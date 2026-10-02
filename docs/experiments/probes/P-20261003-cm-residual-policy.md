# P-20261003-cm-residual-policy

**Status:** `UNPROMISING` Probe; close this implementation route before PPO or a larger policy claim.

**Decision question:** Can a Cm policy improve the frozen HF15 rotation-Cup controller by adding a small object-frame translation residual for two native control steps, while preserving contact and mesh-clearance safety?

**Implementation:** branch `agent/cm-residual-policy`, final route commit `5bc3cb1` (design contract in `362621f`). The baseline is the Cup expert with trigger-anchored wrist orientation. The policy emits only a bounded object-local translation in `[-2,2] mm × [-2,2] mm × [-3,3] mm`; all other native action channels remain baseline-owned. Cm and actor weights are frozen in `cm_residual_policy.pt` (SHA256 `2dcefa6ee5ecaef4cc34c1a6ef7478ab16c7356e54f6a6ebe84cce99e660cfcc`).

**Source and fit:** Four native source attempts contributed 592 valid rows after rejecting partial/reset-contaminated or saturated windows. Fit/calibration/held buckets contained 263/156/173 rows. Three-model Cm and matched shuffled ensembles were trained from the same fit rows. Source execution used real Isaac Gym PhysX, 96 environments, 30 Hz, airplane motions, and the frozen six-expert bank.

**Native contrast:** Five formal native runs (`probe_s734` through `probe_s738`) produced 1167 complete windows. The held hash split (`bucket >= 70`, hash seed 12651) contains 103 baseline, 130 residual, and 130 shuffled rows across 21, 18, and 20 motion/start groups respectively. The held metrics are recorded independently in [the audit JSON](P-20261003-cm-residual-policy-results.json):

| arm | held rows | H10 object-height change (mm) | last-three-step joint contact | contact loss | clearance loss | executed residuals |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 103 | -5.18 | 0.670 | 0.466 | 0.845 | 0 |
| residual | 130 | 10.05 | 0.805 | 0.331 | 0.785 | 4 |
| shuffled | 130 | 8.30 | 0.651 | 0.485 | 0.831 | 4 |

The apparent residual-minus-baseline height difference is descriptive and unpaired. Only four held residual windows actually changed the native action; the rest fell back exactly to baseline because the ensemble margin, uncertainty, or OOD gate rejected the actor proposal. The shuffled arm has a similar descriptive height shift despite the same low action coverage, so these arm means cannot support a Cm utility claim.

**Gate and decision:** The row and motion/start support gates pass, but the predeclared residual coverage gate requires at least 48 changed held residual windows and only 4 changed. The result is `UNPROMISING` for this implementation route. Do not start PPO, do not promote this to a supported scientific claim, and do not interpret the arm means as evidence that residuals improve the strong controller.

**Next decision:** Treat fail-closed coverage as the blocker. If this route is revisited, first run a cheap policy-coverage diagnosis on held states (actor output scale, calibration margin, and OOD distribution) and require an independently justified coverage target before another native comparison. The current route supplies native wiring and risk accounting but does not establish Cm residual utility.
