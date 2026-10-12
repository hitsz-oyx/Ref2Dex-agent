---
schema: ref2dex.probe.v2
probe_id: P-20261012-goal-point-wam-ref22-r-horizon
experiment_id: P-20261012-goal-point-wam-ref22-r-horizon
date: 2026-10-12
task: goal-point-wam
branch: main
git_commit: 738ff2cc
claim_id: C3
hypothesis_family: HF-goal-point-wam-three-domain
probe_index_in_family: 29
seed_pool: probe
seeds: [290, 291]
decision_changed_if_positive: shorten the R execution window before GT-anchored control-aware WAM adaptation
decision_changed_if_negative: retain K24 and isolate R gradient ownership rather than expand WAM capacity
status: UNCLEAR
run_id: wam-ref2-r-horizon-20261012-r1
---

# ref2_2: GT-trajectory execution-horizon diagnostic

## Decision Note

User requests continuing the named session according to ref2_2. The prior
joint run improves early action fitting but loses true-tau R meaning. Test
whether reading all24 future frames contributes to the execution-interface
problem before another WAM run. This is a Decision Probe for the robot
subgoal of self-trained manipulation, not a Cm/RL causal comparison.

Cheapest discriminating experiment: keep WAM completely unchanged, reuse
actual audited expert futures, and train five R windows K=1/4/8/16/24 from
the SAME authenticated base weights, original six normalization buffers and
identical sampled update stream. Mask omitted futures AFTER normalization;
all historical parameter shapes remain identical, avoiding a simultaneous
capacity/initialization change. A short window means no current-control
dependence or gradient on later tau, verified by perturbation/gradient tests.

GPU2, fixed1000 updates per arm, batch128, AdamW1e-4/wd.01/clip1, normalized
current PD L1. Each batch16 startup nominal +48 later nominal +64 true t8
expert recovery. This matches the next anchored-R objective. One engineering
update before the fixed fit; no repeated seeds/checkpoint search. Stage<=600s,
new stage<=100MB (expected<40MB), cumulative ref2<=750MB, global400GB;
current outputs+tmp292.333GB. Stop on input drift, nonfinite, foreign GPU2,
deadline or resource boundary. No new branch or external writes.

## Inputs and development panels

Base R: competent-pd-fit-20261011-r1/tau.pt SHA414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f.
Nominal: authenticated competent-pd-teacher-20261011-r2, trainenv0:16/all519
ticks; fixed16-tick report/env24:28 and startup0:15. Testenv28:32 untouched.

Train recovery: startup-anchor-20261012-r1/native, actual role3 expert
current8..104/97 windows; complete24 causal expert controls, applied action
and PD reconstructed from CURRENT measured q. Report this as TRAIN.
Development-only transfer: recovery-r2/native t16 (current16..152/137),
aggcollect-r1/native t32 (current32..168/137). Fixed source hashes/root audits
2316/9768/3468, expert masks, actual applied/label equality and PD inversion
must pass. Original source actor-future prohibitions remain unchanged;
this card grants R-only true expert future use. No pure/mixed-policy futures.

All sources have already been observed; panels overlap in time and share
motion/reset. They are development diagnostics, NOT independent validation.
Report XYZ mean/p95, rotation, fingers and clipping for each panel and K.

## Predeclared next decision

Score=max XYZp95 across nominal/t8/t16/t32. A shortened arm is eligible only
with>=20% score improvement over identically adapted K24, nominalp95<=1.2x
K24, nominal<=30mm/startupall<=20mm and all panel rotation<=.15rad,
finger<=.3rad/clipping<1%. Among eligible select lowest score (tie shorter).
Otherwise retain24; this does not prove short interfaces cannot work.
Never deploy these diagnostic checkpoints or transfer old R physical
qualification. Next distinct experiment splits gradient ownership:
R trained ONLY from GT tau; predicted-tau action loss differentiates through
R to WAM with R parameters held constant. Pure current-PD rows update WAM
only. Fresh physics comes after offline retention.

## Limitations / future evidence

Warm-start masking does not compare optimally retrained smaller architectures;
K24 inherited original pretraining. Negative results apply to this cheap
adaptation recipe. Independent resets, matched architecture training, fresh
closed-loop coverage and formal Cm-on/off remain deferred evidence.

## Results

Result: **UNCLEAR** on the short-execution-interface hypothesis. Five matched
1000-update arms completed16.14s/GPU2/peak72.15MiB, buffers unchanged.
Engineering1 completed2.28s/no checkpoint;12 horizon/PD contracts PASS.

| K | nominal XYZp95 | startupall | t8 TRAIN | t16 dev | t32 dev |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 24.24 | 9.08 | 7.48 | 24.48 | 54.12 |
| 4 | 20.55 | 4.10 | 5.32 | 19.84 | 40.40 |
| 8 | 15.40 | 3.87 | 5.34 | 21.66 | 42.19 |
| 16 | 15.05 | 3.96 | 5.11 | 20.65 | 36.21 |
| 24 | 15.36 | 3.02 | 4.56 | 21.19 | 39.99 |

All values mm; all arms pass declared development retention. K16 worst-panel
improvement9.5%, below20% cutoff; K8 is not clearly superior. Decision: retain
K24 and perform GT-only R gradient anchoring, without expanding WAM capacity.
Masking a K24-pretrained model is a limitation; this result does not prove
execution horizons interchangeable on predicted trajectories or fresh physics.

Artifacts: `outputs/trajectory-policy/wam-ref2-r-horizon-20261012-r1/`
contains manifest/result, per-K checkpoints, sampled indices and raw PD arrays.
The smoke uses sibling `wam-ref2-r-horizon-smoke-20261012-r1/`.
