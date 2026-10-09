schema: ref2dex.probe.v2
probe_id: P-20261009-gpu-group-value-noise
experiment_id: P-20261009-gpu-group-value-noise
date: 2026-10-09
task: consequence-evaluator
branch: main
git_commit: 4258adf
claim_id: C3
hypothesis_family: HF-native-gpu-execution-contract
probe_index_in_family: 6
seed_pool: probe
seeds: [282]
decision_changed_if_positive: retain a native GPU synchronous group as a noise-calibrated engineering container and proceed to a bounded Value-space Probe
decision_changed_if_negative: close role-permutation noise calibration and keep Gate1 blocked pending a different replay contract
status: UNPROMISING
run_id: gpu-group-value-noise-20261009-default-r1, gpu-group-value-noise-20261009-swapped-r1
---

# Native GPU synchronous group Value-noise calibration

## Decision Note

Strict fresh-process bitwise replay is unavailable in the native GPU contact
regime, while CPU and host tensor backends change the policy behavior. The
remaining actionable question is narrower: can a single native GPU process
produce a stable scalar-
`Y` contrast when its zero-pair noise is measured in the same launch?

The previous packets were not sufficient for this decision. The runner used
fixed env labels even when a non-default zero pair was selected, and the
offline Value audit consequently treated one old multi-zero packet as if env0
were the baseline. This Probe uses the corrected role map and reports rare
contact spikes with p99, peak tick, and first-nonzero tick diagnostics.

This is an engineering Probe. It does not relax strict Gate1, does not train
an evaluator, and does not change the physical reference bank, TCC encoder,
`Y`, checkpoint, or policy weights.

## Hypothesis and decision rule

For a shared native prefix through tick 48, the candidate contrast in physical
bank progress should be interpretable relative to zero-zero progress noise in
the same process. A positive result requires:

1. the prefix controls and semantic initial state pass the existing group
   contract;
2. the packet's explicit role map reproduces the intended baseline/zero/
   positive/negative labels under both role layouts;
3. the scalar candidate contrast is reported with at least two zero roles and
   is not hidden by a rare-contact p95 artifact; and
4. the baseline role remains a valid native behavior reference. The existing
   full-horizon native GPU group behavior packet is the behavior reference;
   the new short windows are not treated as full-horizon success evidence.

If both role layouts show a consistent candidate-vs-zero `Y` contrast and no
new contract error, continue with one bounded rolling engineering Probe. If
the contrast changes with the role map, zero-pair noise dominates, or the
packet contract fails, stop this synchronous noise route. In either case do
not call the result a Gate1 pass or a causal same-state counterfactual.

## Frozen execution contract

- native `gpu_physx_gpu_pipeline` (GPU PhysX, GPU tensor pipeline, GPU actor);
- seed 282, current self-trained checkpoint and one repeated motion;
- one process, four environments, 256 actor rows (`4 x 64`);
- tick 0--47 shared controls; tick 48--71 shared baseline plus the fixed
  positive/negative residuals;
- default role map: zero pair `[0,1]`, candidates `[2,3]`;
- permutation role map: zero pair `[2,3]`, candidates `[0,1]`;
- two bounded 72-step launches on GPU2, no full Gate1 scoring;
- outputs under `outputs/consequence-evaluator/`, with no videos or new
  checkpoints.

## Results

The default layout completed in 18.0 s. Its short-window baseline reached
`0.2585 m` and held 15 frames; this is not a full-horizon behavior result.
The corrected role map was
`baseline=0, zero_repeat=1, positive=2, negative=3`. Physical-bank `Y` over
the query window was `[0.043592, 0.043215, 0.022880, 0.024176]`; the zero-pair
absolute median was `0.000378`, and positive/negative contrasts were
`-0.020713/-0.019416`.

The exchanged layout also completed in 18.0 s. It used
`baseline=2, zero_repeat=3, positive=0, negative=1`; its short-window baseline
reached `0.1530 m` and held 9 frames. `Y` in environment order was
`[0.039070, 0.020899, 0.035543, 0.038167]`, which maps to baseline/positive/
negative contrasts of `+0.003527/-0.014644`; the zero-pair median was
`0.002624`. The positive contrast therefore changed sign and magnitude under
the role layout, while the negative contrast remained harmful but also changed
scale. Neither packet passed the strict mechanical candidate gates, and neither
entered the Gate1 scorer.

The new rare-event diagnostics exposed the same contact-stage issue in both
layouts: raw native contact-force zero-pair differences first appeared at
tick 44, while flattened p95 was `0`; max/p99 were approximately `8.1/0.59`
and `26.2/1.36` respectively. This confirms that p95-only summaries are not a
safe acceptance criterion for reactive contact execution.

The old r18 multi-zero packet is not re-used for this comparison: it has a
non-default zero pair but no role map, so the corrected audit rejects it rather
than silently assigning env0 as baseline.

## Decision

The role-permutation noise-calibration route is **UNPROMISING** for scalar
Value identification at this execution contract. The native GPU synchronous
group remains a useful behavior container (the earlier full r19 packet still
holds `0.8262 m/481`), but these two short launches do not justify a rolling
candidate Probe or a Gate1 claim. Keep the physical reference bank, TCC, `Y`,
checkpoint, and policy weights frozen. Do not train an evaluator or enter
PointWorld from this result; a future attempt needs a new execution contract or
an explicitly changed Gate1 estimand.

## Limits

Two short launches at one seed can only establish engineering identifiability
or expose a role/launch confound. They cannot estimate a population effect,
prove strict same-state equivalence, establish task utility, or replace the
formal Gate1 contract.
