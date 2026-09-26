---
schema: ref2dex.probe.v2
probe_id: P-20260926-trajectory-credit
date: 2026-09-26
branch: agent/cm-trajectory-credit
git_commit: 5475f597348eff79207a0b7810e0e85132247cca
claim_id: C3
hypothesis_family: HF04
probe_index_in_family: 1
seed_pool: probe
status: PLANNED
classification: Decision
---

# Probe: short trajectory-level contact credit

## Decision question

HF02 tested candidate expert selection from history and HF03 tested an
immediate post-action handflow token. This independent hypothesis asks whether
an observed **five-step contact/handflow trajectory** after a randomized
contact-stage action carries incremental information about the later
first-episode held-lift outcome. The intended use is a training-time critic or
credit assignment signal after an option executes; the trajectory token is not
available for choosing the initial action.

This is a CPU-only screen using four already completed three-arm Cm-off
collections. It does not start a simulator, collector, PPO run, online Probe,
or GPU process. It is a new family and does not consume HF02 or HF03 slots.

## Frozen substrate and provenance

All runs use the self-trained `source_e260` checkpoint
`16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`, the
airplane source object, and motion manifest
`2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038`. The
accepted records contain only `motion_id=0`; no multi-motion or cross-object
claim is allowed. Each run randomizes `-0.1`, `0`, or `+0.1` wrist-z at a
contacting state with known propensity `1/3`, then resumes the source actor.

| split | simulator seeds | assignment seeds | expected valid rows |
| --- | --- | --- | ---: |
| fit | 250, 251 | 20260925250, 20260925251 | 63, 58 |
| holdout | 252, 253 | 20260925252, 20260925253 | 63, 63 |

Rows are accepted only when `pre_contact` and `intervention_valid` are true,
the five followup steps remain in the first episode, all final held-lift labels
are finite, and the executed action equals the assigned wrist-z dose with no
other action-coordinate change. Exact transition and manifest hashes are
recorded in the run manifest and result index.

## Frozen CPU comparison and gate

Fit seeds 250/251 and evaluate once on 252/253. Fit-only normalization and
ridge lambda `10.0` are fixed. The post-option trajectory token contains only
observations available after the option has executed:

* `next_q - q`, `next_object_state - object_state`, and `next_contact`;
* five-step `followup_object_state - object_state`,
  `followup_contact_count / 5`, and `followup_contact`.

Compare exactly:

1. `pre_action`: pre-action state, base action, and executed treatment delta;
2. `trajectory_credit`: `pre_action` plus the frozen trajectory token;
3. `action_shuffled`: treatment delta permuted once in the fit split;
4. `trajectory_shuffled`: treatment delta and trajectory token jointly permuted
   once in the fit split.

Targets are `final_lift_success`, `final_max_contact_lift_m` (millimetres),
and `final_contact_fraction`. The continuation gate is fixed before reading
the holdout result:

* at least 15 rows in each of the three arms in both splits;
* all fits and metrics finite;
* `trajectory_credit` improves held-lift Brier score and
  max-contact-lift RMSE by at least **5%** relative to both `pre_action` and
  `trajectory_shuffled`.

If the gate passes, write a new confirmation memo before designing one
trajectory-level critic integration Probe. If it fails, mark HF04
`UNPROMISING` or `UNCLEAR` and freeze it without changing seed, horizon,
regularizer, target, or token definition.

## Resource and stop contract

CPU only, two threads, <=20 MB output. Stop on provenance drift, incomplete
first-episode boundaries, nonfinite values, arm counts below the gate, or any
attempt to use post-option labels for an initial-action selector.

Implementation and tests are pinned at commit
`5475f597348eff79207a0b7810e0e85132247cca`:

* script SHA256:
  `d2b9af83a9f6dbe9312225c0317f3cb6a7af3ee895eff3c3d34f62cb1a3bd56b`;
* test SHA256:
  `9d09bb808ef69a73df19b9a25c712af55f30362b53efcdaa2bf2921214eff5fa`.
