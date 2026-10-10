---
schema: ref2dex.probe.v2
probe_id: P-20261010-serial-exact-h-candidate-bank
experiment_id: P-20261010-serial-exact-h-candidate-bank
date: 2026-10-10
task: consequence-evaluator
branch: main
git_commit: d25f043
claim_id: C3
hypothesis_family: HF-trajectory-conditioned-evaluator
probe_index_in_family: 8
seed_pool: probe
seeds: [421]
decision_changed_if_positive: retain an exact-prefix candidate-bank route and fit a fresh offline selector/evaluator panel
decision_changed_if_negative: close serial exact-H candidate collection and keep selector/PW/native integration frozen
status: RUNNING
run_id: serial-exact-h-candidate-bank-20261010-r1
---

# Can serial deterministic replay create an exact-H, non-tied candidate bank?

## Motivation and decision note

The observed-τ T-only selector has a useful offline signal, but the current
history panel only matches H approximately. The next mainline blocker is to
obtain candidates with the same query-time state before fitting or selecting.
The earlier serial zero replay showed that one environment, one seed, and the
same command prefix can be bitwise deterministic. This Probe uses that contract
to replay one prefix seven times and apply a bounded contact-stage action
perturbation only after tick 120.

The discriminating question is whether exact visible prefix equality is enough
to produce a useful candidate bank: positive means preserve this serial
collector and fit a fresh held panel; negative means serial action perturbations
do not create the required consequence diversity and the selector route remains
data-blocked. This is not a hidden-PhysX fork or a native success experiment.

## Protocol

All seven runs use the frozen `hand-execution-inputs-20261009-r1/inputs.json`,
seed 421, `envs=1`, CPU tensor exchange with GPU PhysX, full reference frame 0,
no initial jitter, and actor observations saved. Candidate 0 is zero residual;
candidate pairs are 32-tick pulses at ticks 120--151:

| candidate | coordinate | residual |
| ---: | ---: | ---: |
| 0 | — | 0 |
| 1 | finger 6 | +.08 |
| 2 | finger 6 | −.08 |
| 3 | finger 8 | +.08 |
| 4 | finger 8 | −.08 |
| 5 | finger 10 | +.08 |
| 6 | finger 10 | −.08 |

The requested action is always checked against the captured native action,
with zero clipping and complete 543-state/542-command traces required. The
prefix gate compares `actor_observation`, hand, q/dq, object pose/velocity,
action, and native PD target before tick 120 across all candidates. The panel
screen then reports candidate hand/object spread, pair/contact proxy coverage,
and U32 labels without using labels to compose candidates. The minimum screen
is exact prefix equality plus at least four distinct candidate trajectories and
at least two non-tied label rows.

## Scope boundary

Even a positive result is only an exact-prefix engineering/data Probe. It does
not clone hidden PhysX state, prove a counterfactual causal effect, validate R,
or authorize C1/C2/PW, online selector, MPC, or native success. A fresh
episode-split panel and a matched evaluator audit would still be required.

## Artifacts

Runs are under
`outputs/consequence-evaluator/serial-exact-h-candidate-bank-20261010-r1/`.
The runner is the frozen
`src/task/consequence-evaluator/tools/run/run_hand_bridge_rollout.py`; the
post-run panel audit will record all input and trajectory hashes.

## Result

Pending the seven serial rollouts and the prefix/candidate audit.
