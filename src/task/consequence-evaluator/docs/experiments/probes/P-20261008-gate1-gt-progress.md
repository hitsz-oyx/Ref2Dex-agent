---
schema: ref2dex.probe.v2
probe_id: P-20261008-gate1-gt-progress
experiment_id: P-20261008-gate1-gt-progress
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: 6842da0
claim_id: C3
hypothesis_family: HF-consequence-gate1-gt-progress
probe_index_in_family: 1
seed_pool: probe
seeds: [282, 283, 284, 285]
decision_changed_if_positive: expand fresh-replay GT-value rolling comparison before fitting an evaluator
decision_changed_if_negative: distinguish candidate coverage from phase-value utility before spending on later Gates
status: PLANNED
run_id: gate1-gt-progress-20261008-r1
---

# Does causal physical-reference GT value improve real rolling control?

Result: Protocol and runner implemented; real native replay checks pending.
Decision: Run bounded early-contact Gate1 Probe using the owned self-trained actor.

## Decision Note and purpose

User full chain requires same-state candidates, direct GT24step future scoring,
rolling8step execution and independent complete-episode outcomes before fitting
an evaluator. ref4_2 physical-bank Probe fixed clean local regressions and passes
its small exposed development panel. That proves a route worth testing, not
candidate value or policy utility. This Decision experiment distinguishes whether
signed reference advancement helps actions under real executed mixed history.
If rescue exceeds harm, expand a fixed matched panel; if no signal, inspect
candidate coverage/scoring before extending training or changing primary GT.
Do not claim a formal Gate1 pass from four episodes.

## Frozen protocol and budget

Actor: owned self-trained airplane_base e260 checkpoint, random-scratch ancestry,
Cm off; the official actor only generated bank references and is not the
controlled baseline. Same543state full reference and native30Hz/PhysX60Hz.
References/encoder are completed physical-reference-bank-20261008-r1 and
physical-bank-tcc-20261008-r1, with fixed uniform bank weights and unchanged
matching parameters. No evaluator, actor updates, endpoint value or PPO critic.

Registered seeds282/283/284/285; initial campaign uses seed282 only after
engineering timing (~48s full baseline/~15s short worker) predicts four seeds
would exceed900s. Expand after this cheapest decision signal, with remaining
seeds in a later bounded run. One native environment per fresh process. Exactly four
replans at full-episode ticks48/56/64/72; each candidate lasts24steps for scoring,
first8controls executed, then replan. After the fourth query, continue the same
actor to full episode end. This is an early-contact intervention Probe, not
whole-episode MPC. Baseline runs full episode without residuals. Fixed candidates:
zero, +0.2 and −0.2 on independent finger channels6/8/10/12/15; coupled distal
and wrist channels are zero. This is a feedback controller plus a decision-known
residual schedule, not a known native24action/hand-flow chunk. ACT-like proposal
is a later gate, as in the user's full-chain supplement.

Y=P[t+24]-P[t] uses the complete actually replayed prefix and candidate measured
object/11hand future, stopping at t+24. Same current P for all candidates.
Choose maxY only if it beats baseline by epsilon0.01; otherwise baseline. No
future-driven confidence threshold, episode outcome selection or success oracle.
No baseline shortcut using an uncertified upper bound. Zero candidate at the
first query must exactly match independent baseline; repeat that zero branch.

Every candidate starts a new process/new simulator, replaying all already
executed native controls from frame0. At every replay tick check declared canonical native
state inventory, Torch/Python/NumPy RNG, observations, measured geometry and
executed controls. Commit only chosen first8controls; next fresh worker actually
executes them and verifies all their states against the chosen branch. The
terminal rolling worker reexecutes the full mixed prefix before continuing to
end. This is not joining prerecorded suffix states. Preserve original simulator
configuration, deterministic actor/RMS, fixed geometry sampling and source hashes;
check60Hz/decimation2 and simulator frame count at every tick, prefix done
signals, and actual physics/controller/RMS identities once per fresh worker.
This inventory is not every hidden PhysX solver state; exact zero-repeat and
full prefix replay are the engineering guard. No mid-state restore. Abort on any mismatch, source drift, early done or budget.

Independent episode metric: measured surface proximity<=1cm, unsupported lift
>=3cm and45consecutive held frames; final tabletop footprint/support<=2cm with
15settled frames (linear<=0.05m/s/angular<=0.3rad/s). Placing after the most recent
stable grasp must avoid6unsupported loss frames or unheld fall faster than
0.25m/s. Prior intermediate losses can be recovered by a later stable grasp;
count them separately. These are weak geometric proxies, not exact contact-pair
or human-confirmed success. Neither P nor Y enters episode success.

One idle GPU (initialGPU1, fallbackGPU2 if occupied), <=900s campaign, <=1GiB artifacts,20GiB free disk reserve. CPU for
orchestration/hash/stat/outcome only; native simulation and phase inference use
GPU in separate sequential processes. Engineering preflight gets separate bounded
outputs, no scientific utility conclusion. Monitor GPU/process state and worker
ETA; only task-owned subprocesses can be terminated on deadline. Unique outputs,
no checkpoint overwrite, no remote push. Stop if a foreign GPU task appears.

## Interpretation and future evidence

Record baseline/rolling successes, rescues and harms, chosen-candidate counts,
intermediate loss events, every exact replay check and distinct initial state
count. No interventions chosen means limited headroom under this candidate bank,
not a negative result for learned evaluators. Duplicate initial states or a
homogeneous outcome panel reduce effective coverage. Four seeds and one motion
are not formal scientific support. Later matched multi-seed/broader stage controls,
ACT chunk candidates, arbitrary-state calibration, delayed hazards and formal
validation are deferred until this minimal Probe changes a decision.


## Engineering failure and repair (not method evidence)

At commitc77aeaf, native single-environment r1 completed a baseline but emitted
incorrect GPU policy means; zero-prefix replay failed at tick45. This run is
invalid as a policy/value result and remains under gate1-engineering-20261008-r1.
The actor/RMS/checkpoint identities match the previously working64env generator.
A hooked24step r2 and CPU recomputation verify matching raw/normalized inputs
and unchanged weights; native GPU outputs disagree sharply with CPU. A direct
archived Torch2.0.1 GPU reproduction without Isaac shows batch1 wrong, batch64
matching CPU and historical baseline controls. This identifies the inference
batch path as the actionable defect; the underlying GPU/library cause remains
unassigned. Do not modify RMS, retrain actor or treat this as a Y failure.

Repair: fixed64identical observation rows in deterministic nonrecurrent
get_action, retain first command for the single simulator. All candidates and
baselines share this adapter and its64row native model/RNG calls. Verify actual
control against CPU/manual checkpoint forward and fresh full-prefix replay
before research control. CPU is used only for the tiny one-frame engineering
reference calculation; all simulator and campaign inference remain GPU.
