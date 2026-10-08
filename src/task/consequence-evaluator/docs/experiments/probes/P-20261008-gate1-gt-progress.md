---
schema: ref2dex.probe.v2
probe_id: P-20261008-gate1-gt-progress
experiment_id: P-20261008-gate1-gt-progress
date: 2026-10-08
task: consequence-evaluator
branch: main
git_commit: 64522f1
claim_id: C3
hypothesis_family: HF-consequence-gate1-gt-progress
probe_index_in_family: 1
seed_pool: probe
seeds: [282, 283, 284, 285]
decision_changed_if_positive: expand fresh-replay GT-value rolling comparison before fitting an evaluator
decision_changed_if_negative: distinguish candidate coverage from phase-value utility before spending on later Gates
status: UNCLEAR
run_id: gate1-gt-progress-20261008-r1
---

# Does causal physical-reference GT value improve real rolling control?

Result: One matched pair completed in314.94s; all exact replay checks passed.
One of four decisions intervened. Baseline/rolling0/1, rescues0, harms0.
CPU baseline maxlift0.74cm versus GPU preflight81.51cm/483held frames.
Decision: UNCLEAR. Stop CPU panel expansion and evaluator fit; restore
representative GPU execution before deciding GT policy utility.

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
controlled baseline. Same543state full reference and native30Hz/PhysX60Hz. Initial research run
uses CPU-PhysX for both arms and GPU64row actor/phase inference, after GPU
force/history replay failed. This is not a comparison with oldGPU success rates.
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


Contact-stage exact replay remains an engineering blocker after the inference
repair. R3/r4 force cache diverges at tick44 despite identical physical tensors,
obs and RNG; an explicitly engineering-only inspection finds different contact
flags in history by tick47 (maxhistory1), so it is not admitted as harmless
rounding. R5 explicit GPU completion barrier also fails tick44 and is abandoned.
Strict research replay remains unchanged; diagnostic traces are rejected by
GT scoring and cannot become value examples.

Decision Note: Try a<=180s single-environment CPU-PhysX exact-replay preflight
with the same GPU64row actor, fixed30Hz/60Hz and known controls. GPU contact
readback fails the required same-state contract, which is the concrete reason
for testing CPU simulation; neural inference still uses GPU. If CPU passes,
record this backend explicitly for both arms of the bounded initial Gate1 run;
it is not a comparison against the historical GPU success rates. If CPU fails,
stop these repeated preflight runs and record the remaining replay blocker
before redesigning execution. No tolerance relaxation or policy/label training.


R6 CPU-PhysX/GPU64row actor preflight completed: independent72step baseline
and zero candidate (48prefix+24future) have bitwise equal73canonical-state
hashes,73RNG hashes, all72controls and measured traces, including contacts.
Each worker took~10.5s. Audit: outputs/consequence-evaluator/
gate1-engineering-20261008-r6/exact-replay-audit.json. This is engineering
readiness for the first window, not whole-episode replay evidence or Gate1 pass.
Initial scientific campaign remains one seed282/four replans under900s; every
later chosen prefix, repeat-zero and full continuation must pass the same checks.


## Completed matched pair and attribution

Actual executing commit64522f1, run_idgate1-gt-progress-20261008-r1, seed282,
CPU-PhysX and GPU2 actor/phase inference. Completed in314.936s and GPU released.
All12candidates, repeated first-zero, each chosen8step mixed-prefix replay and
full542step rolling continuation passed canonical state/RNG/control/measurement
checks. Every candidate at each query shares identical P[t]. No source drift,
evaluator, learning update or candidate evidence beyondt+24. The as-run frozen
protocol is archived in the output directory before updating this card.

| query tick | baseline Y | positive-finger Y | negative-finger Y | choice |
| --- | ---: | ---: | ---: | --- |
|48|0.018941|0.031013|0.016742|positive, +0.012072 over baseline|
|56|−0.001289|−0.036174|−0.001869|baseline|
|64|−0.014890|−0.015896|−0.013714|baseline, improvement<epsilon0.01|
|72|0.005263|−0.025038|0.004583|baseline|

Independent full-episode outcomes: baseline0/1, rolling0/1, rescue0, harm0.
Both maximum held runs are0 and intermediate loss counts0. Maximum elevation:
baseline0.007417m, rolling0.024780m, both below3cm. Final settled frames95/157
without a prior stable grasp do not count as task success. Increased maximum
elevation is not a success or utility claim. Single initial state, only four
early-contact decisions; this panel cannot adjudicate the method.

The restored GPU engineering baseline (same seed/checkpoint) had483consecutive
held frames and0.815144m elevation. CPU/GPU first controls agree within~2e−7
and initial history within9.54e−7, but first-step hand coordinates differ up
to3.40mm/history0.281, growing to contact-stage trajectory divergence. Hence
the CPU backend removes the exact-replay blocker while materially changing
policy behavior. This was not caused by weight/RMS updates; its physical/control
backend cause still needs diagnosis. The CPU matched pair is valid only for
that backend and is not negative evidence about the originally working GPU
policy or the new Y. No formal Gate1 pass andtraining_allowed=false.

Decision Note: This run changes the next investment decision: do not spend the
remaining registered seeds on CPU, and do not train an evaluator yet. Prioritize
restoring a representative native GPU baseline with a defensible same-state
GT rollout protocol, keeping weights and Y fixed. First cheap evidence should
compare identical executed controls across the contact-stage replay; stop on
uncertified decision-state drift. Candidate coverage, longer rolling stages and
formal success comparison remain deferred until execution readiness is restored.

Artifacts: outputs/consequence-evaluator/gate1-gt-progress-20261008-r1/
manifest.json, result.json, analysis.json, protocol-as-run.md, full baseline/
rolling packets and all branch/score logs. Historical GPU/CPU preflights remain
under gate1-engineering-20261008-r1..r6; no failed artifact was overwritten.
