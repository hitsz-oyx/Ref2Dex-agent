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


## Host pipeline engineering probe

The ref13 execution contract was restored explicitly on commit `8a18151`:
GPU PhysX (`sim_device=cuda:0`), CPU tensor pipeline (`pipeline=cpu`),
`physx.num_threads=1`, and GPU actor inference. Runs
`gate1-host-engineering-20261008-r1..r3` used seed282. The first two runs
preserve their startup/path failures; r3 is the accepted bounded result.

The r3 fresh 72-step baseline and fresh 48-prefix plus 24-step zero candidate
were exact across 73 states and 72 controls. State hashes, field hashes, all
Torch/Python/NumPy/CUDA RNG hashes, observations, geometry, history, native
contact forces, object contact forces, executed controls, and done flags all
matched. The audit is
`outputs/consequence-evaluator/gate1-host-engineering-20261008-r3/host-backend-audit.json`.

The contract did not preserve the one-environment policy behavior: host full
baseline reached only 0.154892 m and held for 8 frames, versus the matched GPU
pipeline reference at 0.815144 m and 483 held frames. CPU PhysX was still worse
(about 0.0074 m, zero held frames). This is an execution-distribution failure,
not evidence against the reference-bank Y value. No host trace enters Gate1
scoring or evaluator training.

### Decision Note

The current decision is whether `num_envs=1` is the source of the host behavior
change. The key evidence is exact single-environment replay together with a
large behavior gap, while historical ref13 used synchronous multi-environment
groups under the same host contract. Root therefore stops the single-process
Gate1 route and runs one bounded synchronous group replay probe, retaining
GPU PhysX, CPU tensor exchange, one PhysX thread, the same checkpoint and Y
contract. A group that restores behavior and gives exact within-process
baseline/zero prefixes can carry Gate1; otherwise the host route is closed and
the remaining investigation is simulator execution provenance. Cost is one
idle GPU and at most 180 seconds; no CPU seed expansion, training, or Y changes.

The synchronous group probe also failed the behavior gate. A four-environment
group completed 542 steps in 51.2 s: baseline reached 0.079110 m and held for
2 frames. A ref13-sized 96-environment group completed the bounded first 72
steps in 47.6 s and reached only 0.073761 m/3 held frames. In both groups the
baseline and zero-repeat controls were identical by construction, but the
parallel env instances were not bitwise twins: initial state fields differed,
sub-micro position/observation drift appeared immediately, and contact-force
drift reached about `5.7e-5` at tick44. The full packets and mismatch diagnostics
are in `outputs/consequence-evaluator/gate1-host-group-engineering-20261009-r6/`
and `...-r7/`.

This closes the host/group route for the current policy. The 96-env historical
execution size does not restore the original GPU baseline, and the observed
contact-stage drift still violates the same-state contract. No formal Gate1,
candidate scoring, evaluator fitting, or CPU seed expansion is allowed from
these traces. The remaining route is a fresh GPU execution-contract diagnosis
or a redesign that preserves the original GPU behavior; Y, reference-bank
labels, and policy weights remain frozen.

As a control, the same four-env synchronous harness with the original GPU
PhysX/GPU tensor pipeline preserved the policy behavior: 0.812772 m maximum
lift, 480 held frames, and first stable tick107. It still failed strict group
identity immediately (initial state fields and tick1 history differed, with
contact-stage differences at tick44), so this is not Gate1 readiness. Audit:
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r1/group-audit.json`.

## Native GPU group noise calibration

The host route is closed, so the next bounded engineering probe kept the
original `gpu_physx_gpu_pipeline` contract and moved all four roles into one
GPU process: env0 baseline, env1 zero repeat, env2 positive residual, and env3
negative residual. Env0's actor control was broadcast to every role through
the common prefix; only env2/env3 received their fixed ±0.2 residual on the
`t=48` transition through the next 24 action ticks. This makes the run a
solver/readback noise calibration, not a
formal candidate replay: process-global RNG is recorded once, and a
zero-pair is compared by measured state and observation fields rather than
declared bitwise identical.

Four fresh 4-env full-baseline controls show that the native policy itself is
not stable across these launches: r1/r2/r3 reached 0.8128/0.8239/0.7535 m and
480/484/484 held frames, while r4 reached only 0.1301 m and 7 held frames.
The latest r4 24-step calibration therefore cannot be interpreted as a
candidate result. Its candidate-effect-to-zero-displacement p95 ratios were
about 2.17/1.34 (positive/negative) for object pose, 2.87/1.80 for hand
points, and 2.46/2.76 for joint position, but only 0.38/0.36 for joint
velocity, 1.12/0.82 for object velocity, 0.68/0.57 for history, and
1.18/1.17 for contact force. The effect is consequently not separated from
the solver/readback noise across the state needed by a reactive policy.

The runner now records flattened native tensors after per-environment reshape,
explicit q/dq and contact fields, and rejects non-identical initial semantic
states (1e-7 tolerance) before computing calibration ratios. It also records
the query-relative zero displacement and both pre-query and post-query noise
ratios. Those checks make future calibration packets auditable; they do not retroactively
make the r1--r4 packets exact twins. The full r4 audit is
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r4/gpu-group-noise-audit.json`.

A follow-up 72-step worker on idle GPU2 (`r8`) completed in 18.2 s with all
initial semantic gaps exactly zero, exact env0/env1 controls, and the full
`t=48` through `t=72` state window present. It still showed the expected
world-frame hidden-state hash mismatch at tick0 and contact/trajectory drift
later in the group; the short baseline reached 0.2417 m and held 10 frames,
which is too short to judge full-episode behavior. The r8 packet and audit are
`outputs/consequence-evaluator/gate1-gpu-group-engineering-20261009-r8/`.
Two earlier direct-worker attempts (`r6`/`r7`) exited while saving a relative
mismatch path after the native process changed directory; those engineering
failures produced no simulation conclusion and were fixed by resolving
worker/prefix/score paths and creating the task-owned output parent.

### Decision Note

The synchronous native-GPU group does preserve the original behavior in most
fresh launches, but it does not provide a stable baseline or a uniform
candidate-over-noise margin. There is only one zero pair in this four-env
layout, so env2/env3's own pre-query drift is an additional limitation. Root
therefore does not run formal Gate1, rank
candidates, fit an evaluator, or change Y/reference-bank labels or policy
weights from this probe. The next implementation decision is between a
frozen ACT-like 24-step proposal baseline, which removes reactive action
amplification during the scoring window, and a new native GPU replay contract.
Until that decision is implemented and rechecked, the r1--r4 group packets are
engineering evidence only and the scientific status remains `UNCLEAR`.


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
