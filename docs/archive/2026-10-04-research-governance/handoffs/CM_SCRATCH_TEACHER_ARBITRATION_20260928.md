# Cm scratch teacher-arbitration handoff

TASK_ID: `T-20260928-cm-scratch-mechanism-design`
AGENT: `agent_cm`
STATUS: `HANDOFF_READY`

## Candidate and decision

`CM-SCRATCH-TA-20260928` is a from-scratch physical transition envelope.  It
uses pre-action observation plus a candidate expert action to predict signed
one-step object-local displacement and quantiles for five-step contact
retention.  The lower contact quantile certifies candidates; the candidate
with greatest certified displacement along the lift axis becomes the offline
teacher label for six-expert to one-student distillation.  The frozen C1
observation-router teacher label is both the Cm-off control and the fallback
when no Cm candidate passes; a static `source_e260` fallback is forbidden
because it would confound the comparison.

The decision is a teacher-label assignment before training, not an online
action rank, value critic, residual correction, or shared representation
auxiliary.  This separates it from HF01/HF05 and leaves the six-expert/student
substrate unchanged.  It does not establish Cm utility.

## Implementation and CPU evidence

* Added the pure-Python contract and fail-closed leakage/support checks:
  `src/task/CmResidual/scratch_teacher_arbitration_contract.py`.
* Added the design-only configuration:
  `src/task/CmResidual/configs/cm_scratch_teacher_arbitration_v1.json`.
* Added the decision record:
  `docs/archive/2026-10-04-research-governance/decisions/D-20260928-cm-scratch-teacher-arbitration.md`.
* `python3 -m py_compile` passed.
* CPU contract smoke passed with 12 synthetic schema rows (six fit and six
  holdout episodes), deterministic arbitration, fallback, and a deliberate
  post-action leakage rejection: `CPU_CONTRACT_SMOKE_OK 12 6 6`.

No training, GPU, simulator, online, PPO, collector, or new data was used.

## Identifiability limit

The design requires six-arm randomized candidate-expert actions at matched
pre-action states, propensity `1/6`, and episode-disjoint fit/holdout.  The
existing HF03/HF04 wrist-z support is insufficient for this candidate, so the
route is **CPU-contract ready but training/Probe blocked until a new support
manifest exists**.  Do not infer a policy result from the smoke.

## Execution spec for `agent_rl`

1. Prepare (outside this task) the six-arm randomized transition manifest with
   one-step object displacement and five-step contact targets; do not reuse a
   Cm checkpoint or old representation.
2. Run the contract and CPU calibration/coverage gate.  Stop on missing arms,
   split overlap, target leakage, or failed calibration.
3. If the gate passes and root accepts the next budget, run matched offline
   distillation for `cm_on`, fixed-route `cm_off`, and fixed-seed placebo with
   identical student architecture, optimizer, states, steps, and seeds.
4. Return metrics as `PROMISING`, `UNPROMISING`, or `UNCLEAR`; held-lift
   validation needs a separate Decision Checkpoint.

## Code identity

BRANCH: `agent/cm`
HEAD: `a0ecee73db7bc7bf9ddb6b9016d028ba16943237` (unchanged; artifacts are
working-tree additions and are not committed by this task)

NEXT: root decides whether to authorize support-manifest preparation and a
CPU calibration gate; no GPU/online/PPO/collector action is requested here.
