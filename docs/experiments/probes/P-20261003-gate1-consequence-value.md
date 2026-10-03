# P-20261003-gate1-consequence-value — Gate 1 data and bridge readiness

**Type.** Decision Probe. The experiment asks whether an explicit physical
consequence value bridge can be learned before any whole-hand primitive, Cm
training, or distillation is attempted.

**Question.** Given the same current hand/object state and executed action, do
future object motion (`E`) and future hand/object interaction (`I`) reduce
held-out prediction error for exact simulator return-to-go `G`?

**Decision.** Compare the strict ablations `V_H`, `V_HEI`, and `V_HAEI` on
episode-grouped held-out data. A result is `PROMISING` only if the joint model
reduces the primary held-out MAE by at least 10% against `V_H` with an
episode-cluster bootstrap interval excluding zero, and the effect does not
disappear after controlling for action residual/noise. Otherwise classify the
Probe as `UNPROMISING` or `UNCLEAR`; this Probe cannot establish Cm policy
utility.

**Target.** `G_t` is the exact Monte Carlo return-to-go from recorded simulator
reward with `gamma=0.99`, terminated at the recorded episode boundary. No
bootstrap target is used. The dataset also retains reward components and
episode outcome auxiliaries (stable success, drop-after-success, hold time,
lift and contact fraction) for secondary heads; they do not replace `G_t`.

**Inputs.** `H` is the current state/history (`state`, previous action and
collector context). `E` is a fixed future window of object-relative pose,
quaternion, linear velocity and angular velocity. `I` is the same future
window of contact-body poses/quaternions relative to the object, finite
difference relative hand velocity, measured hand forces and measured object
force. Pair identity is not claimed when the simulator does not provide it.
Each sample carries an explicit future mask and an episode identifier.

**Data protocol.** Collect transitions from the self-trained P0 actor in the
real DExplore physics environment at multiple decision times. Per-episode
bounded residual perturbations are recorded (`noise_std`), and all candidate
windows are split by episode so future frames cannot cross a split. Existing
fixed-option `records.pt` data is excluded: its audit found no exact reward or
explicit E/I definitions and therefore it cannot answer this Gate.

**Resource boundary.** Probe uses one admitted idle GPU for collection and up
to four admitted idle GPUs for independent offline fits after the data
contract passes. It does not
modify the external baseline worktree, overwrite checkpoints, or start Cm
online training. The output stays below the project experiment directory.

**Current state (2026-10-03).** The old records audit is `NOT_READY_FOR_GATE1`
(21 files, 1,924 rows; reward/E/I contract missing). A fresh physical smoke
completed on GPU6 with six complete episodes and 3,362 transitions. Its shard
contains exact reward, reward components, object root state, five contact-body
poses/quaternions, hand/object forces, and explicit E/I definitions. The
offline assembler produced 3,170 episode-grouped horizon-32 windows across six
episodes. The larger probe then produced 21,004 windows across 41 episodes and
passed the same audit. Four GPU-parallel grouped split fits were run with the
same fixed architecture and 40 epochs. `V_HEI` versus `V_H` held-out MAE
relative changes were `-41.6%`, `+9.5%`, `-4.9%`, and `-4.0%`; the positive
case did not have a bootstrap interval excluding zero. `V_HAEI` was similarly
unstable. The predeclared 10% plus nonzero-CI gate therefore classifies this
implementation/data recipe as **UNPROMISING** for Gate 1. This is not a claim
that privileged interaction information is universally useless.

**Artifacts.**

* Collector: [`run_gate1_consequence_environment.py`](../../../scripts/run_gate1_consequence_environment.py)
* Assembler: [`assemble_gate1_dataset.py`](../../../scripts/assemble_gate1_dataset.py)
* Old-data audit: [`audit_gate1_data_contract.py`](../../../scripts/audit_gate1_data_contract.py)
* Smoke output: `src/task/CmResidual/research/contact_consequence/output/P-20261003-gate1-consequence-value/smoke_s86_retry3/`
* Assembled dataset audit: `gate1_dataset_h32.audit.json` in that directory.
* Bridge fit reports: `gate1_value_bridge_seed20261004_gpu4.json` through
  `gate1_value_bridge_seed20261007_gpu7.json` in the probe directory.

**Data-coverage diagnostic.** A separate self-trained `plain_off` e420
checkpoint was evaluated without mixing it into the pinned-source result. Its
actor-only smoke had 2/6 historical five-step holds but 0/6 stable successes.
The explicitly marked diagnostic collector then produced 14,081 transitions,
26 episodes, 1 stable success and 1 drop-after-success. Four parallel bridge
fits gave `V_HEI` relative MAE changes of `-25.5%`, `-6.9%`, `-11.3%`, and
`+1.6%`; no split met the predeclared gate. This does not rescue the route and
does not turn the diagnostic checkpoint into the pinned P0 source.

**Next action.** Freeze this exact bridge recipe and do not spend Cm training
budget on it. Close this Gate 1 route for the current actor/data distributions.
Any future change to target, actor/data distribution, or interaction encoding
requires a new Decision Memo and a separately identified Probe with a matched
Cm-off policy-utility plan.
