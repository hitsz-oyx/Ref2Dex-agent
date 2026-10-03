# P-20261003-gate1-consequence-value — Gate 1 data and bridge readiness

**Type.** Decision Probe. The experiment asks whether an explicit physical
consequence value bridge can be learned before any whole-hand primitive, Cm
training, or distillation is attempted.

**Question.** Given the same current hand/object state and executed action, do
future object motion (`E`) and future hand/object interaction (`I`) reduce
held-out prediction error for exact simulator return-to-go `G`?

**Decision.** Compare the strict ablations `V_H`, `V_HEI`, and `V_HAEI` on
episode-grouped held-out data. A result is `PROMISING` only if the joint model
reduces the primary held-out error by at least 10% against `V_H` with an
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

**Resource boundary.** Probe uses one admitted idle GPU, a short collector run
and offline GPU model fitting only after the data contract passes. It does not
modify the external baseline worktree, overwrite checkpoints, or start Cm
online training. The output stays below the project experiment directory.

**Current state (2026-10-03).** The old records audit is `NOT_READY_FOR_GATE1`
(21 files, 1,924 rows; reward/E/I contract missing). A fresh physical smoke
completed on GPU6 with six complete episodes and 3,362 transitions. Its shard
contains exact reward, reward components, object root state, five contact-body
poses/quaternions, hand/object forces, and explicit E/I definitions. The
offline assembler produced 3,170 episode-grouped horizon-32 windows across six
episodes. These are contract and wiring evidence only; no model fit or Gate 1
classification has been run.

**Artifacts.**

* Collector: [`run_gate1_consequence_environment.py`](../../../scripts/run_gate1_consequence_environment.py)
* Assembler: [`assemble_gate1_dataset.py`](../../../scripts/assemble_gate1_dataset.py)
* Old-data audit: [`audit_gate1_data_contract.py`](../../../scripts/audit_gate1_data_contract.py)
* Smoke output: `src/task/CmResidual/research/contact_consequence/output/P-20261003-gate1-consequence-value/smoke_s86_retry3/`
* Assembled dataset audit: `gate1_dataset_h32_v2.audit.json` in that directory.

**Next action.** Run the grouped bridge fit only after a fresh data-volume
probe has enough independent episodes for the predeclared bootstrap. Report
the ablation table and leakage/shape audit before considering any Cm route.
