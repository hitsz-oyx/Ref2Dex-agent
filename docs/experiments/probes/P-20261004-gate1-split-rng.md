# P-20261004-gate1-split-rng

Classification: **Decision Probe**, not Validation. Status: completed; `UNCLEAR`
for Gate closeout and `PROMISING` only as a local h16 diagnostic.

Question: Does the negative I+ h16 split 3 persist when the model/minibatch RNG
changes while train/test episodes stay fixed? Previous runs changed split and
model seed together. This prevents attribution of their variability.

Decision: if split 3 stays negative and split 1 stays positive across the same
model seeds, prioritize outcome coverage/distribution diagnostics; if signs vary
within a fixed split, investigate fitting stability before collecting more data.
Neither outcome closes the representation hypothesis or permits Cm training.

Cheapest discriminating design: reuse `tmp/e260_all4_h16_i_aug.pt` (112 episodes),
fixed h16, 30 epochs, batch 512, existing six ablations plus future-action
controls. Cross split seeds 20261221/20261223 with model seeds 314159/271828.
Only model/minibatch seed is newly independent; default seed behavior is unchanged.
Record normalized return/aux training losses per epoch to diagnose fitting.
No epoch or seed is selected using test performance.

Run IDs: `split1_model314159`, `split1_model271828`, `split3_model314159`,
`split3_model271828`. Outputs: `tmp/gate1_split_rng/<run_id>.json` and `.log`.
Code identity: commit containing this card and independent model-seed CLI.
Input identity: input SHA256 recorded in `tmp/gate1_split_rng/input.sha256`.
Resources: GPUs 1/4/5/6, at most four concurrently; target <30 minutes and
<100 MB extra artifacts. Stop on nonfinite losses or any input/resource conflict;
do not stop unknown GPU processes. No new simulation, checkpoints, or policy work.

Preflight: original fit train/test MAE (V_H / V_HEI) on split 3 is
3.329/32.654 and 1.500/38.697; split 1 is 5.112/15.419 and 2.467/10.936.
No zero-variance training feature columns were found. Split 3 held-out H has
absolute standardized values up to 828.9 and I+ up to 120.5; split 1 H reaches
30.1. These are distribution diagnostics, not proof of a numerical bug.
Tiny CPU synthetic regression smoke gives bitwise identical old/new predictions
with the default seed and finite per-epoch losses. CPU is used only for this
32-row engineering smoke and tensor statistics.

Result: with the original data and fixed splits, changing only the model seed
kept split 1 positive (`+35.9%`, `+45.0%`) and split 3 negative (`-20.8%`,
`-14.1%`). The disagreement is therefore not explained by initialization.
The split-3 held-out episode `(source_run=2, episode_id=18640000018)` has
`stable_success=1`, `drop_after_success=1`, return-to-go mean `805.6`, and
drives a `-163.0`/`-139.1` MAE delta in the two model-seed fits. Split 1 is
instead helped by episode `(2, 18640000020)` (`+109.7`/`+111.9` before the
history repair). This is outcome coverage sensitivity, not a stable Gate result.

The same probe also exposed and repaired a separate history contract bug: the
assembler had zeroed the oldest preceding action in 99.8% of windows. After
preserving factual preceding actions, the five h16 I+ direct changes were
`+37.8%`, `+20.6%`, `-28.2%`, `+21.2%`, `+6.2%`; matched future-action controls
were `+38.4%`, `+18.0%`, `-20.7%`, `+18.1%`, `+14.3%`. The corrected dataset
passed audit, but split 3 remained negative and outcome sensitivity persisted.
The corrected direct and control cluster-bootstrap diagnostics are archived in
`tmp/gate1_split_rng/histfix_i_aug_actor_cluster.json` and
`histfix_i_aug_ctrl_actor_cluster.json`; they are not formal Validation.
