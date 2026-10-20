# P-20261020-inspire-scheduled686-bc

This probe warm-starts a Scheduled686 Inspire executor from the best Current421
checkpoint and adds only the newly generated spatial WAM tau plus validity.
It is paired with `P-20261020-inspire-scheduled686-cache`; the v4 checkpoint has
no trained CONTACT/RELEASE event input.

- `experiment_id`: `P-20261020-inspire-scheduled686-bc`
- `run_id`: `inspire-scheduled686-bc-20261020-r1`
- source: audited same-execution teacher rollout
- parent: Current421 best at step 3000
- model: 686 input, 12 output, 1024/1024/512/512 ELU
- optimizer: AdamW 3e-4, batch 2048, 50k steps

## Result (2026-10-20)

The run completed 50,000 updates on GPU6 in 1,546.9 s.  The best fixed DEV
command MAE was 0.036683 at step 1,000 (root 0.016646, finger 0.056720);
the final step was 0.052845.  The best checkpoint is retained for later
closed-loop comparison.  These are offline action errors and do not establish
that tau improves physical task success.


## Corrected training boundary in v3

The historical r1 initialization retained random tau/valid columns. Its small
DEV differences cannot be used as a clean WAM ablation. The repaired trainer
uses `initialize_current`, zeros all 265 new columns, and admits only shared
TaskController-goal layout v2. Step0 is saved as both best/latest before updates,
so the exact parent remains selectable if condition training degrades DEV.
Shape feature distribution diagnostics are logged at evaluation. Capacity,
optimizer and the 50k maximum remain the Shadow recipe; no new long BC run is
started as part of the engineering repair.


The real-parent regression at 94ce4175 used Current421 best step3000 and 256
active/new-tau-valid rows from eight TRAIN objects. Initial output max error was
2.384e-7 and all added weights were exactly zero. One engineering AdamW step
produced nonzero tau/valid gradients (max 0.014186/0.004477) and updated the new
columns; no checkpoint was saved. Parent normalized shape features had p99 2.48
and maximum 3.24 on this panel. This small panel does not exclude broader shape
shift or establish grasping. Report:
`outputs/wam-rl-policy-v3/inspire-current421-warmstart-audit-20261020-r1/report.json`.


## Corrected full-data retraining: r2

- classification: Decision Probe; distinguish whether repaired shared-goal tau
  can improve matched DEV imitation/closed-loop behavior over the exact parent.
- next decision: better DEV and physical progress supports a bounded policy
  improvement probe; a step0 win or no closed-loop progress prioritizes causal
  error/conditioning diagnosis before multi-hour PPO. No WAM attribution claim.
- cheapest test: reuse immutable admitted data and v2 tau layout; retrain only
  Scheduled686 from Current421 best, keep step0 in selection, then paired DEV and
  the same fixed TRAIN native panel. Do not retrain Current421 or recapture data.
- task: wam-rl-policy-v3; branch: main.
- run_id: inspire-scheduled686-bc-20261020-r2.
- git_commit: supplied by REF2DEX_CODE_COMMIT in the frozen run manifest.
- inputs: original measured admission, v3 shared-goal cache r2, Current421 best step3000.
- recipe: 50000 steps, batch2048, AdamW3e-4/weight_decay1e-5, seed247,
  hidden1024/1024/512/512 ELU, joint CAD encoder, gamma not applicable to BC.
- engineering: differentiable CAD encoding once per episode point cloud per
  minibatch and gather back to rows; no frozen shape feature cache, no change
  in loss/sample weighting. Output and all-parameter gradient regression required.
- environment: graspenv Python3.8/Torch2.4.1+cu121; GPU6 training, GPU7 evaluation.
- limits: 8h wall-clock, own STOP, bounded best/latest only; leave other GPUs free.
- outputs: outputs/wam-rl-policy-v3/inspire-scheduled686-bc-20261020-r2/.
- evaluation: paired DEV5693 rows, followed by deterministic native controls;
  this small TRAIN panel cannot establish unseen-object generalization.
