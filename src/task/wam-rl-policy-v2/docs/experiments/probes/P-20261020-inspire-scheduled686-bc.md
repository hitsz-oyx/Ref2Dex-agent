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
