# P-20261018-unidex-live-oracle-tau

`task`: `wam-rl-policy`  
`status`: `RUNNING`  
`stage`: Probe  
`code_branch`: `main`  
`git_commit`: recorded in the run manifest  

## Decision

This probe distinguishes whether an oracle future hand plan helps action prediction when it is generated from the **same live simulator state** as the executor. A state-only policy and a state-plus-live-state-`tau` policy are trained and evaluated on matched rows; shuffled `tau` is a control. A positive identification gap justifies a receding-horizon closed-loop follow-up. A null gap stops this route for the current 11-point plan representation and redirects the next experiment to interaction context `C`.

## Protocol

The pinned UniDexGrasp++ state policy (`goal_cond=false`) is used as the official teacher. At each sampled live state, the simulator state is snapshotted, the teacher is rolled forward for `K=8` control steps, and the future 11-point hand landmarks are recorded. The oracle plan is

\[
\tau_t^{GT}(s_t)=H^{teacher}_{t+1:t+8}-H_t.
\]

The oracle branch is reconstructed by replaying the learner action prefix from one fixed post-reset state; it does not write an intermediate PhysX state back into the live branch. The same prefix is replayed again before applying the learner action. The first probe uses bounded deterministic action perturbation (`noise_std=0.05`) to obtain nearby learner states; it does not claim autonomous task success. Prefix replay error is recorded per row and rows above the replay tolerance are invalid.

## Matched identification

Inputs are the official 300-D observation alone or concatenated with the 8-step, 11-point, 3-D oracle `tau`. Targets are the official 24-D teacher actions from the same live state. The validation split is by environment index. The shuffled control keeps state and target matched while permuting `tau` across rows.

## Resource boundary and stop condition

One local RTX 3090, one CAR object, at most 8 environments, 32 sampled ticks, `K=8`, and a bounded GPU deadline. Stop before closed-loop rollout if fork/restore validation fails or if the identification probe produces no trustworthy rows. Outputs belong under `outputs/wam-rl-policy/P-20261018-unidex-live-oracle-tau/`.

## Limitations

This is an engineering Probe, not evidence of manipulation success. It uses one prepared TRAIN object and the official policy as a nearby-state teacher. A positive result requires a later receding-horizon physical validation; a negative result is only decision evidence for this plan representation after restore and data-contract checks pass.
