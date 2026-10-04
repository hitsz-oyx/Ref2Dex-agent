# Decision: test available nominal-motion predictors on fresh interventions

- **Question:** Is the archived nominal-motion advantage responsive to changes
  in action, or primarily factual state/motion correlation?
- **Evidence:** The primary predicted-motion screen failed. Its separately
  reported nominal-motion control improves near factual RMSE11.18%over commands.
  Earlier pulse prediction failed and its force-proxy trigger included distant
  hand/object states. Generic actuation factorization is already established.
- **Action:** Freeze the existing nominal/raw/state models and collect fresh
  cold-start reference/replay interventions, using current geometric proximity
  plus force proxies to acquire interacting states. Test plus/minus physical
  contrasts, without refitting or selecting models on these new responses.
- **Cost/stop:** One idle GPU,<=60min total,<=4GiB output. Stop for insufficient
  acquisition coverage, input drift, unmatched pre-intervention states or errors.
  Keep the old failed gate. Positive causal transfer justifies a new robust
  action-ranking design; failure ends this direct factual-model route.
- **Authorization:** Inside the user's independent worktree and existing compute
  authorization. No original-worktree changes, policy training or external writes.

This changes the next experiment, not the research objective or final claim.
