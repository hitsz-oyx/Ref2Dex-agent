# Decision: test task utility of the simple randomized factual control

Question: direct contrast learning fails; should we abandon predictive control
or cheaply test the independently trained factual control's task value?

Evidence: fixed direct learner fails all five gates on3072fresh windows.
Post-hoc factual-minus-zero/global risk differences are-3.436/-3.158mm²,
upper95%bounds-1.106/-1.222, negative in all three fixed optimization seeds.
More factual updates show no robust incremental gain. These secondary comparisons
were selected after the main screen and remain exploratory.

Action: end the failed direct learner. Choose the simpler1000-update factual
control with an equal-weight three-seed prediction rule; freeze a new randomized
closed-loop first-episode task experiment. A short-horizon predictor must earn
retained-success benefit before further method/PPO investment. This is a
decision-changing utility test, not an attempt to relabel its baseline as novel.

Cost: sequentialGPU4, four768-env fresh batches,<=3600s/2GiB. Compare native
actor, random correction, global fit-response selection and conditional factual
selection at the same current proximity events and bounded correction budget.
If its task gate fails, stop this controller and reconsider the one-step
objective/contact-time representation. If it passes, test broader object/task
coverage and a distinctive method under matched budgets. Stop incomplete cohorts,
input/source drift or resource overrun; preserve failures. No new external
authorization is needed within the isolated worktree and campaign boundaries.
