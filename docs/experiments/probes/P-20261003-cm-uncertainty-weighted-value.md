# P-20261003-cm-uncertainty-weighted-value

Family: Cm uncertainty-guided value allocation
Type: Offline value-fitting Decision Probe
Status: RUNNING

## Question

Can Cm ensemble disagreement improve direct-Q learning by allocating more loss weight to
hard dynamics rows, while the task-value target remains the real complete return?

## Fixed contract

- The frozen physical ensemble supplies only translation/velocity/contact/reward/terminal
  disagreement; orientation is excluded as in HF28.
- A direct-Q copy starts from the frozen direct-Q checkpoint.
- Both arms use identical fit rows, real complete-return targets, optimizer, updates and
  initialization. The control uses weight `1.0` for every row. The treatment uses weight `2.0`
  for the fit top-20% uncertainty rows and `1.0` otherwise.
- Actor input, actor action, reward, success definition and native data are unchanged.
- No synthetic target, policy training, threshold scan, coefficient scan or extra data is used.

## Predeclared held gates

The treatment is useful only if held top-20% direct-Q residual RMSE improves by at least `5%`,
overall RMSE and MAE do not worsen by more than `1%`, and episode Spearman is no lower than
`0.01` below frozen direct-Q. Failure closes this route without a policy follow-up.

Evidence is written to the paired JSON result and the immutable source hashes are recorded there.
