# Decision Memo — HF05 selective causal gate 之后（2026-09-26）

HF05 是在用户重新授权后建立的独立高层假设：让 Cm 在不确定时 abstain，
只在 fit-only 不确定性界支持安全增益时介入。既有三臂记录的 CPU-only
判别运行完成，输入和 finite checks 通过，但 selective policy 只在 1/126
个 holdout 状态介入，held-lift 没有超过 always-base，未通过预先声明的
coverage/policy gate。

处置：HF05 标为 `UNPROMISING` 并冻结。HF01–HF04 和 HF05 都没有通过 Cm
policy-utility Probe 门槛；当前不启动新的 Cm 实验。未来若要改变路线，需由
新的高层 goal 和 Decision Checkpoint 明确授权。
