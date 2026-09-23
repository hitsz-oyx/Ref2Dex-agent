# V1.33 用高抬升策略真实转移适配低延迟 CmLite

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- run_status: `PLANNED`
- official actor checkpoint: `null`

V1.32 的 Cm-off e160 seed74 `eval_s74_e160_full/transitions.pt`
将筛为首 episode 非终止训练输入；seed75 的同一自训练策略评估
只用于独立门禁。训练还使用 V1.29 四个原始训练源，模型结构与
超参不变。完成后用 `analyze_cmlite_on_policy.py` 比较旧/新模型，
并用 `train_cmlite.py` 的 s1 外部验证检查遗忘。具体阈值见
experiment card。
