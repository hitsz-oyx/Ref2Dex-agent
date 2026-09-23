# V1.32 s3 微调策略分布上的 CmLite 一步预测审计

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- run_status: `PLANNED`
- official actor checkpoint: `null`

输入仍为 s3 `corrected_manifest_r2.json` 与对应 motion root，
两枚自训练 V1.30 e160 checkpoint 的训练 manifest 均为
`COMPLETED`。`eval_dexplore_full_episode_grid.py --save-transitions`
将保存逐步转移并在评估 manifest 中记录 SHA256；
`analyze_cmlite_on_policy.py` 对第一 episode 非终止转移计算冻结
CmLite 指标。新增筛选逻辑测试 `7 passed`。详细假设及停损门槛
见对应 experiment card。
