# Gate 1 corrected outcome-coverage probe

**当前需要决定的问题。** history preceding-action 合同修复后，e260 的 h16 I+
仍在 split 3 反向、corrected e420 也未复现；是否只是现有 success/drop outcome
覆盖太稀疏。

**关键证据。** corrected e260 direct h16 为 `+37.8,+20.6,-28.2,+21.2,+6.2%`，
corrected e420 为 `+3.0,+7.4,-19.8,+3.1,-22.8%`。e260 split 3 的 held-out
success/drop episode `(2,18640000018)` 单独造成约 `-231` MAE delta；现有 112
episodes 中只有极少同时满足 stable-success/drop 的 episode。

**root 选择的行动及理由。** 在同一 pinned e260 checkpoint、同一 collector、同一
pre-step raw contract 下并行采集四个独立 run（seed namespaces 1866--1869，每个
目标 14k rows），然后只做 history-fixed h16/I+ offline assembly、audit 和固定
五个 split fit。这样增加 outcome 覆盖而不改变模型、目标或 Gate 定义。

**预计成本、成功/失败后的下一步和停止条件。** 四 GPU 并行采集目标不超过 15 min，
随后组装和 fit 约 10 min；不生成 policy checkpoint。若新增覆盖后 split 仍由单一
episode 主导，停止扩大 e260 collector，转为正式 actor/outcome cluster Validation
设计；若敏感性显著下降，只记录为 `PROMISING`，仍不启动 Cm，直到 matched actor
Validation。

**外部授权边界。** 只读取 pinned checkpoint、配置和 motion 数据，在项目 `tmp/`
写入新 raw shards；不修改外部项目、已有 raw shards 或未知 GPU 进程。

**结果。** 四个新 corrected run 完成 56 episodes、60,756 rows；其中 1 个
`stable_success`、0 个 `drop_after_success`。与既有四个 run 合并后得到 224
episodes、116,067 h16 windows，base dataset 和独立 interaction augmentation audit
均通过。五个 fixed split 的 direct I+ `V_HEI`/`V_H` 改善为
`+12.6%`、`+5.2%`、`+18.3%`、`+26.9%`、`+14.9%`；episode bootstrap CI 中
4/5 排除零。matched future-action control `V_HFEI`/`V_HF` 为
`+15.9%`、`+18.9%`、`+25.3%`、`+25.3%`、`+40.7%`，5/5 CI 排除零。
按 8 个 source-run 聚类的 direct CI 只有 split 1 跨零，control 的 5 个 CI 均为正。

这说明增加独立普通失败和一个 stable-success 后，原 split 3 的极端反向影响不再
主导；I+ 的方向性从旧的 split-sensitive `PROMISING` 变为更一致的
`PROMISING` coverage probe。但该数据扩展是在探索阶段作出的，success/drop 覆盖
仍不平衡，不能直接写成 `SUPPORTED` 或启动 Cm；下一步应冻结 split、actor/outcome
cluster 单位和 matched control 后再做正式 Validation。
